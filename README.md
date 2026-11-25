# Qwen3.5-4B Multi-Turn Code Reasoning Agent — Dr. GRPO in Pure PyTorch

A from-scratch implementation of **Dr. GRPO** (Group Relative Policy Optimization, *Done Right*) for a
multi-turn, self-healing Python code agent, trained on `evalplus/mbppplus` with `Qwen/Qwen3.5-4B`
in 4-bit NF4 + LoRA — on a **single 16 GB Colab T4**.

No TRL. No `GRPOTrainer`. The rollout loop, reward function, advantage computation, clipped
surrogate objective, KL estimator, and optimizer step are all written directly against PyTorch.

```
mbppplus problem ──► Turn-1 rollout (G=4) ──► sandbox grade ──► <obs> traceback ──► Turn-2 rollout
                                                                       │
                          Dr. GRPO update ◄── per-token logps ◄─────────┘
                          (mean-centred advantage, asymmetric clip, k3 KL, /768 normalisation)
```

---

## 1. What makes this non-trivial

| Constraint | Naive approach | What this repo does |
|---|---|---|
| `vocab_size = 248,320` | `F.log_softmax` over `(B, T, V)` fp32 — ~3 GB per sequence at `T≈1536`, doubled by the softmax output | Row-wise `selective_per_token_logps`: one sequence at a time, so peak is independent of `G` |
| Reference policy | Load a second 4-bit model (+3 GB) | `with model.disable_adapter(), torch.no_grad():` on the **same** weights — **0 GB** |
| 16 GB T4, 4-bit weights | `micro_batch=2`, ~2.45 GB logit spike → allocator OOM | `micro_batch=1` + gradient accumulation over the group (`grad_accum = 4`) |
| Generation needs KV cache, training must not keep it | One mode for both | Explicit `set_generation_mode()` / `set_training_mode()` toggles around each phase |
| Credit assignment across turns | Train on the concatenated string | Exact `action_mask`: `1` **only** on policy tokens, `0` on prompts, `<obs>` repairs, and padding |
| Length bias in GRPO | Divide by `|o_i|` → longer wrong answers are penalised less | Fixed constant `MAX_GEN_TOKENS = 768` (Dr. GRPO) |
| Untrusted model output | `exec()` in-process | AST validation + isolated `python -I` subprocess with a 2 s timeout |

---

## 2. The objective, written out

For a group of `G = 4` rollouts per prompt, with shaped reward `r_i ∈ [0, 1]`:

**Advantage (Dr. GRPO — mean-centred only, no std division):**

```
A_i = r_i − mean(r)          # deliberately *not* divided by std(r)
```

**Asymmetric PPO clipping** — the penalty side is tighter than the reward side, so a bad update is
corrected faster than a good one is reinforced:

```
ρ_i,t      = exp(log π_θ(y_t) − log π_old(y_t))
clipped    = clamp(ρ, 1 − 0.20, 1 + 0.28)
surrogate  = min(ρ · A, clipped · A)
```

**Unbiased k3 KL estimator (Schulman)** — non-negative, low variance, no need for the reference
distribution's entropy:

```
KL_t = exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1
```

**Per-trajectory loss with fixed-constant normalisation** (the "Done Right" part):

```
L_i = − (1 / 768) · Σ_t [ surrogate_i,t − β · KL_i,t ] · m_i,t        β = 0.02
```

where `m_i,t = action_mask[i, t+1]` after the causal shift. Dividing by a *constant* instead of
`|o_i|` removes the incentive to pad a wrong answer with tokens, and keeps the gradient scale
comparable across rollouts of different lengths.

**Dead-group fast path:** if `max(r) − min(r) < 1e-5`, every advantage is zero, so the group carries
no signal — the step is skipped entirely (no forward, no backward) and counted in `dead_group_ratio`.
This is the dominant compute saving on easy problems, where the model is already correct 4/4 times.

---

## 3. VRAM budget on a T4 (16 GB)

| Component | Cost | Mitigation |
|---|---|---|
| 4-bit NF4 weights (4B params) | ≈ 2.3 GB | `bnb_4bit_quant_type="nf4"`, double quant |
| LoRA adapters (r=16, 7 projections) | ≈ 40 MB | `r=16, α=32, dropout=0` |
| KV cache during generation | ≈ 1–2 GB (`G=4`, 768 tok) | Enabled **only** for rollout/eval |
| One logits row, `T=1536` | 0.76 GB bf16 / 1.53 GB fp32 | Never upcast the whole batch |
| Full `(B,T,V)` fp32 log-softmax, `B=2` | **≈ 7.6 GB transient** → OOM spike | `micro_batch=1` + row-wise log-softmax |
| Activations | ≈ 2–3 GB | `gradient_checkpointing` enabled only during training |

The row-wise loop bounds the log-prob transient at one sequence (~3.8 GB peak at `T=1536`, including
the fp32 upcast and the softmax output) instead of scaling with `G`, and because `old_logps` and
`ref_logps` are computed under `no_grad`, no autograd graph is retained across rows.

Result: peak stays under ~11 GB, leaving headroom for the allocator's fragmentation.

---

## 4. Multi-turn self-healing and the action mask

Turn 1 fails → the harness returns the first failing assertion with its traceback, which is injected
as an observation, and the model gets one repair attempt:

```
<|im_start|>user
Solve: <problem> … <|im_end|>
<|im_start|>assistant
… </think><patch>def f(…)</patch>            ← action_mask = 1
<|im_end|>
<|im_start|>user
<obs>Failed at test #3/108: AssertionError …</obs>
Analyze the test failure above and output the corrected code.<|im_end|>   ← action_mask = 0
<|im_start|>assistant
… </think><patch>def f(…)</patch>            ← action_mask = 1
<|im_end|>  [pad][pad]                        ← action_mask = 0
```

This is the subtle part of multi-turn RLVR: the `<obs>` block is *context*, not *policy output*.
Masking it out means the gradient only credits tokens the model actually chose, and the observation
never gets reinforced as if the model had written the traceback itself.

---

## 5. Reward design

Shaped reward in `[0.0, 1.0]`, computed by `sandbox.grade_completion`:

| Term | Value | Condition |
|---|---|---|
| Missing ` thinking…</think>` or code block | **0.00** (total) | hard gate |
| Format compliance | **+0.15** | both blocks present and non-empty |
| Partial test credit | **+0.75 × passed / total** | per-assertion, ~107 assertions/problem |
| Turn-1 completion bonus | **+0.10** | 100 % of assertions pass on the first attempt |
| Duplicate-code penalty | **−0.10** | Turn 2 reproduces Turn 1's code verbatim |

Assertions are unrolled from the `evalplus/mbppplus` `test` field (which packs ~107 cases into a
`for i, (inp, exp) in enumerate(zip(inputs, results))` loop) into **individual statements**, each
wrapped in its own `try/except` inside a single subprocess run. That yields continuous partial
credit instead of a binary pass/fail, which is what makes the reward useful as a learning signal.

Security: every candidate solution is AST-validated against blocked modules (`os`, `sys`,
`subprocess`, `shutil`, `socket`, `urllib`, `http`, `pathlib`, `importlib`) and builtins (`eval`,
`exec`, `open`, `compile`, `__import__`), then executed as `python -I` in a temp file with a 2.0 s
wall-clock timeout. Infinite loops are killed and graded, not crashed.

---

## 6. Dataset

`evalplus/mbppplus`, `test` split — 378 sanitized problems with edge-case-heavy assertions.

| Split | Size | Purpose |
|---|---|---|
| `train` | 250 | 1 epoch = 250 Dr. GRPO steps |
| `test` | 128 | held-out benchmark (pre- vs post-RL) |
| `val` | 50 | first 50 of `test`, fast mid-training check every 50 steps |

Split is deterministic (`seed=42`), and `val ⊂ test` by construction so the mid-training signal is
measured on genuinely held-out problems.

---

## 7. Quickstart

```bash
git clone https://github.com/sarthexe/qwen35-dr-grpo-code-agent.git
cd qwen35-dr-grpo-code-agent
pip install -r requirements.txt
pip install -e . --no-build-isolation        # optional: enables `python -m dr_grpo.*`
```

**Verify without a GPU** (no model download, ~10 s, exercises the whole pipeline on CPU mocks):

```bash
python scripts/verify_phase1.py    # dataset splits, assertion extraction, sandbox reward
python scripts/verify_phase2.py    # rollout masking, log-prob equivalence, Dr. GRPO step
python scripts/verify_phase3.py    # base eval → train → checkpoint → resume → compare
```

**Live smoke test on a T4** (2 steps of the real model, real generation):

```bash
python scripts/verify_phase3.py --use-qwen35
```

**Full run** (250 steps, pre-RL baseline on the 128 held-out problems, post-RL benchmark + table):

```bash
python -m dr_grpo.train --run-baseline-eval
```

**Benchmark only:**

```bash
python -m dr_grpo.eval_benchmark --mode base    --split test
python -m dr_grpo.eval_benchmark --mode trained --split test --adapter-path checkpoints/dr_grpo_final
python -m dr_grpo.eval_benchmark --mode compare
```

---

## 8. Repository layout

```
src/dr_grpo/
├── dataset.py         mbppplus loading, assertion unrolling, prompt formatting, 250/50/128 split
├── sandbox.py         AST validator, subprocess test harness, shaped reward
├── rollout.py         multi-turn rollout collection, action_mask construction, RolloutBatch
├── grpo_engine.py     4-bit loader, selective_per_token_logps, Dr. GRPO loss + train step
├── eval_benchmark.py  batched eval, benchmark metrics, Markdown comparison table
└── train.py           training loop, telemetry, checkpointing, resume
scripts/
├── verify_phase1.py   dataset + sandbox
├── verify_phase2.py   rollout + engine
├── verify_phase3.py   end-to-end train + benchmark
├── mocks.py           deterministic CPU test doubles (tokenizer, adapter LM, scripted LM)
└── _bootstrap.py      puts src/ on sys.path for a fresh clone
docs/architecture.md   deeper design notes, tensor shapes, sequence diagrams
benchmarks/            JSON reports (base / trained) + schema documentation
```

---

## 9. Verification status

All three suites pass locally on CPU with no GPU and no model download:

| Check | Result |
|---|---|
| Split sizes | `train=250, val=50, test=128`, `val ⊂ test` |
| Assertions per problem | min 3, max 150, mean ≈ 107.5 |
| Full-pass reward | `1.0000` (108/108 assertions) |
| Partial credit | `0.6153` at 67/108 (expected `0.15 + 0.75·67/108`) |
| `import os` blocked | `0.1500`, `SecurityError` surfaced in `<obs>` |
| `while True:` | killed at `2.03 s`, `TimeoutError` in `<obs>` |
| `selective_per_token_logps` vs naive `F.log_softmax` | max abs diff `0.00e+00` |
| `action_mask` | exactly `1` on Turn-1/Turn-2 policy tokens, `0` on prompt, `<obs>`, padding |
| Dead group `[0.15]×4` | `skipped=True`, parameter delta `0.00e+00` |
| Active group `[1.0, 0.2, 0.6, 0.2]` | backward runs, LoRA weights move |
| Checkpoint / resume | `step_2` saved, resumed to step 3, `train_metrics.jsonl` appended |

> These numbers come from the mock-model harness in `scripts/`, which exists precisely so the
> pipeline is testable without a GPU. They validate the *mechanics*, not model quality.

---

## 10. Results (filled in by the Colab run)

`benchmarks/base_qwen3_5_4b.json` and `benchmarks/dr_grpo_qwen3_5_4b.json` are generated by the run
below; `--mode compare` renders this table from them.

```bash
python -m dr_grpo.train --run-baseline-eval
```

| Metric | Base Qwen3.5-4B | Dr. GRPO Qwen3.5-4B | Delta |
|---|---:|---:|---:|
| Turn-1 Pass@1 (%) | — | — | — |
| Turn-2 Self-Healed Pass@1 (%) | — | — | — |
| Self-Healing Recovery Gain (%) | — | — | — |
| Mean Assertion Pass Ratio (%) | — | — | — |
| Format Compliance (%) | — | — | — |
| Avg `think` Token Length | — | — | — |

---

## 11. Design decisions worth defending

**Why mean-centre without dividing by std?** Dividing by the group std amplifies noise when the
group is nearly uniform (e.g. three rollouts at 0.9 and one at 0.85), producing large advantages from
reward differences that are essentially grader jitter. Dr. GRPO drops the division; combined with the
dead-group skip, near-uniform groups contribute nothing instead of contributing noise.

**Why `/768` instead of `/(|o_i|)`?** Per-sequence normalisation makes a 700-token wrong answer cost
the same as a 70-token wrong answer, so the model can dilute a negative advantage by rambling. A
fixed denominator keeps the penalty proportional to how much *bad text* was produced.

**Why k3 KL?** It is unbiased and non-negative per token, so `β` behaves like a genuine penalty
rather than a noisy regulariser. It also needs only `log π_ref` and `log π_θ`, which we already have.

**Why a subprocess instead of `exec`?** Model output is untrusted. AST filtering stops obvious
escapes, but a 2 s isolated process is the real containment boundary: it survives segfaults, runaway
loops, and `sys.exit()`, and it cannot touch the trainer's state.

**Why `disable_adapter()` for the reference?** A second 4-bit copy costs ~3 GB that a T4 does not
have. PEFT's context manager gives the exact base-model log-probs from the same weights, so the KL
term is computed against the true reference with zero extra VRAM.

---

## 12. Limitations & next steps

- Single-GPU, single-process; no vLLM/continuous batching, so rollout throughput is the bottleneck.
- `avg@k` evaluation is sequential per sample (simple, but slow at k=4 on 128 problems).
- Reward is assertion-count based; it does not model runtime complexity or code style.
- No curriculum: the 250 training problems are shuffled uniformly each epoch.
- Next: reward-hacking probes (does partial credit reward degenerate one-liners?), longer
  self-healing chains (`max_turns=3`), and a CI job that runs the CPU suite on every push.

## License

MIT — see [LICENSE](LICENSE).
