# Benchmark reports

Two JSON reports are produced by `dr_grpo.eval_benchmark` and consumed by
`dr_grpo.eval_benchmark --mode compare`:

| File | Produced by | Meaning |
|---|---|---|
| `base_qwen3_5_4b.json` | `--mode base` | `Qwen/Qwen3.5-4B` evaluated through `model.disable_adapter()` |
| `dr_grpo_qwen3_5_4b.json` | `--mode trained` | the same base weights + the trained LoRA adapter |

Both are evaluated on the **128 held-out `test` problems**, greedy (`temperature=0.0`), with
`max_turns=2`. An `Avg@4` variant (`--temperature 0.2 --avg-k 4`) can be written to a third file.

Generate them:

```bash
python -m dr_grpo.eval_benchmark --mode base    --split test
python -m dr_grpo.eval_benchmark --mode trained --split test --adapter-path checkpoints/dr_grpo_final
python -m dr_grpo.eval_benchmark --mode compare
```

Or in one shot, as part of training:

```bash
python -m dr_grpo.train --run-baseline-eval
```

---

## Schema

```jsonc
{
  "mode": "base",                  // "base" | "trained"
  "num_problems": 128,
  "avg_k": 1,
  "temperature": 0.0,
  "max_turns": 2,
  "metrics": {
    "turn1_pass_at_1": 0.0,             // % of problems passing 100% of assertions on Turn 1
    "turn2_pass_at_1": 0.0,             // % passing after Turn-2 self-healing
    "self_healing_gain": 0.0,           // turn2_pass_at_1 - turn1_pass_at_1
    "mean_assertion_pass_ratio": 0.0,   // % mean of passed/total across all problems
    "format_compliance_rate": 0.0,      // % with valid  thinking + code block
    "avg_think_tokens": 0.0,
    "avg_latency_sec": 0.0              // wall clock per problem
  },
  "total_elapsed_sec": 0.0,
  "per_problem": [
    {
      "problem_index": 0,
      "sample_index": 0,
      "task_id": 410,
      "entry_point": "min_val",
      "turn1_passed": false,
      "turn2_passed": true,
      "self_healed": true,
      "passed_assertions": 107,
      "total_assertions": 107,
      "assertion_pass_ratio": 1.0,
      "format_valid": true,
      "think_tokens": 143,
      "reward": 0.9,
      "security_violation": null,
      "timed_out": false,
      "first_failure": "",
      "turn1_completion": "…first 500 chars of the Turn-1 output…",
      "turn2_completion": "…first 500 chars of the Turn-2 output, or null…"
    }
  ]
}
```

`turn1_completion` / `turn2_completion` are truncated to 500 characters on purpose: enough to see
whether the model rambled, emitted a malformed patch, or got cut off mid-`</think>`, without turning
a 128-problem report into a multi-megabyte file. `turn2_completion` is `null` when the problem was
already solved on Turn 1.

`first_failure` holds the sandbox's first failing assertion plus traceback for unsolved problems —
that is exactly the string the model would have seen inside `<obs>`.
