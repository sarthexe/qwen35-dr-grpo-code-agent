"""Phase 2 verification: multi-turn rollout collection and the Dr. GRPO engine.

Runs fast CPU verification by default (no model download, no GPU) and supports
``--use-qwen35`` for a live smoke test on a Colab T4 with Qwen/Qwen3.5-4B.

Checks:
  1. ``<think>\n`` chat-template prefill alignment survives grading
  2. multi-turn ``action_mask`` is exactly 1 on policy tokens, 0 elsewhere
  3. ``selective_per_token_logps`` matches a naive ``F.log_softmax``
  4. Dr. GRPO produces gradients on differing rewards and skips dead groups

Usage:
    python scripts/verify_phase2.py
    python scripts/verify_phase2.py --use-qwen35
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: F401  (puts src/ on sys.path for a fresh clone)

from dr_grpo.dataset import load_mbppplus_splits
from dr_grpo.grpo_engine import (
    compute_dr_grpo_loss,
    dr_grpo_train_step,
    load_qwen35_4bit,
    selective_per_token_logps,
)
from dr_grpo.rollout import collect_group_rollouts
from mocks import CharWordTokenizer, TinyAdapterCausalLM

def verify_local() -> None:
    print("=== Phase 2 Local Verification ===")
    splits = load_mbppplus_splits(seed=42)
    # Use `is_not_prime` (task_id=3) for clean passing + partial-pass verification
    all_probs = splits["train"] + splits["test"]
    problem = next(p for p in all_probs if p.entry_point == "is_not_prime")
    print(f"Using problem task_id={problem.task_id} ({problem.entry_point})")

    tokenizer = CharWordTokenizer()
    model = TinyAdapterCausalLM(vocab_size=tokenizer.vocab_size)

    # Note: NO leading `<think>` in model output — simulates Qwen3.5 `<think>\n` prefill!
    passing_resp = (
        "Check divisors up to sqrt(n).\n</think>\n"
        f"<patch>\n{problem.canonical_solution.strip()}\n</patch><|im_end|>"
    )
    partial_buggy_resp = (
        "Check even numbers only.\n</think>\n"
        "<patch>\ndef is_not_prime(n):\n    if n <= 2:\n        return False\n    return n % 2 == 0\n</patch><|im_end|>"
    )
    bad_format_resp = "def is_not_prime(n): return False<|im_end|>"
    repaired_resp = (
        "Aha! Test failed on n=1 or odd composite. Fixing loop.\n</think>\n"
        f"<patch>\n{problem.canonical_solution.strip()}\n</patch><|im_end|>"
    )

    # Turn 1: traj 0 passes (reward=1.0), traj 1 partial, traj 2 partial, traj 3 bad format
    model.scripted_turn1 = [
        tokenizer.encode(passing_resp),
        tokenizer.encode(partial_buggy_resp),
        tokenizer.encode(partial_buggy_resp),
        tokenizer.encode(bad_format_resp),
    ]
    # Turn 2 (called on failing trajectories [1, 2, 3]):
    # traj 1 repairs completely, traj 2 repeats buggy code (duplicate penalty), traj 3 still fails format
    model.scripted_turn2 = [
        tokenizer.encode(repaired_resp),
        tokenizer.encode(partial_buggy_resp),
        tokenizer.encode(bad_format_resp),
    ]

    batch = collect_group_rollouts(
        model=model,
        tokenizer=tokenizer,
        problem=problem,
        group_size=4,
        max_turns=2,
        max_new_tokens=256,
    )

    # 1. Verify <think>\n prefill alignment & rewards
    print(f"Rewards: {batch.rewards.tolist()}")
    print(
        f"turn1_pass_rate={batch.turn1_pass_rate:.2f}, "
        f"turn2_pass_rate={batch.turn2_pass_rate:.2f}, "
        f"format_compliance={batch.format_compliance_rate:.2f}, "
        f"avg_think_tokens={batch.avg_think_tokens:.1f}"
    )
    assert abs(batch.rewards[0].item() - 1.0) < 1e-5, "Turn-1 pass should score 1.0"
    assert abs(batch.rewards[1].item() - 0.90) < 1e-5, "Turn-2 repair should score 0.90"
    assert batch.turn1_results[0].has_think and batch.turn1_results[0].all_passed
    assert batch.completions_turn2[0] is None
    assert batch.final_results[2].duplicate_code_penalty_applied
    assert batch.rewards[3].item() == 0.0
    assert abs(batch.turn1_pass_rate - 0.25) < 1e-6
    assert abs(batch.turn2_pass_rate - 0.50) < 1e-6

    # 2. Verify action_mask exactness on trajectory 0 (1-turn) and trajectory 1 (2-turn)
    for idx in range(4):
        attn = batch.attention_mask[idx]
        act = batch.action_mask[idx]
        # Padding positions must have action_mask == 0
        assert torch.all(act[attn == 0] == 0), "Padding must have action_mask == 0"

    # Trajectory 0: [prompt_1 (0s)] + [gen_1 (1s)] + [pad (0s)]
    ids_0 = batch.input_ids[0].tolist()
    act_0 = batch.action_mask[0].tolist()
    gen_0_tokens = [t for t, m in zip(ids_0, act_0) if m == 1]
    assert tokenizer.decode(gen_0_tokens) == passing_resp, (
        f"Traj 0 action_mask==1 decoded mismatch: {tokenizer.decode(gen_0_tokens)!r}"
    )

    # Trajectory 1: [prompt_1 (0s)] + [gen_1 (1s)] + [obs_wrapper (0s)] + [gen_2 (1s)] + [pad (0s)]
    ids_1 = batch.input_ids[1].tolist()
    act_1 = batch.action_mask[1].tolist()
    attn_1 = batch.attention_mask[1].tolist()
    # Find contiguous spans where act_1 == 1
    spans: List[List[int]] = []
    cur: List[int] = []
    obs_spans: List[List[int]] = []
    cur_zero: List[int] = []
    seen_one = False
    for tid, a_m, att_m in zip(ids_1, act_1, attn_1):
        if att_m == 0:
            break
        if a_m == 1:
            if cur_zero and seen_one:
                obs_spans.append(cur_zero)
                cur_zero = []
            seen_one = True
            cur.append(tid)
        else:
            if cur:
                spans.append(cur)
                cur = []
            if seen_one:
                cur_zero.append(tid)
    if cur:
        spans.append(cur)

    assert len(spans) == 2, f"Expected 2 policy action spans in Turn-2 traj, got {len(spans)}"
    assert len(obs_spans) == 1, f"Expected 1 <obs> mask==0 gap between turns, got {len(obs_spans)}"
    assert tokenizer.decode(spans[0]) == partial_buggy_resp
    assert tokenizer.decode(spans[1]) == repaired_resp
    obs_text_decoded = tokenizer.decode(obs_spans[0])
    assert "<obs>" in obs_text_decoded and "</obs>" in obs_text_decoded
    assert "<|im_start|>assistant\n<think>\n" in obs_text_decoded
    print("Check 1 & 2 passed: <think> alignment and multi-turn action_mask exact.")

    # 3. Verify selective_per_token_logps matches naive F.log_softmax within 1e-5
    model.eval()
    with torch.no_grad():
        sel_logps = selective_per_token_logps(
            model, batch.input_ids, batch.attention_mask
        )
        full_logits = model(batch.input_ids, attention_mask=batch.attention_mask).logits[:, :-1, :]
        naive_logps = (
            F.log_softmax(full_logits.float(), dim=-1)
            .gather(-1, batch.input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
    max_diff = float((sel_logps - naive_logps).abs().max().item())
    print(f"selective_per_token_logps max abs diff vs naive: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Logps mismatch: {max_diff}"
    print("Check 3 passed: selective_per_token_logps matches naive F.log_softmax.")

    # 4. Verify Dr. GRPO gradients when rewards differ & dead-group skip when identical
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    before_lora_b = model.lora_B.weight.detach().clone()

    # Fixed non-trivial rewards [1.0, 0.2, 0.6, 0.2]
    test_batch = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "action_mask": batch.action_mask,
        "rewards": torch.tensor([1.0, 0.2, 0.6, 0.2], dtype=torch.float32),
    }
    stats_active = dr_grpo_train_step(
        model=model,
        optimizer=optimizer,
        rollout_batch=test_batch,
        micro_batch_size=2,
        max_gen_tokens=768,
    )
    after_lora_b = model.lora_B.weight.detach().clone()
    param_delta = float((after_lora_b - before_lora_b).abs().max().item())
    print(
        f"Active step stats: skipped={stats_active['skipped']}, "
        f"loss={stats_active['loss']:.6f}, param_delta={param_delta:.2e}"
    )
    assert not stats_active["skipped"]
    assert param_delta > 0.0, "Expected non-zero parameter update on active group"

    # Verify dead-group skip with identical rewards [0.15, 0.15, 0.15, 0.15]
    before_dead = model.lora_B.weight.detach().clone()
    dead_batch = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "action_mask": batch.action_mask,
        "rewards": torch.tensor([0.15, 0.15, 0.15, 0.15], dtype=torch.float32),
    }
    stats_dead = dr_grpo_train_step(
        model=model,
        optimizer=optimizer,
        rollout_batch=dead_batch,
        micro_batch_size=2,
        max_gen_tokens=768,
    )
    after_dead = model.lora_B.weight.detach().clone()
    dead_delta = float((after_dead - before_dead).abs().max().item())
    print(f"Dead group stats: skipped={stats_dead['skipped']}, param_delta={dead_delta:.2e}")
    assert stats_dead["skipped"] is True
    assert dead_delta == 0.0, "Dead group must skip backward & optimizer step"

    # Verify Dr. GRPO fixed max_gen_tokens normalization (no length bias)
    ones_mask = torch.ones((1, 5), dtype=torch.long)
    loss_768, _ = compute_dr_grpo_loss(
        per_token_logps=torch.full((1, 4), -1.0),
        old_logps=torch.full((1, 4), -1.0),
        ref_logps=torch.full((1, 4), -1.0),
        action_mask=ones_mask,
        advantages=torch.tensor([1.0]),
        max_gen_tokens=768,
    )
    # 4 active tokens * 1.0 advantage / 768 -> -4 / 768
    expected_loss = -4.0 / 768.0
    assert abs(float(loss_768.item()) - expected_loss) < 1e-6

    print("\nALL PHASE 2 LOCAL VERIFICATION CHECKS PASSED.")


def verify_qwen35() -> None:
    print("=== Phase 2 Live Qwen/Qwen3.5-4B Verification ===")
    splits = load_mbppplus_splits(seed=42)
    problem = splits["train"][0]
    model, tokenizer = load_qwen35_4bit()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-6
    )

    batch = collect_group_rollouts(
        model=model,
        tokenizer=tokenizer,
        problem=problem,
        group_size=4,
        max_turns=2,
        max_new_tokens=768,
    )
    print(
        f"Live rollout: shape={tuple(batch.input_ids.shape)}, "
        f"rewards={batch.rewards.tolist()}, "
        f"t1_pass={batch.turn1_pass_rate:.2f}, t2_pass={batch.turn2_pass_rate:.2f}"
    )

    with torch.no_grad():
        sel_logps = selective_per_token_logps(
            model, batch.input_ids[:1], batch.attention_mask[:1]
        )
        assert sel_logps.shape == (1, batch.input_ids.size(1) - 1)

    step_stats = dr_grpo_train_step(
        model=model,
        optimizer=optimizer,
        rollout_batch=batch,
        micro_batch_size=1,
        max_gen_tokens=768,
    )
    print(f"Live Dr. GRPO step stats: {step_stats}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak CUDA VRAM: {peak_gb:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-qwen35",
        action="store_true",
        help="Run live verification with Qwen/Qwen3.5-4B 4-bit LoRA on GPU.",
    )
    args = parser.parse_args()
    if args.use_qwen35:
        verify_qwen35()
    else:
        verify_local()


if __name__ == "__main__":
    main()
