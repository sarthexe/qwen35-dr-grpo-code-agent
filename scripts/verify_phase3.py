"""Phase 3 verification: end-to-end training and benchmarking.

Runs a fast offline smoke test (no model download, no GPU) covering:
  1. base-model evaluation through ``disable_adapter()``
  2. a 3-step Dr. GRPO loop with checkpointing, JSONL telemetry and ``--resume-from``
  3. trained-model evaluation plus the Markdown comparison table

Supports ``--use-qwen35`` for a live smoke test on a Colab T4.

Usage:
    python scripts/verify_phase3.py
    python scripts/verify_phase3.py --use-qwen35
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: F401  (puts src/ on sys.path for a fresh clone)

from dr_grpo.dataset import load_mbppplus_splits
from dr_grpo.eval_benchmark import evaluate_split, print_comparison_table
from dr_grpo.grpo_engine import load_qwen35_4bit
from dr_grpo.train import run_training
from mocks import CharWordTokenizer, DynamicMockCausalLM

def verify_local() -> None:
    print("=== Phase 3 Local End-to-End Verification ===")
    splits = load_mbppplus_splits(seed=42)
    # Pick 4 deterministic real problems with unique entry points
    small_train = splits["train"][:4]
    small_val = splits["val"][:4]
    small_test = splits["test"][:4]
    all_small = small_train + small_val + small_test

    tokenizer = CharWordTokenizer()
    model = DynamicMockCausalLM(tokenizer=tokenizer, problems=all_small)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_json = os.path.join(tmpdir, "benchmarks", "base_qwen3_5_4b.json")
        trained_json = os.path.join(tmpdir, "benchmarks", "dr_grpo_qwen3_5_4b.json")
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        log_path = os.path.join(tmpdir, "logs", "train_metrics.jsonl")

        # 1. Base model evaluation via disable_adapter()
        base_report = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            problems=small_test,
            mode="base",
            eval_batch_size=2,
            max_turns=2,
            max_new_tokens=256,
            temperature=0.0,
            output_path=base_json,
        )
        print(f"Base report metrics: {base_report['metrics']}")
        assert os.path.exists(base_json)
        assert base_report["metrics"]["turn1_pass_at_1"] == 0.0
        assert base_report["metrics"]["turn2_pass_at_1"] == 25.0
        assert base_report["metrics"]["self_healing_gain"] == 25.0
        for rec in base_report["per_problem"]:
            assert isinstance(rec["turn1_completion"], str)
            assert len(rec["turn1_completion"]) <= 500
            assert rec["turn2_completion"] is not None
            assert len(rec["turn2_completion"]) <= 500

        # 2. Run 3-step Dr. GRPO training loop (eval_every=2, save_every=2)
        train_args = SimpleNamespace(
            model_id="mock",
            epochs=1,
            max_steps=3,
            group_size=4,
            max_turns=2,
            max_new_tokens=256,
            micro_batch_size=2,
            lr=1e-3,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            eps_low=0.20,
            eps_high=0.28,
            beta=0.02,
            eval_every=2,
            eval_batch_size=2,
            save_every=2,
            run_baseline_eval=False,
            resume_from=None,
            output_dir=ckpt_dir,
            log_path=log_path,
            base_report=base_json,
            trained_report=trained_json,
            seed=42,
        )
        small_splits = {"train": small_train, "val": small_val, "test": small_test}
        train_summary = run_training(
            train_args, model=model, tokenizer=tokenizer, splits=small_splits
        )
        print(f"Training summary: {train_summary}")

        step2_ckpt = os.path.join(ckpt_dir, "step_2")
        final_ckpt = os.path.join(ckpt_dir, "dr_grpo_final")
        assert os.path.exists(os.path.join(step2_ckpt, "trainer_state.json"))
        assert os.path.exists(os.path.join(step2_ckpt, "optimizer.pt"))
        assert os.path.exists(os.path.join(step2_ckpt, "scheduler.pt"))
        assert os.path.exists(os.path.join(final_ckpt, "trainer_state.json"))
        assert train_summary["last_val_report"] is not None

        with open(log_path, "r", encoding="utf-8") as f:
            log_lines = [json.loads(line) for line in f if line.strip()]
        assert len(log_lines) == 3, f"Expected 3 log entries, got {len(log_lines)}"
        assert [row["step"] for row in log_lines] == [1, 2, 3]
        assert all(not row["dead_group_skipped"] for row in log_lines)

        # Verify generation/training mode toggles occurred cleanly
        assert False in model.gradient_checkpointing_calls
        assert True in model.gradient_checkpointing_calls

        # 2b. Verify --resume-from step_2 executes only step 3 and appends
        resume_args = SimpleNamespace(**vars(train_args))
        resume_args.resume_from = step2_ckpt
        resume_summary = run_training(
            resume_args, model=model, tokenizer=tokenizer, splits=small_splits
        )
        assert resume_summary["total_steps"] == 3
        with open(log_path, "r", encoding="utf-8") as f:
            resumed_lines = [json.loads(line) for line in f if line.strip()]
        assert len(resumed_lines) == 4, (
            f"Expected 3 initial + 1 resumed log entries, got {len(resumed_lines)}"
        )
        assert resumed_lines[-1]["step"] == 3

        # 3. Trained model evaluation + Markdown comparison table
        trained_report = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            problems=small_test,
            mode="trained",
            eval_batch_size=2,
            max_turns=2,
            max_new_tokens=256,
            temperature=0.0,
            output_path=trained_json,
        )
        print(f"Trained report metrics: {trained_report['metrics']}")
        assert trained_report["metrics"]["turn1_pass_at_1"] == 50.0
        assert trained_report["metrics"]["turn2_pass_at_1"] == 100.0
        assert trained_report["metrics"]["self_healing_gain"] == 50.0
        assert trained_report["per_problem"][0]["turn2_completion"] is None
        assert trained_report["per_problem"][2]["turn2_completion"] is not None

        print("\n=== Comparison Table Verification ===")
        table = print_comparison_table(base_json, trained_json)
        assert "| Base Qwen3.5-4B | Dr. GRPO Qwen3.5-4B | Delta (+/-) |" in table
        assert "+50.00" in table
        assert "+75.00" in table

    print("\nALL PHASE 3 LOCAL END-TO-END VERIFICATION CHECKS PASSED.")


def verify_qwen35() -> None:
    print("=== Phase 3 Live Qwen/Qwen3.5-4B Smoke Verification ===")
    splits = load_mbppplus_splits(seed=42)
    small_splits = {
        "train": splits["train"][:2],
        "val": splits["val"][:2],
        "test": splits["test"][:2],
    }
    model, tokenizer = load_qwen35_4bit()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            model_id="Qwen/Qwen3.5-4B",
            epochs=1,
            max_steps=2,
            group_size=4,
            max_turns=2,
            max_new_tokens=768,
            micro_batch_size=1,
            lr=5e-6,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            eps_low=0.20,
            eps_high=0.28,
            beta=0.02,
            eval_every=2,
            eval_batch_size=2,
            save_every=2,
            run_baseline_eval=True,
            resume_from=None,
            output_dir=os.path.join(tmpdir, "checkpoints"),
            log_path=os.path.join(tmpdir, "logs", "train_metrics.jsonl"),
            base_report=os.path.join(tmpdir, "benchmarks", "base_qwen3_5_4b.json"),
            trained_report=os.path.join(tmpdir, "benchmarks", "dr_grpo_qwen3_5_4b.json"),
            seed=42,
        )
        run_training(args, model=model, tokenizer=tokenizer, splits=small_splits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-qwen35",
        action="store_true",
        help="Run live 2-step smoke test with Qwen/Qwen3.5-4B on GPU.",
    )
    args = parser.parse_args()
    if args.use_qwen35:
        verify_qwen35()
    else:
        verify_local()


if __name__ == "__main__":
    main()
