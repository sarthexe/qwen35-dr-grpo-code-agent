"""Pure-PyTorch Dr. GRPO training loop for Qwen3.5-4B on Colab T4 (16GB VRAM)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import torch

from dataset import MBPPPlusProblem, load_mbppplus_splits
from eval_benchmark import (
    evaluate_split,
    print_comparison_table,
    set_generation_mode,
    set_training_mode,
)
from grpo_engine import dr_grpo_train_step, load_qwen35_4bit
from rollout import collect_group_rollouts


def _build_optimizer(model: Any, lr: float) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit(trainable_params, lr=lr)
        except Exception:
            pass
    return torch.optim.AdamW(trainable_params, lr=lr)


def _build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if total_steps > 1 else 0

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_checkpoint(
    ckpt_dir: str,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    dead_groups: int,
    total_groups: int,
    seed: int,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt_dir)
    else:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_state.pt"))

    if hasattr(tokenizer, "save_pretrained"):
        try:
            tokenizer.save_pretrained(ckpt_dir)
        except Exception:
            pass

    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

    state = {
        "step": step,
        "dead_groups": dead_groups,
        "total_groups": total_groups,
        "seed": seed,
    }
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _load_checkpoint(
    ckpt_dir: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    adapter_cfg = os.path.join(ckpt_dir, "adapter_config.json")
    model_state_pt = os.path.join(ckpt_dir, "model_state.pt")

    if os.path.exists(adapter_cfg) and hasattr(model, "peft_config"):
        from peft import load_peft_weights, set_peft_model_state_dict

        adapters_weights = load_peft_weights(ckpt_dir, device=str(device))
        set_peft_model_state_dict(model, adapters_weights)
    elif os.path.exists(model_state_pt):
        state_dict = torch.load(model_state_pt, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(
            torch.load(opt_path, map_location="cpu", weights_only=False)
        )

    sched_path = os.path.join(ckpt_dir, "scheduler.pt")
    if os.path.exists(sched_path):
        scheduler.load_state_dict(
            torch.load(sched_path, map_location="cpu", weights_only=False)
        )

    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"step": 0, "dead_groups": 0, "total_groups": 0}


def run_training(
    args: argparse.Namespace,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    splits: Optional[Dict[str, List[MBPPPlusProblem]]] = None,
) -> Dict[str, Any]:
    """Runs the multi-turn Dr. GRPO training pipeline."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if splits is None:
        splits = load_mbppplus_splits(seed=42)
    train_problems = splits["train"]
    val_problems = splits["val"]
    test_problems = splits["test"]

    if model is None or tokenizer is None:
        model, tokenizer = load_qwen35_4bit(model_name=args.model_id)

    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else args.epochs * len(train_problems)
    )
    optimizer = _build_optimizer(model, lr=args.lr)
    scheduler = _build_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    start_step = 0
    dead_groups = 0
    total_groups = 0

    if args.resume_from:
        resumed = _load_checkpoint(
            ckpt_dir=args.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_step = int(resumed.get("step", 0))
        dead_groups = int(resumed.get("dead_groups", 0))
        total_groups = int(resumed.get("total_groups", 0))
        print(
            f"Resumed from {args.resume_from} at step={start_step} "
            f"(dead_groups={dead_groups}/{total_groups})"
        )

    # Optional pre-RL baseline benchmark on held-out test split
    if args.run_baseline_eval and not os.path.exists(args.base_report):
        print(f"Running pre-RL baseline evaluation on {len(test_problems)} test problems...")
        base_rep = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            problems=test_problems,
            mode="base",
            eval_batch_size=args.eval_batch_size,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            avg_k=1,
            output_path=args.base_report,
        )
        print(f"Baseline metrics: {base_rep['metrics']}")

    # Build deterministic problem schedule for all steps
    schedule: List[MBPPPlusProblem] = []
    epoch_idx = 0
    while len(schedule) < total_steps:
        epoch_probs = list(train_problems)
        random.Random(args.seed + epoch_idx).shuffle(epoch_probs)
        schedule.extend(epoch_probs)
        epoch_idx += 1
    schedule = schedule[:total_steps]

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_mode = "a" if (args.resume_from and os.path.exists(args.log_path)) else "w"

    rolling_window: Deque[Dict[str, float]] = deque(maxlen=10)
    last_val_report: Optional[Dict[str, Any]] = None

    with open(args.log_path, log_mode, encoding="utf-8") as log_file:
        for step in range(start_step + 1, total_steps + 1):
            t0 = time.perf_counter()
            problem = schedule[step - 1]

            # 1. Rollout phase (KV cache ON, gradient checkpointing OFF)
            set_generation_mode(model)
            rollout_batch = collect_group_rollouts(
                model=model,
                tokenizer=tokenizer,
                problem=problem,
                group_size=args.group_size,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
                top_p=0.95,
            )

            # 2. Policy optimization phase (KV cache OFF, gradient checkpointing ON)
            set_training_mode(model)
            step_stats = dr_grpo_train_step(
                model=model,
                optimizer=optimizer,
                rollout_batch=rollout_batch,
                micro_batch_size=args.micro_batch_size,
                max_gen_tokens=args.max_new_tokens,
                eps_low=args.eps_low,
                eps_high=args.eps_high,
                beta=args.beta,
                max_grad_norm=args.max_grad_norm,
            )

            if not step_stats["skipped"]:
                scheduler.step()

            total_groups += 1
            if step_stats["skipped"]:
                dead_groups += 1

            step_time = time.perf_counter() - t0
            dead_group_ratio = dead_groups / float(max(1, total_groups))
            current_lr = float(optimizer.param_groups[0]["lr"])
            peak_vram_gb = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            mean_reward = float(rollout_batch.rewards.mean().item())

            record = {
                "step": step,
                "task_id": problem.task_id,
                "loss": round(float(step_stats["loss"]), 6),
                "mean_reward": round(mean_reward, 4),
                "turn1_pass": round(rollout_batch.turn1_pass_rate, 4),
                "turn2_pass": round(rollout_batch.turn2_pass_rate, 4),
                "dead_group_skipped": bool(step_stats["skipped"]),
                "dead_group_ratio": round(dead_group_ratio, 4),
                "mean_kl": round(float(step_stats["mean_kl"]), 6),
                "mean_ratio": round(float(step_stats["mean_ratio"]), 4),
                "avg_think_tokens": round(rollout_batch.avg_think_tokens, 1),
                "format_compliance": round(rollout_batch.format_compliance_rate, 4),
                "lr": current_lr,
                "step_time_sec": round(step_time, 3),
                "peak_vram_gb": round(peak_vram_gb, 3),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            rolling_window.append(
                {
                    "mean_reward": mean_reward,
                    "turn1_pass": rollout_batch.turn1_pass_rate,
                    "turn2_pass": rollout_batch.turn2_pass_rate,
                    "kl_div": float(step_stats["mean_kl"]),
                    "avg_think_len": rollout_batch.avg_think_tokens,
                }
            )

            if step % 10 == 0 or step == total_steps:
                n_w = float(len(rolling_window))
                avg_r = sum(x["mean_reward"] for x in rolling_window) / n_w
                avg_t1 = sum(x["turn1_pass"] for x in rolling_window) / n_w
                avg_t2 = sum(x["turn2_pass"] for x in rolling_window) / n_w
                avg_kl = sum(x["kl_div"] for x in rolling_window) / n_w
                avg_th = sum(x["avg_think_len"] for x in rolling_window) / n_w
                print(
                    f"[Step {step}/{total_steps}] "
                    f"mean_reward={avg_r:.4f} | turn1_pass={avg_t1:.2f} | "
                    f"turn2_pass={avg_t2:.2f} | dead_group_ratio={dead_group_ratio:.2f} | "
                    f"kl_div={avg_kl:.5f} | avg_think_len={avg_th:.1f} | "
                    f"vram={peak_vram_gb:.2f}GB"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.eval_every > 0 and step % args.eval_every == 0:
                print(f"--- Validation check at step {step} ({len(val_problems)} problems) ---")
                last_val_report = evaluate_split(
                    model=model,
                    tokenizer=tokenizer,
                    problems=val_problems,
                    mode="trained",
                    eval_batch_size=args.eval_batch_size,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    avg_k=1,
                )
                print(f"Val metrics: {last_val_report['metrics']}")

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
                _save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    dead_groups=dead_groups,
                    total_groups=total_groups,
                    seed=args.seed,
                )
                print(f"Saved checkpoint to {ckpt_dir}")

    final_dir = os.path.join(args.output_dir, "dr_grpo_final")
    _save_checkpoint(
        ckpt_dir=final_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        step=total_steps,
        dead_groups=dead_groups,
        total_groups=total_groups,
        seed=args.seed,
    )
    print(f"Saved final Dr. GRPO checkpoint to {final_dir}")

    if args.run_baseline_eval:
        print(f"Running post-RL evaluation on {len(test_problems)} test problems...")
        evaluate_split(
            model=model,
            tokenizer=tokenizer,
            problems=test_problems,
            mode="trained",
            eval_batch_size=args.eval_batch_size,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            avg_k=1,
            output_path=args.trained_report,
        )
        print("\n=== Pre-RL vs. Post-RL Benchmark Comparison ===")
        print_comparison_table(args.base_report, args.trained_report)

    return {
        "final_dir": final_dir,
        "total_steps": total_steps,
        "dead_groups": dead_groups,
        "total_groups": total_groups,
        "last_val_report": last_val_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5-4B multi-turn code reasoning agent with pure-PyTorch Dr. GRPO."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eps-low", type=float, default=0.20)
    parser.add_argument("--eps-high", type=float, default=0.28)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--run-baseline-eval", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--log-path", default="logs/train_metrics.jsonl")
    parser.add_argument(
        "--base-report", default="benchmarks/base_qwen3_5_4b.json"
    )
    parser.add_argument(
        "--trained-report", default="benchmarks/dr_grpo_qwen3_5_4b.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
