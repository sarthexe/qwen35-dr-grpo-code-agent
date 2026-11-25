"""Pure-PyTorch multi-turn code reasoning agent trained with Dr. GRPO on Qwen/Qwen3.5-4B.

Public entry points:
    dr_grpo.dataset        -> mbppplus loading, assertion extraction, prompt formatting
    dr_grpo.sandbox        -> AST validation, subprocess test harness, shaped reward
    dr_grpo.rollout        -> multi-turn rollout collection + action_mask construction
    dr_grpo.grpo_engine    -> 4-bit loader, memory-safe log-probs, Dr. GRPO objective
    dr_grpo.eval_benchmark -> batched benchmark + Markdown comparison table
    dr_grpo.train          -> training loop, telemetry, checkpointing
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
