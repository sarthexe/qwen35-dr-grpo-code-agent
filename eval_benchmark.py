"""Batched multi-turn evaluation and benchmark comparison for Qwen3.5-4B Dr. GRPO."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from dataset import MBPPPlusProblem, format_chat_messages, load_mbppplus_splits
from rollout import (
    _build_turn2_suffix_ids,
    _decode_for_grading,
    _generate_batch,
    _get_stop_token_ids,
    _left_pad_sequences,
    _render_chat_template,
    _trim_generated_ids,
)
from sandbox import SandboxResult, grade_completion


def set_generation_mode(model: Any) -> None:
    """Enables KV cache and disables gradient checkpointing for fast generation."""
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = True


def set_training_mode(model: Any) -> None:
    """Disables KV cache and enables gradient checkpointing for 4-bit LoRA training."""
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def load_model_for_eval(
    model_id: str = "Qwen/Qwen3.5-4B",
    adapter_path: Optional[str] = None,
) -> tuple[Any, Any]:
    """Loads 4-bit Qwen3.5-4B for base (`disable_adapter`) or trained LoRA evaluation."""
    from grpo_engine import _freeze_vision_modules, load_qwen35_4bit

    if not adapter_path:
        return load_qwen35_4bit(model_name=model_id)

    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs: Dict[str, Any] = {
        "quantization_config": bnb_config,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, KeyError, TypeError):
        try:
            base_model = AutoModelForImageTextToText.from_pretrained(
                model_id, **load_kwargs
            )
        except (ValueError, KeyError, TypeError):
            from transformers import Qwen3_5ForConditionalGeneration

            base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id, **load_kwargs
            )

    _freeze_vision_modules(base_model)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    _freeze_vision_modules(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def evaluate_split(
    model: Any,
    tokenizer: Any,
    problems: List[MBPPPlusProblem],
    mode: str = "base",
    eval_batch_size: int = 8,
    max_turns: int = 2,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.95,
    avg_k: int = 1,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates `model` on `problems` with batched Turn-1 and Turn-2 self-healing.

    When `mode == "base"`, inference runs inside `with model.disable_adapter():`.
    When `temperature == 0.0`, greedy decoding is used (`avg_k` effectively 1).
    """
    if mode not in ("base", "trained"):
        raise ValueError(f"mode must be 'base' or 'trained', got {mode!r}")
    if max_turns not in (1, 2):
        raise ValueError(f"max_turns must be 1 or 2, got {max_turns}")
    if not problems:
        raise ValueError("problems list is empty.")

    effective_k = 1 if temperature == 0.0 else max(1, avg_k)
    device = next(model.parameters()).device
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id)
    stop_ids = _get_stop_token_ids(tokenizer)
    if not stop_ids:
        stop_ids = [pad_id]
    stop_set = set(stop_ids)

    was_training = model.training
    set_generation_mode(model)

    adapter_ctx = (
        model.disable_adapter()
        if (mode == "base" and hasattr(model, "disable_adapter"))
        else contextlib.nullcontext()
    )

    work_items = [
        (prob_idx, sample_idx, prob)
        for prob_idx, prob in enumerate(problems)
        for sample_idx in range(effective_k)
    ]

    sample_records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    try:
        with adapter_ctx, torch.no_grad():
            for start in range(0, len(work_items), eval_batch_size):
                batch_items = work_items[start : start + eval_batch_size]
                batch_len = len(batch_items)

                # --- Turn 1 batched generation ---
                prompt_1_ids_list: List[List[int]] = []
                for _, _, prob in batch_items:
                    msgs = format_chat_messages(prob)
                    prompt_1_text = _render_chat_template(
                        tokenizer, msgs, add_generation_prompt=True
                    )
                    prompt_1_ids_list.append(
                        list(tokenizer.encode(prompt_1_text, add_special_tokens=False))
                    )

                t1_input_ids, t1_attn_mask = _left_pad_sequences(
                    prompt_1_ids_list, pad_id=pad_id, device=device
                )
                t1_padded_len = t1_input_ids.size(1)

                gen_1_out = _generate_batch(
                    model=model,
                    input_ids=t1_input_ids,
                    attention_mask=t1_attn_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pad_id=pad_id,
                    stop_ids=stop_ids,
                )

                gen_1_ids_list: List[List[int]] = []
                turn1_results: List[SandboxResult] = []
                for idx in range(batch_len):
                    _, _, prob = batch_items[idx]
                    raw_gen_1 = gen_1_out[idx, t1_padded_len:].tolist()
                    trimmed_1 = _trim_generated_ids(
                        raw_gen_1, stop_ids=stop_ids, pad_id=pad_id
                    )
                    gen_1_ids_list.append(trimmed_1)
                    comp_1 = _decode_for_grading(
                        tokenizer, trimmed_1, stop_ids=stop_ids, pad_id=pad_id
                    )
                    res_1 = grade_completion(
                        completion=comp_1,
                        assertions=prob.assertions,
                        test_preamble=prob.test_preamble,
                        turn=1,
                    )
                    turn1_results.append(res_1)

                final_results: List[SandboxResult] = list(turn1_results)

                # --- Turn 2 batched self-healing for failing items ---
                failing_indices = [
                    idx for idx, res in enumerate(turn1_results) if not res.all_passed
                ]
                if max_turns == 2 and failing_indices:
                    t2_prompts: List[List[int]] = []
                    for idx in failing_indices:
                        _, _, prob = batch_items[idx]
                        ended_with_stop = bool(
                            gen_1_ids_list[idx]
                            and gen_1_ids_list[idx][-1] in stop_set
                        )
                        obs_suffix_ids = _build_turn2_suffix_ids(
                            tokenizer=tokenizer,
                            problem=prob,
                            obs_text=turn1_results[idx].next_prompt_suffix,
                            turn1_ended_with_stop=ended_with_stop,
                        )
                        t2_prompts.append(
                            prompt_1_ids_list[idx]
                            + gen_1_ids_list[idx]
                            + obs_suffix_ids
                        )

                    t2_input_ids, t2_attn_mask = _left_pad_sequences(
                        t2_prompts, pad_id=pad_id, device=device
                    )
                    t2_padded_len = t2_input_ids.size(1)

                    gen_2_out = _generate_batch(
                        model=model,
                        input_ids=t2_input_ids,
                        attention_mask=t2_attn_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        pad_id=pad_id,
                        stop_ids=stop_ids,
                    )

                    for k, item_idx in enumerate(failing_indices):
                        _, _, prob = batch_items[item_idx]
                        raw_gen_2 = gen_2_out[k, t2_padded_len:].tolist()
                        trimmed_2 = _trim_generated_ids(
                            raw_gen_2, stop_ids=stop_ids, pad_id=pad_id
                        )
                        comp_2 = _decode_for_grading(
                            tokenizer, trimmed_2, stop_ids=stop_ids, pad_id=pad_id
                        )
                        res_2 = grade_completion(
                            completion=comp_2,
                            assertions=prob.assertions,
                            test_preamble=prob.test_preamble,
                            turn=2,
                            prev_code=turn1_results[item_idx].extracted_code,
                        )
                        final_results[item_idx] = res_2

                for idx in range(batch_len):
                    prob_idx, sample_idx, prob = batch_items[idx]
                    r1 = turn1_results[idx]
                    rf = final_results[idx]
                    think_tokens = (
                        len(tokenizer.encode(rf.think_content, add_special_tokens=False))
                        if rf.think_content
                        else 0
                    )
                    sample_records.append(
                        {
                            "problem_index": prob_idx,
                            "sample_index": sample_idx,
                            "task_id": prob.task_id,
                            "entry_point": prob.entry_point,
                            "turn1_passed": r1.all_passed,
                            "turn2_passed": rf.all_passed,
                            "self_healed": (not r1.all_passed) and rf.all_passed,
                            "passed_assertions": rf.passed_count,
                            "total_assertions": rf.total_count,
                            "assertion_pass_ratio": rf.pass_ratio,
                            "format_valid": rf.format_valid,
                            "think_tokens": think_tokens,
                            "reward": rf.reward,
                            "security_violation": rf.security_violation,
                            "timed_out": rf.timed_out,
                            "first_failure": "" if rf.all_passed else rf.first_failure_traceback,
                        }
                    )
    finally:
        tokenizer.padding_side = orig_padding_side
        if was_training:
            set_training_mode(model)

    total_elapsed = time.perf_counter() - t_start
    n_samples = len(sample_records)
    n_problems = len(problems)

    turn1_pass_at_1 = (
        100.0 * sum(1 for r in sample_records if r["turn1_passed"]) / n_samples
    )
    turn2_pass_at_1 = (
        100.0 * sum(1 for r in sample_records if r["turn2_passed"]) / n_samples
    )
    self_healing_gain = turn2_pass_at_1 - turn1_pass_at_1
    mean_assertion_pass_ratio = (
        100.0 * sum(r["assertion_pass_ratio"] for r in sample_records) / n_samples
    )
    format_compliance_rate = (
        100.0 * sum(1 for r in sample_records if r["format_valid"]) / n_samples
    )
    avg_think_tokens = sum(r["think_tokens"] for r in sample_records) / n_samples
    avg_latency_sec = total_elapsed / n_problems

    report: Dict[str, Any] = {
        "mode": mode,
        "num_problems": n_problems,
        "avg_k": effective_k,
        "temperature": temperature,
        "max_turns": max_turns,
        "metrics": {
            "turn1_pass_at_1": round(turn1_pass_at_1, 2),
            "turn2_pass_at_1": round(turn2_pass_at_1, 2),
            "self_healing_gain": round(self_healing_gain, 2),
            "mean_assertion_pass_ratio": round(mean_assertion_pass_ratio, 2),
            "format_compliance_rate": round(format_compliance_rate, 2),
            "avg_think_tokens": round(avg_think_tokens, 1),
            "avg_latency_sec": round(avg_latency_sec, 3),
        },
        "total_elapsed_sec": round(total_elapsed, 2),
        "per_problem": sample_records,
    }

    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def print_comparison_table(base_json_path: str, trained_json_path: str) -> str:
    """Loads base and trained JSON reports and prints a Markdown comparison table."""
    with open(base_json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open(trained_json_path, "r", encoding="utf-8") as f:
        trained_data = json.load(f)

    bm = base_data["metrics"]
    tm = trained_data["metrics"]

    rows = [
        ("Turn-1 Pass@1 (%)", "turn1_pass_at_1", "{:.2f}", "{:+.2f}"),
        ("Turn-2 Self-Healed Pass@1 (%)", "turn2_pass_at_1", "{:.2f}", "{:+.2f}"),
        ("Self-Healing Recovery Gain (+%)", "self_healing_gain", "{:.2f}", "{:+.2f}"),
        (
            "Mean Assertion Pass Ratio (%)",
            "mean_assertion_pass_ratio",
            "{:.2f}",
            "{:+.2f}",
        ),
        ("Format Compliance (%)", "format_compliance_rate", "{:.2f}", "{:+.2f}"),
        ("Avg <think> Token Length", "avg_think_tokens", "{:.1f}", "{:+.1f}"),
        ("Avg Latency (s/problem)", "avg_latency_sec", "{:.3f}", "{:+.3f}"),
    ]

    lines = [
        "| Metric | Base Qwen3.5-4B | Dr. GRPO Qwen3.5-4B | Delta (+/-) |",
        "|---|---:|---:|---:|",
    ]
    for label, key, val_fmt, delta_fmt in rows:
        b_val = float(bm[key])
        t_val = float(tm[key])
        delta = t_val - b_val
        lines.append(
            f"| {label} | {val_fmt.format(b_val)} | {val_fmt.format(t_val)} | {delta_fmt.format(delta)} |"
        )

    table = "\n".join(lines)
    print(table)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Base or Dr. GRPO Qwen3.5-4B on evalplus/mbppplus."
    )
    parser.add_argument(
        "--mode",
        choices=["base", "trained", "compare"],
        default="base",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path", default="checkpoints/dr_grpo_final")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--avg-k", type=int, default=1)
    parser.add_argument(
        "--base-report", default="benchmarks/base_qwen3_5_4b.json"
    )
    parser.add_argument(
        "--trained-report", default="benchmarks/dr_grpo_qwen3_5_4b.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "compare":
        print_comparison_table(args.base_report, args.trained_report)
        return

    splits = load_mbppplus_splits(seed=42)
    problems = splits[args.split]
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    adapter = args.adapter_path if args.mode == "trained" else None
    model, tokenizer = load_model_for_eval(
        model_id=args.model_id, adapter_path=adapter
    )

    out_path = args.base_report if args.mode == "base" else args.trained_report
    report = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        mode=args.mode,
        eval_batch_size=args.eval_batch_size,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        avg_k=args.avg_k,
        output_path=out_path,
    )
    print(f"Saved {args.mode} evaluation report to {out_path}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
