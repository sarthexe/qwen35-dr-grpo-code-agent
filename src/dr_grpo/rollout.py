"""Multi-turn rollout collection and action_mask construction for Dr. GRPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .dataset import MBPPPlusProblem, format_chat_messages
from .sandbox import SandboxResult, grade_completion


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    rewards: torch.Tensor
    turn1_pass_rate: float
    turn2_pass_rate: float
    avg_think_tokens: float
    format_compliance_rate: float
    completions_turn1: List[str]
    completions_turn2: List[Optional[str]]
    turn1_results: List[SandboxResult]
    final_results: List[SandboxResult]


def _render_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Applies the tokenizer chat template with `enable_thinking=True` (with ChatML fallback)."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n<think>\n")
    return "".join(parts)


def _get_stop_token_ids(tokenizer: Any) -> List[int]:
    stop_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                if tid not in stop_ids:
                    stop_ids.append(tid)
        except Exception:
            pass
    return stop_ids


def _trim_generated_ids(
    raw_ids: Sequence[int],
    stop_ids: List[int],
    pad_id: int,
) -> List[int]:
    """Keeps generated tokens up to and including the first stop token (dropping trailing pads)."""
    out: List[int] = []
    stop_set = set(stop_ids)
    for tid in raw_ids:
        tid_int = int(tid)
        if tid_int == pad_id and pad_id not in stop_set:
            break
        out.append(tid_int)
        if tid_int in stop_set:
            break
    return out


def _decode_for_grading(
    tokenizer: Any,
    gen_ids: Sequence[int],
    stop_ids: List[int],
    pad_id: int,
) -> str:
    """Decodes generated tokens preserving `<think>` / `</think>` and aligning `<think>\n` prefill."""
    stop_set = set(stop_ids) | {pad_id}
    clean_ids = [int(t) for t in gen_ids if int(t) not in stop_set]
    text = tokenizer.decode(clean_ids, skip_special_tokens=False)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    if not text.lstrip().startswith("<think>"):
        text = "<think>\n" + text
    return text


def _build_turn2_suffix_ids(
    tokenizer: Any,
    problem: MBPPPlusProblem,
    obs_text: str,
    turn1_ended_with_stop: bool,
) -> List[int]:
    """Tokenizes only the Turn-2 ChatML boundary + `<obs>...</obs>` user turn + assistant `<think>` prefix."""
    marker = "__TURN1_ASSISTANT_PLACEHOLDER__"
    msgs = format_chat_messages(problem) + [
        {"role": "assistant", "content": marker},
        {"role": "user", "content": obs_text},
    ]
    rendered = _render_chat_template(tokenizer, msgs, add_generation_prompt=True)
    if marker not in rendered:
        # Fallback if a custom template omits assistant content
        suffix_str = f"<|im_end|>\n<|im_start|>user\n{obs_text}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    else:
        _, suffix_str = rendered.split(marker, 1)

    if turn1_ended_with_stop:
        for stop_tok in ("<|im_end|>", tokenizer.eos_token or ""):
            if stop_tok and suffix_str.startswith(stop_tok):
                suffix_str = suffix_str[len(stop_tok) :]
                break

    return list(tokenizer.encode(suffix_str, add_special_tokens=False))


def _left_pad_sequences(
    sequences: List[List[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    batch_ids = []
    batch_mask = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        batch_ids.append([pad_id] * pad_len + seq)
        batch_mask.append([0] * pad_len + [1] * len(seq))
    return (
        torch.tensor(batch_ids, dtype=torch.long, device=device),
        torch.tensor(batch_mask, dtype=torch.long, device=device),
    )


def _right_pad_trajectories(
    sequences: List[List[int]],
    action_masks: List[List[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    padded_ids = []
    padded_attn = []
    padded_act = []
    for seq, amask in zip(sequences, action_masks):
        pad_len = max_len - len(seq)
        padded_ids.append(seq + [pad_id] * pad_len)
        padded_attn.append([1] * len(seq) + [0] * pad_len)
        padded_act.append(amask + [0] * pad_len)
    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(padded_attn, dtype=torch.long, device=device),
        torch.tensor(padded_act, dtype=torch.long, device=device),
    )


def _generate_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pad_id: int,
    stop_ids: List[int],
) -> torch.Tensor:
    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "eos_token_id": stop_ids[0] if len(stop_ids) == 1 else stop_ids,
        "use_cache": True,
    }
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        return model.generate(**gen_kwargs)


def collect_group_rollouts(
    model: Any,
    tokenizer: Any,
    problem: MBPPPlusProblem,
    group_size: int = 4,
    max_turns: int = 2,
    max_new_tokens: int = 768,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> RolloutBatch:
    """Runs 1- or 2-turn rollout collection for a group of `G` trajectories."""
    if max_turns not in (1, 2):
        raise ValueError(f"max_turns must be 1 or 2, got {max_turns}")

    device = next(model.parameters()).device
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id)
    stop_ids = _get_stop_token_ids(tokenizer)
    if not stop_ids:
        stop_ids = [pad_id]

    was_training = model.training
    model.eval()

    try:
        # --- Turn 1 ---
        messages = format_chat_messages(problem)
        prompt_1_text = _render_chat_template(
            tokenizer, messages, add_generation_prompt=True
        )
        prompt_1_ids = list(tokenizer.encode(prompt_1_text, add_special_tokens=False))
        prompt_1_len = len(prompt_1_ids)

        prompt_1_batch = torch.tensor(
            [prompt_1_ids] * group_size, dtype=torch.long, device=device
        )
        attn_1_batch = torch.ones_like(prompt_1_batch)

        gen_1_out = _generate_batch(
            model=model,
            input_ids=prompt_1_batch,
            attention_mask=attn_1_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_id=pad_id,
            stop_ids=stop_ids,
        )

        gen_1_ids_list: List[List[int]] = []
        completions_turn1: List[str] = []
        turn1_results: List[SandboxResult] = []

        for i in range(group_size):
            raw_gen = gen_1_out[i, prompt_1_len:].tolist()
            trimmed = _trim_generated_ids(raw_gen, stop_ids=stop_ids, pad_id=pad_id)
            gen_1_ids_list.append(trimmed)

            comp_text = _decode_for_grading(
                tokenizer, trimmed, stop_ids=stop_ids, pad_id=pad_id
            )
            completions_turn1.append(comp_text)
            res_1 = grade_completion(
                completion=comp_text,
                assertions=problem.assertions,
                test_preamble=problem.test_preamble,
                turn=1,
            )
            turn1_results.append(res_1)

        # Initialize trajectory buffers with Turn 1
        full_sequences: List[List[int]] = [
            prompt_1_ids + gen_1_ids_list[i] for i in range(group_size)
        ]
        full_action_masks: List[List[int]] = [
            [0] * prompt_1_len + [1] * len(gen_1_ids_list[i])
            for i in range(group_size)
        ]
        completions_turn2: List[Optional[str]] = [None] * group_size
        final_results: List[SandboxResult] = list(turn1_results)

        # --- Turn 2 (Self-Healing for failing trajectories) ---
        failing_indices = [
            i for i, res in enumerate(turn1_results) if not res.all_passed
        ]
        if max_turns == 2 and failing_indices:
            stop_set = set(stop_ids)
            t2_prompts: List[List[int]] = []
            t2_mask_prefixes: List[List[int]] = []

            for i in failing_indices:
                ended_with_stop = bool(
                    gen_1_ids_list[i] and gen_1_ids_list[i][-1] in stop_set
                )
                obs_suffix_ids = _build_turn2_suffix_ids(
                    tokenizer=tokenizer,
                    problem=problem,
                    obs_text=turn1_results[i].next_prompt_suffix,
                    turn1_ended_with_stop=ended_with_stop,
                )
                t2_prompt = full_sequences[i] + obs_suffix_ids
                t2_mask = full_action_masks[i] + [0] * len(obs_suffix_ids)
                t2_prompts.append(t2_prompt)
                t2_mask_prefixes.append(t2_mask)

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

            for k, traj_idx in enumerate(failing_indices):
                raw_gen_2 = gen_2_out[k, t2_padded_len:].tolist()
                trimmed_2 = _trim_generated_ids(
                    raw_gen_2, stop_ids=stop_ids, pad_id=pad_id
                )
                comp_text_2 = _decode_for_grading(
                    tokenizer, trimmed_2, stop_ids=stop_ids, pad_id=pad_id
                )
                completions_turn2[traj_idx] = comp_text_2

                res_2 = grade_completion(
                    completion=comp_text_2,
                    assertions=problem.assertions,
                    test_preamble=problem.test_preamble,
                    turn=2,
                    prev_code=turn1_results[traj_idx].extracted_code,
                )
                final_results[traj_idx] = res_2
                full_sequences[traj_idx] = t2_prompts[k] + trimmed_2
                full_action_masks[traj_idx] = t2_mask_prefixes[k] + [1] * len(trimmed_2)

        input_ids_t, attention_mask_t, action_mask_t = _right_pad_trajectories(
            full_sequences, full_action_masks, pad_id=pad_id, device=device
        )
        rewards_t = torch.tensor(
            [r.reward for r in final_results], dtype=torch.float32, device=device
        )

        turn1_pass_rate = sum(1 for r in turn1_results if r.all_passed) / float(
            group_size
        )
        turn2_pass_rate = sum(1 for r in final_results if r.all_passed) / float(
            group_size
        )
        format_compliance_rate = sum(
            1 for r in final_results if r.format_valid
        ) / float(group_size)
        think_token_counts = [
            len(tokenizer.encode(r.think_content, add_special_tokens=False))
            if r.think_content
            else 0
            for r in final_results
        ]
        avg_think_tokens = sum(think_token_counts) / float(group_size)

        return RolloutBatch(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            action_mask=action_mask_t,
            rewards=rewards_t,
            turn1_pass_rate=turn1_pass_rate,
            turn2_pass_rate=turn2_pass_rate,
            avg_think_tokens=avg_think_tokens,
            format_compliance_rate=format_compliance_rate,
            completions_turn1=completions_turn1,
            completions_turn2=completions_turn2,
            turn1_results=turn1_results,
            final_results=final_results,
        )
    finally:
        tokenizer.padding_side = orig_padding_side
        if was_training:
            model.train()
