"""Pure-PyTorch Dr. GRPO engine and memory-safe Qwen3.5-4B 4-bit LoRA loader."""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .rollout import RolloutBatch


def _freeze_vision_modules(model: Any) -> None:
    """Freezes any multimodal vision tower parameters so only the text backbone trains."""
    candidate_attrs = [
        ("visual",),
        ("model", "visual"),
        ("vision_tower",),
        ("model", "vision_tower"),
        ("base_model", "model", "visual"),
        ("base_model", "model", "model", "visual"),
    ]
    for path in candidate_attrs:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "parameters"):
            obj.eval()
            for param in obj.parameters():
                param.requires_grad = False

    # Belt-and-suspenders pass across named parameters
    for name, param in model.named_parameters():
        if "visual" in name or "vision_tower" in name:
            param.requires_grad = False


def load_qwen35_4bit(
    model_name: str = "Qwen/Qwen3.5-4B",
    device_map: Union[str, Dict[str, Any]] = "auto",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
) -> Tuple[Any, Any]:
    """Loads Qwen/Qwen3.5-4B in 4-bit NF4 (fp16 compute for Colab T4) with LoRA on the text backbone."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
        "device_map": device_map,
        "trust_remote_code": True,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except (ValueError, KeyError, TypeError):
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, **load_kwargs
            )
        except (ValueError, KeyError, TypeError):
            from transformers import Qwen3_5ForConditionalGeneration

            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name, **load_kwargs
            )

    _freeze_vision_modules(model)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    _freeze_vision_modules(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def selective_per_token_logps(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes per-token log-probabilities with causal shift row-by-row over batch B.

    Avoids materializing a full 3D `(B, T, V)` fp32 tensor for Qwen3.5's 248,320 vocab.
    Returns tensor of shape `(B, T - 1)`.
    """
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    except TypeError:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, :-1, :]  # (B, T - 1, V)
    targets = input_ids[:, 1:]  # (B, T - 1)

    per_token_logps = []
    for logit_row, target_row in zip(logits, targets):
        log_probs_row = F.log_softmax(logit_row.float(), dim=-1)
        token_logps_row = log_probs_row.gather(
            dim=-1, index=target_row.unsqueeze(-1)
        ).squeeze(-1)
        per_token_logps.append(token_logps_row)

    return torch.stack(per_token_logps, dim=0)


def _compute_logps_microbatched(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    micro_batch_size: int = 2,
    disable_adapter: bool = False,
) -> torch.Tensor:
    chunks = []
    adapter_ctx = (
        model.disable_adapter()
        if (disable_adapter and hasattr(model, "disable_adapter"))
        else contextlib.nullcontext()
    )
    was_training = model.training
    model.eval()
    try:
        with adapter_ctx, torch.no_grad():
            for start in range(0, input_ids.size(0), micro_batch_size):
                end = min(start + micro_batch_size, input_ids.size(0))
                mb_logps = selective_per_token_logps(
                    model=model,
                    input_ids=input_ids[start:end],
                    attention_mask=attention_mask[start:end],
                )
                chunks.append(mb_logps)
    finally:
        if was_training:
            model.train()
    return torch.cat(chunks, dim=0)


def compute_dr_grpo_loss(
    per_token_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    max_gen_tokens: int = 768,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
    beta: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Computes the Dr. GRPO objective with asymmetric PPO clipping, Schulman k3 KL,
    and fixed-constant `max_gen_tokens` sequence length normalization.

    Returns `(loss, metrics_dict)`.
    """
    if action_mask.shape[1] == per_token_logps.shape[1] + 1:
        shifted_mask = action_mask[:, 1:].float()
    elif action_mask.shape[1] == per_token_logps.shape[1]:
        shifted_mask = action_mask.float()
    else:
        raise ValueError(
            f"Incompatible action_mask shape {tuple(action_mask.shape)} "
            f"for per_token_logps shape {tuple(per_token_logps.shape)}"
        )

    adv = advantages.float().unsqueeze(-1)  # (B, 1)

    log_ratio = per_token_logps - old_logps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    surrogate = torch.min(ratio * adv, clipped_ratio * adv)

    # Schulman unbiased k3 KL estimator: exp(ref - pi) - (ref - pi) - 1
    ref_diff = ref_logps - per_token_logps
    kl = torch.exp(ref_diff) - ref_diff - 1.0

    per_token_obj = (surrogate - beta * kl) * shifted_mask

    # Dr. GRPO: divide by fixed constant `max_gen_tokens`, NOT variable |o_i|
    seq_loss = -per_token_obj.sum(dim=-1) / float(max_gen_tokens)
    loss = seq_loss.mean()

    mask_sum = shifted_mask.sum().clamp(min=1.0)
    mean_kl = ((kl * shifted_mask).sum() / mask_sum).detach()
    mean_ratio = ((ratio * shifted_mask).sum() / mask_sum).detach()
    clipped_tokens = ((ratio < (1.0 - eps_low)) | (ratio > (1.0 + eps_high))).float()
    clip_frac = ((clipped_tokens * shifted_mask).sum() / mask_sum).detach()

    return loss, {
        "mean_kl": mean_kl,
        "mean_ratio": mean_ratio,
        "clip_frac": clip_frac,
    }


def dr_grpo_train_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    rollout_batch: Union[RolloutBatch, Dict[str, Any]],
    micro_batch_size: int = 2,
    max_gen_tokens: int = 768,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
    beta: float = 0.02,
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    """Executes one Dr. GRPO update step using micro-batch gradient accumulation."""
    if isinstance(rollout_batch, dict):
        input_ids = rollout_batch["input_ids"]
        attention_mask = rollout_batch["attention_mask"]
        action_mask = rollout_batch["action_mask"]
        rewards = rollout_batch["rewards"]
    else:
        input_ids = rollout_batch.input_ids
        attention_mask = rollout_batch.attention_mask
        action_mask = rollout_batch.action_mask
        rewards = rollout_batch.rewards

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    action_mask = action_mask.to(device)
    rewards = rewards.to(device=device, dtype=torch.float32)

    # Fast-path dead-group skip
    reward_range = float((rewards.max() - rewards.min()).item())
    if reward_range < 1e-5:
        return {
            "loss": 0.0,
            "mean_kl": 0.0,
            "mean_ratio": 1.0,
            "clip_frac": 0.0,
            "skipped": True,
            "mean_reward": float(rewards.mean().item()),
        }

    # Dr. GRPO advantage: mean-centered ONLY (no std normalization)
    advantages = rewards - rewards.mean()

    old_logps = _compute_logps_microbatched(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        micro_batch_size=micro_batch_size,
        disable_adapter=False,
    )
    ref_logps = _compute_logps_microbatched(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        micro_batch_size=micro_batch_size,
        disable_adapter=True,
    )

    group_size = input_ids.size(0)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    total_loss = 0.0
    total_kl = 0.0
    total_ratio = 0.0
    total_clip_frac = 0.0

    for start in range(0, group_size, micro_batch_size):
        end = min(start + micro_batch_size, group_size)
        mb_weight = float(end - start) / float(group_size)

        mb_logps = selective_per_token_logps(
            model=model,
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
        )
        mb_loss, mb_metrics = compute_dr_grpo_loss(
            per_token_logps=mb_logps,
            old_logps=old_logps[start:end],
            ref_logps=ref_logps[start:end],
            action_mask=action_mask[start:end],
            advantages=advantages[start:end],
            max_gen_tokens=max_gen_tokens,
            eps_low=eps_low,
            eps_high=eps_high,
            beta=beta,
        )
        (mb_loss * mb_weight).backward()

        total_loss += float(mb_loss.detach().item()) * mb_weight
        total_kl += float(mb_metrics["mean_kl"].item()) * mb_weight
        total_ratio += float(mb_metrics["mean_ratio"].item()) * mb_weight
        total_clip_frac += float(mb_metrics["clip_frac"].item()) * mb_weight

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_grad_norm,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss,
        "mean_kl": total_kl,
        "mean_ratio": total_ratio,
        "clip_frac": total_clip_frac,
        "skipped": False,
        "mean_reward": float(rewards.mean().item()),
    }
