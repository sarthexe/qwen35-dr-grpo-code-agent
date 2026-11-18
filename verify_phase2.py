"""Phase 2 Verification Script for rollout.py and grpo_engine.py.

Runs fast local unit verification by default (no model download required),
and supports `--use-qwen35` for live Colab T4 verification with Qwen/Qwen3.5-4B.
"""

from __future__ import annotations

import argparse
import contextlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_mbppplus_splits
from grpo_engine import (
    compute_dr_grpo_loss,
    dr_grpo_train_step,
    load_qwen35_4bit,
    selective_per_token_logps,
)
from rollout import collect_group_rollouts


class CharWordTokenizer:
    """Minimal deterministic tokenizer with ChatML + <think> prefill chat template."""

    def __init__(self) -> None:
        self.pad_token = "<|pad|>"
        self.eos_token = "<|im_end|>"
        self.unk_token = "<|unk|>"
        self.padding_side = "left"
        self.chat_template = "chatml_with_thinking"

        self._tok2id: Dict[str, int] = {
            self.pad_token: 0,
            self.eos_token: 1,
            self.unk_token: 2,
            "<|endoftext|>": 3,
            "<|im_start|>": 4,
            "<think>": 5,
            "</think>": 6,
            "<patch>": 7,
            "</patch>": 8,
            "<obs>": 9,
            "</obs>": 10,
        }
        # Reserve single-byte ASCII tokens 11..266
        for b in range(256):
            self._tok2id[f"<byte_{b}>"] = 11 + b

        self._id2tok: Dict[int, str] = {v: k for k, v in self._tok2id.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

        self._special_strs = [
            "<|pad|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<think>",
            "</think>",
            "<patch>",
            "</patch>",
            "<obs>",
            "</obs>",
        ]

    @property
    def vocab_size(self) -> int:
        return len(self._tok2id)

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._tok2id.get(tok, self.unk_token_id)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            if enable_thinking:
                parts.append("<|im_start|>assistant\n<think>\n")
            else:
                parts.append("<|im_start|>assistant\n")
        rendered = "".join(parts)
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)  # type: ignore[return-value]
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for sp in self._special_strs:
                if text.startswith(sp, i):
                    ids.append(self._tok2id[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                for b in text[i].encode("utf-8"):
                    ids.append(11 + b)
                i += 1
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        out_bytes = bytearray()
        chunks: List[str] = []
        for tid in ids:
            tid = int(tid)
            if tid >= 11:
                out_bytes.append(tid - 11)
            else:
                if out_bytes:
                    chunks.append(out_bytes.decode("utf-8", errors="replace"))
                    out_bytes.clear()
                tok = self._id2tok.get(tid, "")
                if skip_special_tokens and tok in (
                    self.pad_token,
                    self.eos_token,
                    "<|endoftext|>",
                    "<|im_start|>",
                ):
                    continue
                chunks.append(tok)
        if out_bytes:
            chunks.append(out_bytes.decode("utf-8", errors="replace"))
        return "".join(chunks)


class TinyAdapterCausalLM(nn.Module):
    """Tiny causal LM with LoRA-like adapter + scripted multi-turn generation for testing."""

    def __init__(self, vocab_size: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.base_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lora_A = nn.Linear(hidden_dim, 8, bias=False)
        self.lora_B = nn.Linear(8, hidden_dim, bias=False)
        # Non-zero small init so ref_logps != per_token_logps is also testable
        nn.init.normal_(self.lora_A.weight, std=0.05)
        nn.init.normal_(self.lora_B.weight, std=0.05)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._adapter_enabled = True
        self.config = SimpleNamespace(use_cache=False)

        # Scripted generation responses per turn
        self.scripted_turn1: List[List[int]] = []
        self.scripted_turn2: List[List[int]] = []
        self._gen_call = 0

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self._adapter_enabled
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = prev

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        x = self.embed(input_ids)
        h = torch.tanh(self.base_proj(x))
        if self._adapter_enabled:
            h = h + self.lora_B(self.lora_A(x))
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._gen_call += 1
        batch_size = input_ids.size(0)
        seqs = self.scripted_turn1 if self._gen_call == 1 else self.scripted_turn2
        assert len(seqs) == batch_size, (
            f"Expected {batch_size} scripted sequences, got {len(seqs)}"
        )
        max_gen = max(len(s) for s in seqs)
        padded = []
        for s in seqs:
            padded.append(s + [pad_token_id] * (max_gen - len(s)))
        gen_tensor = torch.tensor(padded, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, gen_tensor], dim=1)


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
        micro_batch_size=2,
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
