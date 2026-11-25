"""Deterministic test doubles used by the verification suite.

These mocks exist so the full rollout -> reward -> Dr. GRPO update -> benchmark
pipeline can be exercised on CPU in seconds, with no model download and no GPU:

* `CharWordTokenizer`  - byte-level tokenizer that implements the same surface the
  pipeline touches (`apply_chat_template`, `encode`, `decode`, `padding_side`,
  `pad_token_id`, `eos_token_id`) and, importantly, reproduces Qwen3.5's
  `enable_thinking=True` behaviour of pre-filling ` thinking\\n` at the end of the
  generation prompt.
* `TinyAdapterCausalLM` - tiny causal LM with a LoRA-shaped low-rank branch and a
  `disable_adapter()` context manager, so `ref_logps` can be computed from the same
  weights exactly like the real PEFT model.
* `DynamicMockCausalLM` - scripted generation that behaves differently in base mode,
  sampling (rollout) mode and greedy (eval) mode, which lets Phase 3 assert concrete
  base-vs-trained deltas end to end.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from dr_grpo.dataset import MBPPPlusProblem


class CharWordTokenizer:
    """Minimal deterministic tokenizer with ChatML + ` thinking` prefill chat template."""

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
            " thinking": 5,
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
            " thinking",
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
    """Tiny causal LM with a LoRA-like adapter + scripted multi-turn generation."""

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


class DynamicMockCausalLM(TinyAdapterCausalLM):
    """Adapter-aware mock LM that generates valid/buggy completions based on turn,
    sampling mode, and adapter state (`disable_adapter()`).
    """

    def __init__(
        self,
        tokenizer: CharWordTokenizer,
        problems: List[MBPPPlusProblem],
        hidden_dim: int = 32,
    ) -> None:
        super().__init__(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim)
        self.tok = tokenizer
        self.prob_by_entry: Dict[str, MBPPPlusProblem] = {
            p.entry_point: p for p in problems if p.entry_point
        }
        self.test_entries: List[str] = [p.entry_point for p in problems[-4:]]
        self.gradient_checkpointing_calls: List[bool] = []

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_calls.append(True)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_calls.append(False)

    def _identify_problem(self, prompt_text: str) -> MBPPPlusProblem:
        for entry, prob in self.prob_by_entry.items():
            if entry in prompt_text:
                return prob
        return next(iter(self.prob_by_entry.values()))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: Any,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        seqs: List[List[int]] = []

        for idx in range(batch_size):
            prompt_ids = [
                int(t)
                for t, m in zip(input_ids[idx].tolist(), attention_mask[idx].tolist())
                if m == 1
            ]
            prompt_text = self.tok.decode(prompt_ids, skip_special_tokens=False)
            prob = self._identify_problem(prompt_text)
            is_turn2 = "<obs>" in prompt_text
            prob_rank = (
                self.test_entries.index(prob.entry_point)
                if prob.entry_point in self.test_entries
                else 0
            )

            passing_resp = (
                f"Solving {prob.entry_point}.\n</think>\n"
                f"<patch>\n{prob.canonical_solution.strip()}\n</patch><|im_end|>"
            )
            buggy_resp = (
                f"Initial guess for {prob.entry_point}.\n</think>\n"
                f"<patch>\ndef {prob.entry_point}(*args, **kwargs):\n    return None\n</patch><|im_end|>"
            )

            if not self._adapter_enabled:
                # Base mode: Turn 1 fails; Turn 2 self-heals only problem 0
                if is_turn2 and prob_rank == 0:
                    resp = passing_resp
                else:
                    resp = buggy_resp
            elif do_sample:
                # Training rollout mode: diverse group [pass, heal, heal, fail]
                if not is_turn2:
                    resp = passing_resp if (idx % 4 == 0) else buggy_resp
                else:
                    resp = passing_resp if (idx % 2 == 0) else buggy_resp
            else:
                # Trained greedy eval: problems 0,1 pass Turn 1; 2,3 heal on Turn 2
                if not is_turn2:
                    resp = passing_resp if prob_rank < 2 else buggy_resp
                else:
                    resp = passing_resp

            seqs.append(self.tok.encode(resp))

        max_gen = max(len(s) for s in seqs)
        padded = [s + [pad_token_id] * (max_gen - len(s)) for s in seqs]
        gen_tensor = torch.tensor(padded, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, gen_tensor], dim=1)
