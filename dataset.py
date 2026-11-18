"""Dataset loading, assertion extraction, prompt formatting, and deterministic splitting
for evalplus/mbppplus.
"""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset


@dataclass
class MBPPPlusProblem:
    task_id: int
    prompt: str
    formatted_prompt: str
    entry_point: str
    signature_hint: str
    test_preamble: str
    assertions: List[str]
    test_list: List[str]
    test_imports: List[str]
    canonical_solution: str


class _LoopVarReplacer(ast.NodeTransformer):
    """Replaces loop variable ast.Name references (`inp`, `exp`) with indexed lookups."""

    def __init__(self, idx: int, replace_exp: bool) -> None:
        self.idx = idx
        self.replace_exp = replace_exp

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "inp":
            return ast.Subscript(
                value=ast.Name(id="inputs", ctx=ast.Load()),
                slice=ast.Constant(value=self.idx),
                ctx=ast.Load(),
            )
        if self.replace_exp and node.id == "exp":
            return ast.Subscript(
                value=ast.Name(id="results", ctx=ast.Load()),
                slice=ast.Constant(value=self.idx),
                ctx=ast.Load(),
            )
        return node


def _extract_entry_point(expr_node: ast.AST, fallback_code: str = "") -> str:
    """Extracts the tested function name from an `assertion(func(*inp), ...)` AST node."""
    if isinstance(expr_node, ast.Expr) and isinstance(expr_node.value, ast.Call):
        call = expr_node.value
        if isinstance(call.func, ast.Name) and call.func.id == "assertion" and call.args:
            first_arg = call.args[0]
            if isinstance(first_arg, ast.Call) and isinstance(first_arg.func, ast.Name):
                return first_arg.func.id

    if fallback_code:
        try:
            tree = ast.parse(fallback_code)
            defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            if defs:
                return defs[-1]
        except SyntaxError:
            pass
    return ""


def extract_assertions_from_test(
    test_code: str,
    test_list: Optional[List[str]] = None,
    test_imports: Optional[List[str]] = None,
    canonical_solution: str = "",
) -> tuple[str, List[str], str]:
    """Extracts `(test_preamble, assertions, entry_point)` from an mbppplus `test` field.

    In `evalplus/mbppplus`, `test` defines helper functions (`is_floats`, `assertion`,
    and occasionally `ref_func`), `inputs = [...]`, optional `results = [...]`, and a
    trailing `for` loop calling `assertion(...)` for every test case. Unrolling that
    loop into individual statements lets the sandbox execute every test case inside its
    own `try/except` block for granular partial credit.
    """
    test_list = test_list or []
    test_imports = test_imports or []

    if not test_code or not test_code.strip():
        preamble = "\n".join(test_imports)
        entry_point = _extract_entry_point(ast.Pass(), canonical_solution)
        return preamble, list(test_list), entry_point

    tree = ast.parse(test_code)
    for_nodes = [n for n in tree.body if isinstance(n, ast.For)]

    if not for_nodes:
        # Fallback for plain `assert ...` style tests
        preamble_nodes: List[ast.stmt] = []
        assertions: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assert):
                assertions.append(ast.unparse(node))
            else:
                preamble_nodes.append(node)
        preamble = "\n".join(ast.unparse(n) for n in preamble_nodes)
        if not assertions and test_list:
            assertions = list(test_list)
        entry_point = _extract_entry_point(ast.Pass(), canonical_solution)
        return preamble, assertions, entry_point

    last_for = for_nodes[-1]
    lines = test_code.splitlines()
    # Keep exact original preamble lines before the `for` loop
    preamble_lines = lines[: last_for.lineno - 1]
    preamble = "\n".join(preamble_lines).rstrip()

    # Determine number of test cases from `inputs = [...]`
    n_inputs = 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "inputs":
                    if isinstance(node.value, ast.List):
                        n_inputs = len(node.value.elts)

    if n_inputs == 0 or not last_for.body:
        entry_point = _extract_entry_point(ast.Pass(), canonical_solution)
        return preamble, list(test_list), entry_point

    iter_src = ast.unparse(last_for.iter)
    replace_exp = "results" in iter_src
    body_stmt = last_for.body[0]
    entry_point = _extract_entry_point(body_stmt, canonical_solution)

    assertions = []
    for idx in range(n_inputs):
        cloned = copy.deepcopy(body_stmt)
        transformed = _LoopVarReplacer(idx=idx, replace_exp=replace_exp).visit(cloned)
        ast.fix_missing_locations(transformed)
        assertions.append(ast.unparse(transformed))

    return preamble, assertions, entry_point


def format_problem_prompt(
    prompt_text: str,
    first_assertion: str,
    test_imports: Optional[List[str]] = None,
) -> str:
    """Formats an mbppplus problem with thinking + patch/code instructions and signature hint."""
    imports_block = ""
    if test_imports:
        imports_str = "\n".join(test_imports)
        imports_block = f"\nExpected helper imports:\n```python\n{imports_str}\n```\n"

    return (
        "Solve the following Python programming problem.\n\n"
        f"Problem:\n{prompt_text.strip()}\n\n"
        "Sample assertion (your function MUST match this exact name and argument signature):\n"
        f"```python\n{first_assertion.strip()}\n```\n"
        f"{imports_block}\n"
        "Instructions:\n"
        "1. Reason step-by-step about the algorithm and edge cases inside <think>...</think>.\n"
        "2. Provide your complete, self-contained Python solution inside <patch>...</patch> "
        "(or a ```python ... ``` code block) after </think>.\n"
        "3. Include any necessary standard library imports inside your solution, and do not "
        "include test assertions or forbidden modules (os, sys, subprocess, etc.)."
    )


def format_chat_messages(problem: MBPPPlusProblem) -> List[Dict[str, str]]:
    """Builds chat messages suitable for `tokenizer.apply_chat_template(..., enable_thinking=True)`."""
    return [{"role": "user", "content": problem.formatted_prompt}]


def load_mbppplus_splits(
    seed: int = 42,
    train_size: int = 250,
    val_size: int = 50,
) -> Dict[str, List[MBPPPlusProblem]]:
    """Loads `evalplus/mbppplus` (`test` split, 378 problems) and returns deterministic splits:
    - `train`: 250 problems
    - `test`: 128 held-out problems
    - `val`: first 50 problems of `test`
    """
    raw_ds = load_dataset("evalplus/mbppplus", split="test")
    total = len(raw_ds)
    if train_size + val_size > total:
        raise ValueError(
            f"Requested train_size={train_size} + val_size={val_size} exceeds dataset size {total}."
        )

    problems: List[MBPPPlusProblem] = []
    for ex in raw_ds:
        test_list = list(ex.get("test_list") or [])
        test_imports = list(ex.get("test_imports") or [])
        canonical = ex.get("code") or ""
        first_assertion = test_list[0] if test_list else ""

        test_preamble, assertions, entry_point = extract_assertions_from_test(
            test_code=ex.get("test") or "",
            test_list=test_list,
            test_imports=test_imports,
            canonical_solution=canonical,
        )

        formatted_prompt = format_problem_prompt(
            prompt_text=ex.get("prompt") or "",
            first_assertion=first_assertion,
            test_imports=test_imports,
        )

        problems.append(
            MBPPPlusProblem(
                task_id=int(ex["task_id"]),
                prompt=ex.get("prompt") or "",
                formatted_prompt=formatted_prompt,
                entry_point=entry_point,
                signature_hint=first_assertion,
                test_preamble=test_preamble,
                assertions=assertions,
                test_list=test_list,
                test_imports=test_imports,
                canonical_solution=canonical,
            )
        )

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    shuffled = [problems[i] for i in indices]

    train_dataset = shuffled[:train_size]
    test_dataset = shuffled[train_size:]
    val_dataset = test_dataset[:val_size]

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
