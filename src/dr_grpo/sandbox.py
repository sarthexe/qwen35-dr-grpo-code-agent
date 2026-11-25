"""AST-validated subprocess unit-test sandbox and Dr. GRPO reward shaping."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

BLOCKED_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "http",
        "pathlib",
        "importlib",
    }
)

BLOCKED_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
    }
)

ALL_BLOCKED_NAMES = BLOCKED_MODULES | BLOCKED_BUILTINS

EXECUTION_TIMEOUT_SEC = 2.0
MAX_TRACEBACK_CHARS = 1500

FORMAT_BONUS = 0.15
TEST_CREDIT_WEIGHT = 0.75
TURN1_COMPLETION_BONUS = 0.10
DUPLICATE_CODE_PENALTY = 0.10

_RESULT_PREFIX = "__SANDBOX_RESULT__:"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_PATCH_RE = re.compile(r"<patch>(.*?)</patch>", re.DOTALL)
_PY_FENCE_RE = re.compile(r"```python\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_FENCE_RE = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)


@dataclass
class ParsedCompletion:
    has_think: bool
    think_content: str
    has_code: bool
    code: str

    @property
    def format_valid(self) -> bool:
        return self.has_think and self.has_code and bool(self.code.strip())


@dataclass
class SandboxResult:
    reward: float
    passed_count: int
    total_count: int
    pass_ratio: float
    all_passed: bool
    format_valid: bool
    has_think: bool
    has_code: bool
    think_content: str
    extracted_code: str
    first_failure_traceback: str
    obs_message: str
    security_violation: Optional[str]
    timed_out: bool
    duplicate_code_penalty_applied: bool

    @property
    def next_prompt_suffix(self) -> str:
        return self.obs_message


def _strip_inner_fences(text: str) -> str:
    """Unwraps optional ```python ... ``` fences inside a <patch> block."""
    stripped = text.strip()
    m = _PY_FENCE_RE.search(stripped)
    if m:
        return m.group(1).strip()
    m = _GENERIC_FENCE_RE.search(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def parse_completion(completion: str) -> ParsedCompletion:
    """Parses `<think>...</think>` and `<patch>...</patch>` or ```python ... ``` blocks.

    All `<think>...</think>` sections are stripped before searching for code blocks
    so scratchpad code inside reasoning is never mistaken for the final solution.
    """
    if not completion:
        return ParsedCompletion(
            has_think=False, think_content="", has_code=False, code=""
        )

    think_matches = _THINK_RE.findall(completion)
    think_content = "\n".join(m.strip() for m in think_matches if m.strip()).strip()
    has_think = bool(think_matches) and bool(think_content)

    # Strip <think>...</think> blocks before looking for code
    outside_think = _THINK_RE.sub("", completion)

    code = ""
    has_code = False

    patch_matches = _PATCH_RE.findall(outside_think)
    if patch_matches:
        candidate = _strip_inner_fences(patch_matches[-1])
        if candidate:
            code = candidate
            has_code = True

    if not has_code:
        py_matches = _PY_FENCE_RE.findall(outside_think)
        if py_matches:
            candidate = py_matches[-1].strip()
            if candidate:
                code = candidate
                has_code = True

    if not has_code:
        gen_matches = _GENERIC_FENCE_RE.findall(outside_think)
        if gen_matches:
            candidate = gen_matches[-1].strip()
            if candidate:
                code = candidate
                has_code = True

    return ParsedCompletion(
        has_think=has_think,
        think_content=think_content,
        has_code=has_code,
        code=code,
    )


def validate_code_ast(code: str) -> Tuple[bool, Optional[str]]:
    """Validates candidate Python code against syntax errors and blocked modules/builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno}, offset {exc.offset})"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in ALL_BLOCKED_NAMES:
                    return False, f"SecurityError: Forbidden import '{alias.name}'"

        elif isinstance(node, ast.ImportFrom):
            root_pkg = (node.module or "").split(".")[0]
            if root_pkg in ALL_BLOCKED_NAMES:
                return False, f"SecurityError: Forbidden import from '{node.module}'"
            for alias in node.names:
                if alias.name in ALL_BLOCKED_NAMES:
                    return (
                        False,
                        f"SecurityError: Forbidden imported symbol '{alias.name}'",
                    )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ALL_BLOCKED_NAMES:
                return (
                    False,
                    f"SecurityError: Forbidden call to '{node.func.id}'",
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in BLOCKED_BUILTINS
            ):
                return (
                    False,
                    f"SecurityError: Forbidden call to '.{node.func.attr}'",
                )

        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id in ALL_BLOCKED_NAMES:
                return False, f"SecurityError: Forbidden identifier '{node.id}'"

        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_BUILTINS:
                return False, f"SecurityError: Forbidden attribute '{node.attr}'"

    return True, None


def _indent_block(stmt: str, spaces: int = 8) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in stmt.splitlines())


def _truncate_tb(tb: str, max_chars: int = MAX_TRACEBACK_CHARS) -> str:
    tb = tb.strip()
    if len(tb) <= max_chars:
        return tb
    return tb[:max_chars] + "\n... [traceback truncated]"


def _build_runner_script(
    code: str,
    assertions: List[str],
    test_preamble: str = "",
) -> str:
    """Builds a self-contained runner script wrapping every assertion in try/except."""
    header = """\
import json as __sandbox_json
import sys as __sandbox_sys
import traceback as __sandbox_tb

try:
    import numpy as np  # noqa: F401
except ImportError:
    import math as __sandbox_math
    import types as __sandbox_types

    np = __sandbox_types.ModuleType("numpy")

    class _NDArray:
        pass

    np.ndarray = _NDArray
    np.float32 = float
    np.float64 = float

    def _allclose(a, b, rtol=1e-05, atol=1e-08):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(
                _allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b)
            )
        try:
            return __sandbox_math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
        except (TypeError, ValueError):
            return a == b

    np.allclose = _allclose
    __sandbox_sys.modules["numpy"] = np
    del __sandbox_math, __sandbox_types
"""
    parts = [header]
    parts.append("# --- Candidate Solution ---")
    parts.append(code)
    if test_preamble.strip():
        parts.append("\n# --- Test Preamble ---")
        parts.append(test_preamble)

    total = len(assertions)
    parts.append(
        f"\n# --- Per-Assertion Harness ---\n"
        f"__sandbox_passed = 0\n"
        f"__sandbox_total = {total}\n"
        f"__sandbox_first_tb = None\n"
    )

    for idx, stmt in enumerate(assertions):
        indented_stmt = _indent_block(stmt, spaces=4)
        stmt_repr = repr(stmt)
        block = (
            f"try:\n"
            f"{indented_stmt}\n"
            f"    __sandbox_passed += 1\n"
            f"except BaseException:\n"
            f"    if __sandbox_first_tb is None:\n"
            f"        __sandbox_inp_hint = ''\n"
            f"        if 'inputs' in globals() and isinstance(inputs, list) and {idx} < len(inputs):\n"
            f"            __sandbox_inp_hint = f'Input args: {{inputs[{idx}]!r}}\\n'\n"
            f"        __sandbox_first_tb = (\n"
            f"            f'Failed at test #{idx + 1}/{total}: ' + {stmt_repr} + '\\n'\n"
            f"            + __sandbox_inp_hint\n"
            f"            + __sandbox_tb.format_exc()\n"
            f"        )\n"
        )
        parts.append(block)

    parts.append(
        f'print("{_RESULT_PREFIX}" + __sandbox_json.dumps({{\n'
        f'    "passed": __sandbox_passed,\n'
        f'    "total": __sandbox_total,\n'
        f'    "first_tb": __sandbox_first_tb or "",\n'
        f"}}))\n"
    )
    return "\n".join(parts)


def run_assertions_in_sandbox(
    code: str,
    assertions: List[str],
    test_preamble: str = "",
    timeout: float = EXECUTION_TIMEOUT_SEC,
) -> Tuple[int, int, str]:
    """Executes `code` against `assertions` in an isolated subprocess (`python -I`).

    Returns `(passed_count, total_count, first_failure_traceback)`.
    """
    total_count = len(assertions)
    if total_count == 0:
        return 0, 0, "No assertions provided."

    ok, err = validate_code_ast(code)
    if not ok:
        return 0, total_count, err or "SecurityError: Code failed AST validation."

    script = _build_runner_script(
        code=code, assertions=assertions, test_preamble=test_preamble
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False,
        ) as tmp_file:
            tmp_file.write(script)
            tmp_path = tmp_file.name

        proc = subprocess.run(
            [sys.executable, "-I", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            0,
            total_count,
            f"TimeoutError: Execution timed out after {timeout:.1f}s (possible infinite loop).",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            payload = line[len(_RESULT_PREFIX) :]
            try:
                data = json.loads(payload)
                passed = int(data.get("passed", 0))
                total = int(data.get("total", total_count))
                first_tb = _truncate_tb(str(data.get("first_tb", "")))
                return passed, total, first_tb
            except (json.JSONDecodeError, ValueError, TypeError):
                break

    # Top-level runtime error before reaching the assertion loop (e.g. NameError / exit)
    err_out = proc.stderr.strip() or proc.stdout.strip()
    if not err_out:
        err_out = f"RuntimeError: Subprocess exited with code {proc.returncode} without test output."
    return 0, total_count, _truncate_tb(err_out)


def format_observation(traceback_text: str) -> str:
    """Formats the Turn-2 repair observation prompt block."""
    return (
        f"\n<obs>{traceback_text.strip()}</obs>\n"
        "Analyze the test failure above and output the corrected code."
    )


def grade_completion(
    completion: str,
    assertions: List[str],
    test_preamble: str = "",
    turn: int = 1,
    prev_code: Optional[str] = None,
    timeout: float = EXECUTION_TIMEOUT_SEC,
) -> SandboxResult:
    """Parses, AST-validates, runs unit tests, and computes the shaped Dr. GRPO reward."""
    parsed = parse_completion(completion)
    total_count = len(assertions)

    if not parsed.format_valid:
        tb = (
            "FormatError: Missing valid <think>...</think> reasoning block "
            "or code block (<patch>...</patch> or ```python ... ```)."
        )
        return SandboxResult(
            reward=0.0,
            passed_count=0,
            total_count=total_count,
            pass_ratio=0.0,
            all_passed=False,
            format_valid=False,
            has_think=parsed.has_think,
            has_code=parsed.has_code,
            think_content=parsed.think_content,
            extracted_code=parsed.code,
            first_failure_traceback=tb,
            obs_message=format_observation(tb),
            security_violation=None,
            timed_out=False,
            duplicate_code_penalty_applied=False,
        )

    ok, sec_or_syntax_err = validate_code_ast(parsed.code)
    security_violation = (
        sec_or_syntax_err
        if (not ok and sec_or_syntax_err and sec_or_syntax_err.startswith("SecurityError:"))
        else None
    )

    passed_count, total_count, first_tb = run_assertions_in_sandbox(
        code=parsed.code,
        assertions=assertions,
        test_preamble=test_preamble,
        timeout=timeout,
    )

    pass_ratio = (passed_count / total_count) if total_count > 0 else 0.0
    all_passed = total_count > 0 and passed_count == total_count
    timed_out = first_tb.startswith("TimeoutError:")

    reward = FORMAT_BONUS + TEST_CREDIT_WEIGHT * pass_ratio
    if turn == 1 and all_passed:
        reward += TURN1_COMPLETION_BONUS

    duplicate_penalty = False
    if turn == 2 and prev_code is not None and parsed.code.strip() == prev_code.strip():
        reward -= DUPLICATE_CODE_PENALTY
        duplicate_penalty = True

    reward = max(0.0, min(1.0, reward))
    obs_message = "" if all_passed else format_observation(first_tb)

    return SandboxResult(
        reward=reward,
        passed_count=passed_count,
        total_count=total_count,
        pass_ratio=pass_ratio,
        all_passed=all_passed,
        format_valid=True,
        has_think=parsed.has_think,
        has_code=parsed.has_code,
        think_content=parsed.think_content,
        extracted_code=parsed.code,
        first_failure_traceback=first_tb,
        obs_message=obs_message,
        security_violation=security_violation,
        timed_out=timed_out,
        duplicate_code_penalty_applied=duplicate_penalty,
    )
