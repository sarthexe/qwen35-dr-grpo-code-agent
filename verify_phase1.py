"""Phase 1 Verification Script for dataset.py and sandbox.py."""

from __future__ import annotations

import math
import time

from dataset import load_mbppplus_splits
from sandbox import grade_completion


def main() -> None:
    print("=== 1. Loading evalplus/mbppplus splits (seed=42) ===")
    splits = load_mbppplus_splits(seed=42)
    train_ds = splits["train"]
    val_ds = splits["val"]
    test_ds = splits["test"]

    assert len(train_ds) == 250, f"Expected 250 train, got {len(train_ds)}"
    assert len(val_ds) == 50, f"Expected 50 val, got {len(val_ds)}"
    assert len(test_ds) == 128, f"Expected 128 test, got {len(test_ds)}"
    assert [p.task_id for p in val_ds] == [
        p.task_id for p in test_ds[:50]
    ], "val_dataset must be the first 50 problems of test_dataset"

    print(
        f"Split sizes verified: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    sample_problems = train_ds[:3]
    print("\n--- 3 Real Problems from evalplus/mbppplus ---")
    for idx, prob in enumerate(sample_problems, start=1):
        print(
            f"[{idx}] task_id={prob.task_id} | entry_point={prob.entry_point} "
            f"| assertions={len(prob.assertions)} | hint={prob.signature_hint}"
        )
        assert len(prob.assertions) >= 3, f"Expected >=3 assertions for task {prob.task_id}"

    # Choose a problem where we can construct a clear partial-pass buggy solution.
    # Let's pick `is_not_prime` (task_id=3) or fallback to searching train/test for it,
    # or use sample_problems[0] for (a)/(c)/(d) and a problem with deterministic partial pass for (b).
    target_prob = next(
        (p for p in train_ds + test_ds if p.entry_point == "is_not_prime"),
        sample_problems[0],
    )
    print(
        f"\n=== 2. Testing sandbox.py on task_id={target_prob.task_id} "
        f"({target_prob.entry_point}, {len(target_prob.assertions)} assertions) ==="
    )

    # (a) Valid <think> + passing code (canonical solution)
    comp_pass = (
        f"<think>\nLet's solve {target_prob.entry_point} carefully using standard Python.\n"
        f"Even if I write `def fake(): pass` in my thoughts, it must be ignored.\n</think>\n"
        f"<patch>\n{target_prob.canonical_solution.strip()}\n</patch>"
    )
    res_a = grade_completion(
        comp_pass,
        assertions=target_prob.assertions,
        test_preamble=target_prob.test_preamble,
        turn=1,
    )
    print(
        f"(a) Passing: passed={res_a.passed_count}/{res_a.total_count}, "
        f"reward={res_a.reward:.4f}, all_passed={res_a.all_passed}"
    )
    assert res_a.format_valid
    assert res_a.all_passed
    assert res_a.passed_count == res_a.total_count
    assert math.isclose(res_a.reward, 1.0, rel_tol=1e-6)
    assert res_a.obs_message == ""

    # (b) Valid <think> + buggy code that passes only a subset of assertions
    if target_prob.entry_point == "is_not_prime":
        # Returns True for even numbers > 2 and False otherwise:
        # passes on primes (except 2) and even composites, fails on 1, 2, odd composites (35, etc.)
        buggy_code = (
            "def is_not_prime(n):\n"
            "    if n <= 2:\n"
            "        return False\n"
            "    return n % 2 == 0\n"
        )
    else:
        buggy_code = f"def {target_prob.entry_point}(*args, **kwargs):\n    return None\n"

    comp_buggy = (
        "<think>\nI think checking even numbers might handle most cases.\n</think>\n"
        f"```python\n{buggy_code}```"
    )
    res_b = grade_completion(
        comp_buggy,
        assertions=target_prob.assertions,
        test_preamble=target_prob.test_preamble,
        turn=1,
    )
    print(
        f"(b) Partial buggy: passed={res_b.passed_count}/{res_b.total_count}, "
        f"reward={res_b.reward:.4f}"
    )
    print(f"    <obs> preview: {res_b.obs_message.splitlines()[1][:120]}...")
    assert res_b.format_valid
    assert not res_b.all_passed
    assert 0 < res_b.passed_count < res_b.total_count, (
        f"Expected partial credit, got {res_b.passed_count}/{res_b.total_count}"
    )
    expected_b_reward = 0.15 + 0.75 * (res_b.passed_count / res_b.total_count)
    assert math.isclose(res_b.reward, expected_b_reward, rel_tol=1e-6)
    assert "<obs>" in res_b.obs_message and "</obs>" in res_b.obs_message
    assert "Analyze the test failure above and output the corrected code." in res_b.obs_message

    # (c) Code containing forbidden `import os`
    comp_forbidden = (
        "<think>\nLet me inspect the environment using os.\n</think>\n"
        "<patch>\nimport os\n"
        f"def {target_prob.entry_point}(*args):\n    return os.getcwd()\n"
        "</patch>"
    )
    res_c = grade_completion(
        comp_forbidden,
        assertions=target_prob.assertions,
        test_preamble=target_prob.test_preamble,
        turn=1,
    )
    print(
        f"(c) Forbidden import os: passed={res_c.passed_count}/{res_c.total_count}, "
        f"reward={res_c.reward:.4f}, violation={res_c.security_violation}"
    )
    assert res_c.format_valid
    assert res_c.passed_count == 0
    assert res_c.security_violation is not None and "os" in res_c.security_violation
    assert math.isclose(res_c.reward, 0.15, rel_tol=1e-6)
    assert "<obs>SecurityError:" in res_c.obs_message

    # (d) Infinite `while True:` loop -> must terminate at ~2.0s timeout
    comp_loop = (
        "<think>\nLet me write a loop.\n</think>\n"
        f"<patch>\ndef {target_prob.entry_point}(*args):\n    while True:\n        pass\n</patch>"
    )
    t0 = time.perf_counter()
    res_d = grade_completion(
        comp_loop,
        assertions=target_prob.assertions,
        test_preamble=target_prob.test_preamble,
        turn=1,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"(d) Infinite loop: passed={res_d.passed_count}/{res_d.total_count}, "
        f"reward={res_d.reward:.4f}, timed_out={res_d.timed_out}, elapsed={elapsed:.2f}s"
    )
    assert res_d.timed_out
    assert res_d.passed_count == 0
    assert 1.8 <= elapsed <= 4.0, f"Unexpected timeout elapsed: {elapsed:.2f}s"
    assert math.isclose(res_d.reward, 0.15, rel_tol=1e-6)
    assert "<obs>TimeoutError:" in res_d.obs_message

    # (e) Extra checks: missing <think>, Turn-2 duplicate penalty
    res_no_think = grade_completion(
        f"<patch>\n{target_prob.canonical_solution}\n</patch>",
        assertions=target_prob.assertions,
        test_preamble=target_prob.test_preamble,
        turn=1,
    )
    assert res_no_think.reward == 0.0 and not res_no_think.format_valid

    res_dup_t2 = grade_completion(
        comp_buggy,
        assertions=target_prob.assertions,
        test_preamble=target_prob.test_preamble,
        turn=2,
        prev_code=res_b.extracted_code,
    )
    assert res_dup_t2.duplicate_code_penalty_applied
    assert math.isclose(res_dup_t2.reward, max(0.0, expected_b_reward - 0.10), rel_tol=1e-6)

    print("\nALL PHASE 1 VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
