"""Runs the full CPU verification suite in order.

    python scripts/verify_all.py

No GPU and no model download required. Each phase is an independent script; this
runner simply stops on the first failure and prints a summary.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PHASES = [
    ("Phase 1", "verify_phase1.py", "dataset splits, assertion extraction, sandbox reward"),
    ("Phase 2", "verify_phase2.py", "rollout masking, log-prob equivalence, Dr. GRPO step"),
    ("Phase 3", "verify_phase3.py", "base eval, training, checkpoint, resume, comparison table"),
]


def main() -> int:
    results = []
    for label, script, description in PHASES:
        path = os.path.join(HERE, script)
        print(f"\n{'=' * 70}\n{label}: {description}\n{'=' * 70}", flush=True)
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, path], cwd=os.path.dirname(HERE))
        elapsed = time.perf_counter() - started
        results.append((label, proc.returncode, elapsed))
        if proc.returncode != 0:
            print(f"\n{label} FAILED (exit {proc.returncode}) after {elapsed:.1f}s")
            break

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for label, code, elapsed in results:
        status = "PASS" if code == 0 else f"FAIL (exit {code})"
        print(f"  {label}: {status}  [{elapsed:.1f}s]")

    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
