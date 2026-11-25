"""Adds `src/` to `sys.path` so the scripts run from a fresh clone (and on Colab)
without requiring `pip install -e .` first.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
