"""Pytest bootstrap for local import paths.

This repository keeps importable library code under ``src/`` while some test
cases also import experiment entrypoints from ``experiments/`` at repo root.
When pytest is executed without ``PYTHONPATH``, those imports can fail during
collection. We add both paths explicitly so test invocation is stable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_path(path: Path) -> None:
    """Prepend ``path`` to ``sys.path`` when it is not already present."""

    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


REPO_ROOT = Path(__file__).resolve().parents[1]
_ensure_path(REPO_ROOT)
_ensure_path(REPO_ROOT / "src")
