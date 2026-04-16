"""Utility helpers shared by phase1 runs and tests.

These functions live under `src/fwi` so they are importable in CI setups that
set `PYTHONPATH=src` and do not expose repository-root script packages.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def select_final_metric_shot_positions(
    all_shot_indices: jnp.ndarray,
    final_shots: int | None,
    seed: int,
) -> jnp.ndarray:
    """Select deterministic shot positions used for final metric evaluation."""

    total = int(all_shot_indices.shape[0])
    if final_shots is None or final_shots <= 0 or final_shots >= total:
        return jnp.arange(total, dtype=jnp.int32)

    rng = np.random.default_rng(seed + 7_919)
    chosen = np.sort(
        rng.choice(np.arange(total, dtype=np.int32), size=final_shots, replace=False)
    )
    return jnp.asarray(chosen, dtype=jnp.int32)


def format_shot_ids_for_log(shot_ids: jnp.ndarray, max_items: int = 8) -> str:
    """Build a compact preview string for active source/shot IDs."""

    values = [int(v) for v in jnp.asarray(shot_ids).tolist()]
    if len(values) <= max_items:
        return str(values)
    head_count = max_items // 2
    tail_count = max_items - head_count
    return f"{values[:head_count]} ... {values[-tail_count:]} (total={len(values)})"


def write_run_complete_marker(
    output_dir: Path,
    optimizer: str,
    *,
    steps: int,
    max_freqs_hz: tuple[float, ...],
    metrics_path: Path,
    history_path: Path,
    reconstruction_path: Path,
    history_plot_path: Path,
) -> Path:
    """Persist an explicit completion marker for long runs."""

    marker_path = output_dir / f"{optimizer}_RUN_COMPLETE.json"
    payload = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": optimizer,
        "steps": int(steps),
        "max_freqs_hz": [float(v) for v in max_freqs_hz],
        "artifacts": {
            "metrics_json": str(metrics_path),
            "history_json": str(history_path),
            "reconstruction_png": str(reconstruction_path),
            "history_png": str(history_plot_path),
        },
    }
    with marker_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return marker_path
