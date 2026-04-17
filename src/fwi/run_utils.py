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


def write_run_state_marker(
    output_dir: Path,
    optimizer: str,
    *,
    state: str,
    steps: int,
    max_freqs_hz: tuple[float, ...],
    message: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> Path:
    """Persist a coarse run-state marker such as RUNNING or FAILED.

    These markers are intentionally lightweight. They let us tell at a glance
    whether the latest run for one optimiser is still in progress, completed
    cleanly, or died before the final completion marker was written.
    """

    state_upper = state.upper()
    state_lower = state.lower()
    marker_path = output_dir / f"{optimizer}_{state_upper}.json"
    payload = {
        "status": state_lower,
        "state": state_upper,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": optimizer,
        "steps": int(steps),
        "max_freqs_hz": [float(v) for v in max_freqs_hz],
    }
    if message is not None:
        payload["message"] = str(message)
    if artifacts:
        payload["artifacts"] = dict(artifacts)
    with marker_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return marker_path


def clear_run_outputs(output_dir: Path, optimizer: str) -> list[Path]:
    """Remove stale artifacts for one optimiser without touching other runs.

    We intentionally clear only files owned by the current optimiser prefix so
    repeated `sgd` runs do not leave stale plots/JSON behind, while unrelated
    `adam` or `lbfgsb` artifacts in the same directory remain available for
    comparison.
    """

    patterns = (
        f"{optimizer}_metrics.json",
        f"{optimizer}_history.json",
        f"{optimizer}_history.png",
        f"{optimizer}_reconstruction.png",
        f"{optimizer}_RUNNING.json",
        f"{optimizer}_FAILED.json",
        f"{optimizer}_RUN_COMPLETE.json",
        f"{optimizer}_stage*_diagnostics.json",
        f"{optimizer}_stage*_diagnostics.png",
    )

    removed: list[Path] = []
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(path)
    return sorted(removed)
