"""Run the bundled Stride brain-ultrasound scripts as a benchmark workflow.

This entrypoint is intentionally separate from the differentiable JAX
experiment. The JAX path remains the research path for meta-learning, while
this script acts as a convenience wrapper around the tracked Stride reference
implementation shipped under `experiments/stride_brain_reference`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

# Keep Matplotlib's runtime cache in a writable location for shared/dev shells.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from fwi.backends import build_backend
from fwi.stride_benchmark import StrideBenchmarkLayout, StrideBenchmarkRunner


def parse_args():
    """Parse command-line options for the Stride benchmark wrapper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["forward", "inverse", "both"],
        default="both",
        help="Which part of the bundled Stride workflow to run.",
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=Path("experiments/stride_brain_reference"),
        help="Directory containing the tracked Stride reference scripts.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=str,
        default="python",
        help="Python executable used to launch the Stride scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/stride_benchmark"),
        help=(
            "Directory for benchmark run artifacts (summary JSON and PNG plots). "
            "This path is under `experiments/outputs/` and is gitignored."
        ),
    )
    return parser.parse_args()


def _load_stride_scalar_field(path: Path) -> np.ndarray:
    """Load one Stride scalar field from an HDF5 file.

    The tracked Stride model files store velocity under dataset key `data`.
    Keeping this loader tiny and explicit avoids coupling the benchmark script
    to the JAX problem module internals.
    """

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["data"][()], dtype=np.float32)


def _symmetric_limits(images: list[np.ndarray]) -> tuple[float, float]:
    """Return symmetric color limits for signed difference maps."""

    vmax = max(float(np.max(np.abs(image))) for image in images)
    return -vmax, vmax


def _save_model_snapshot_figure(
    runner: StrideBenchmarkRunner,
    output_dir: Path,
    *,
    true_model_path: Path | None = None,
    starting_model_path: Path | None = None,
) -> dict[str, str | None]:
    """Persist a figure comparing starting/true/recovered Stride models.

    The inverse script writes a sequence of recovered velocity snapshots.
    We use the latest one as the current recovered model and visualise:
    - true model
    - starting model
    - recovered model
    - recovered minus starting
    - true minus recovered
    """

    true_model_path = (
        Path("data/alpha2D-TrueModel.h5").resolve()
        if true_model_path is None
        else true_model_path.resolve()
    )
    starting_model_path = (
        Path("data/alpha2D-StartingModel.h5").resolve()
        if starting_model_path is None
        else starting_model_path.resolve()
    )
    snapshots = runner.list_velocity_snapshots()
    latest_snapshot_path = snapshots[-1] if snapshots else None

    # Without a recovered snapshot there is no parity figure to compare.
    if latest_snapshot_path is None:
        return {
            "stride_reconstruction_png": None,
            "starting_model_path": str(starting_model_path),
            "true_model_path": str(true_model_path),
            "recovered_model_path": None,
        }

    true_model = _load_stride_scalar_field(true_model_path)
    starting_model = _load_stride_scalar_field(starting_model_path)
    recovered_model = _load_stride_scalar_field(latest_snapshot_path)

    panels = [
        ("True velocity", true_model),
        ("Starting model", starting_model),
        ("Recovered model", recovered_model),
    ]
    diff_panels = [
        ("Recovered - Starting", recovered_model - starting_model),
        ("True - Recovered", true_model - recovered_model),
    ]

    abs_vmin = min(float(np.min(image)) for _, image in panels)
    abs_vmax = max(float(np.max(image)) for _, image in panels)
    diff_vmin, diff_vmax = _symmetric_limits([image for _, image in diff_panels])

    figure = plt.figure(figsize=(20, 4.2))
    axes = figure.subplots(1, len(panels) + len(diff_panels))

    for axis, (title, image) in zip(axes[: len(panels)], panels):
        im = axis.imshow(
            image.T,
            origin="lower",
            cmap="viridis",
            vmin=abs_vmin,
            vmax=abs_vmax,
        )
        figure.colorbar(im, ax=axis)
        axis.set_title(title)

    for axis, (title, image) in zip(axes[len(panels) :], diff_panels):
        im = axis.imshow(
            image.T,
            origin="lower",
            cmap="coolwarm",
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        figure.colorbar(im, ax=axis)
        axis.set_title(title)

    figure.tight_layout()
    reconstruction_path = output_dir / "stride_reconstruction.png"
    figure.savefig(reconstruction_path, dpi=150)
    plt.close(figure)

    return {
        "stride_reconstruction_png": str(reconstruction_path.resolve()),
        "starting_model_path": str(starting_model_path),
        "true_model_path": str(true_model_path),
        "recovered_model_path": str(latest_snapshot_path.resolve()),
    }


def _summary(
    runner: StrideBenchmarkRunner,
    *,
    mode: str,
    timings_s: dict[str, float],
    output_dir: Path,
) -> dict[str, object]:
    """Summarise the benchmark artefacts we know how to locate locally."""

    snapshots = runner.list_velocity_snapshots()
    acquisitions_path = runner.acquisitions_path()
    backend = build_backend("stride")
    acquisition = backend.build_acquisition(config=None)
    figures = _save_model_snapshot_figure(runner, output_dir)
    return {
        "resource_dir": str(runner.layout.resource_dir.resolve()),
        "mode": mode,
        "shared_acquisition": {
            "geometry_type": acquisition.geometry_type,
            "n_transducers": acquisition.n_transducers,
            "n_shots": acquisition.n_shots,
            "n_time_samples": acquisition.n_time_samples,
        },
        "reference_settings": runner.reference_settings(),
        "acquisitions_exists": acquisitions_path.exists(),
        "acquisitions_path": str(acquisitions_path),
        "n_velocity_snapshots": len(snapshots),
        "latest_velocity_snapshot": str(snapshots[-1].resolve()) if snapshots else None,
        "timings_s": dict(timings_s),
        "figures": figures,
    }


def main():
    """Launch the requested Stride benchmark stages and print a short summary."""

    args = parse_args()
    runner = StrideBenchmarkRunner(
        layout=StrideBenchmarkLayout(resource_dir=args.resource_dir),
        python_executable=args.python_executable,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        commands = []
        if args.mode in {"forward", "both"}:
            commands.append(runner.forward_command())
        if args.mode in {"inverse", "both"}:
            commands.append(runner.inverse_command())
        print(
            json.dumps(
                {
                    "commands": commands,
                    "reference_settings": runner.reference_settings(),
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
            )
        )
        return

    timings_s: dict[str, float] = {}
    total_start = perf_counter()

    if args.mode in {"forward", "both"}:
        forward_start = perf_counter()
        runner.run_forward()
        timings_s["forward"] = perf_counter() - forward_start

    if args.mode in {"inverse", "both"}:
        inverse_start = perf_counter()
        runner.run_inverse()
        timings_s["inverse"] = perf_counter() - inverse_start

    timings_s["total"] = perf_counter() - total_start

    summary = _summary(
        runner,
        mode=args.mode,
        timings_s=timings_s,
        output_dir=args.output_dir,
    )
    summary_path = args.output_dir / "stride_benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_json"] = str(summary_path.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
