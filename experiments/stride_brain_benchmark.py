"""Run the bundled Stride brain-ultrasound scripts as a benchmark workflow.

This entrypoint is intentionally separate from the differentiable JAX
experiment. The JAX path remains the research path for meta-learning, while
this script acts as a convenience wrapper around the local Stride reference
implementation shipped under `resources/stride_fwi_brain`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        default=Path("resources/stride_fwi_brain"),
        help="Directory containing the local Stride reference scripts and data.",
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
    return parser.parse_args()


def _summary(runner: StrideBenchmarkRunner) -> dict[str, object]:
    """Summarise the benchmark artefacts we know how to locate locally."""

    snapshots = runner.list_velocity_snapshots()
    acquisitions_path = runner.acquisitions_path()
    return {
        "resource_dir": str(runner.layout.resource_dir.resolve()),
        "acquisitions_exists": acquisitions_path.exists(),
        "acquisitions_path": str(acquisitions_path),
        "n_velocity_snapshots": len(snapshots),
        "latest_velocity_snapshot": str(snapshots[-1].resolve()) if snapshots else None,
    }


def main():
    """Launch the requested Stride benchmark stages and print a short summary."""

    args = parse_args()
    runner = StrideBenchmarkRunner(
        layout=StrideBenchmarkLayout(resource_dir=args.resource_dir),
        python_executable=args.python_executable,
    )

    if args.dry_run:
        commands = []
        if args.mode in {"forward", "both"}:
            commands.append(runner.forward_command())
        if args.mode in {"inverse", "both"}:
            commands.append(runner.inverse_command())
        print(json.dumps({"commands": commands}, indent=2))
        return

    if args.mode in {"forward", "both"}:
        runner.run_forward()

    if args.mode in {"inverse", "both"}:
        runner.run_inverse()

    print(json.dumps(_summary(runner), indent=2))


if __name__ == "__main__":
    main()
