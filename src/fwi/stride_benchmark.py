"""Helpers for running the local Stride benchmark workflow.

This module deliberately keeps the Stride benchmark separate from the
differentiable JAX backend API. Stride is useful to benchmark a higher-fidelity
reference workflow that is already present under `resources/stride_fwi_brain`,
but it is not part of the end-to-end JAX autodiff path used for meta-learning.

The main job of this module is therefore orchestration:

* locate the local Stride reference scripts and data
* run the forward and inverse scripts from the correct working directory
* report the artefacts produced by those benchmark runs

Keeping this logic in one place makes the benchmark reproducible without
entangling it with the differentiable research code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class StrideBenchmarkLayout:
    """Describe the local directory layout of the bundled Stride benchmark."""

    resource_dir: Path = Path("experiments/stride_brain_reference")
    forward_script: str = "01_script_forward.py"
    inverse_script: str = "02_script_inverse.py"

    def forward_script_path(self) -> Path:
        """Return the absolute path to the bundled forward script."""

        return (self.resource_dir / self.forward_script).resolve()

    def inverse_script_path(self) -> Path:
        """Return the absolute path to the bundled inverse script."""

        return (self.resource_dir / self.inverse_script).resolve()

    def validate(self) -> None:
        """Ensure the benchmark resources are present before we launch them."""

        if not self.resource_dir.exists():
            raise FileNotFoundError(
                "Stride benchmark resource directory not found at "
                f"{self.resource_dir!s}."
            )

        for script_path in (self.forward_script_path(), self.inverse_script_path()):
            if not script_path.exists():
                raise FileNotFoundError(
                    "Expected Stride benchmark script not found at " f"{script_path!s}."
                )


@dataclass(frozen=True)
class StrideBenchmarkRunner:
    """Run the tracked Stride forward and inverse reference scripts.

    The runner assumes the Stride scripts should execute from inside their own
    resource directory because they reference local `data/` files with relative
    paths. We therefore always launch them with `cwd=resource_dir`.
    """

    layout: StrideBenchmarkLayout = StrideBenchmarkLayout()
    python_executable: str = sys.executable

    def _build_command(self, script_path: Path) -> list[str]:
        """Construct the Python command used to launch one Stride script."""

        return [self.python_executable, script_path.name]

    def forward_command(self) -> list[str]:
        """Return the command that launches the bundled forward script."""

        self.layout.validate()
        return self._build_command(self.layout.forward_script_path())

    def inverse_command(self) -> list[str]:
        """Return the command that launches the bundled inverse script."""

        self.layout.validate()
        return self._build_command(self.layout.inverse_script_path())

    def run_forward(self, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Execute the bundled Stride forward script."""

        return subprocess.run(
            self.forward_command(),
            cwd=self.layout.resource_dir,
            check=check,
            text=True,
        )

    def run_inverse(self, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Execute the bundled Stride inverse script."""

        return subprocess.run(
            self.inverse_command(),
            cwd=self.layout.resource_dir,
            check=check,
            text=True,
        )

    def list_velocity_snapshots(self) -> list[Path]:
        """Return the velocity snapshots written by the Stride inverse script.

        The inverse reference currently writes files such as
        `alpha2D-Vp-00001.h5`, `alpha2D-Vp-00002.h5`, and so on. Sorting the
        filenames lexicographically is enough because the snapshot numbers are
        zero-padded.
        """

        self.layout.validate()
        return sorted(self.layout.resource_dir.glob("alpha2D-Vp-*.h5"))

    def acquisitions_path(self) -> Path:
        """Return the observed-data file produced by the Stride forward script."""

        self.layout.validate()
        return (self.layout.resource_dir / "alpha2D-Acquisitions.h5").resolve()

    def reference_settings(self) -> dict[str, object]:
        """Describe the benchmark settings encoded in the bundled scripts.

        These values are intentionally documented here because the wrapper does
        not reimplement the benchmark itself; it launches the tracked Stride
        scripts verbatim. Making the expected settings explicit helps other
        researchers understand what they are reproducing when they use this
        wrapper with the default resource directory.
        """

        self.layout.validate()
        return {
            "depends_on_bundled_stride_scripts": True,
            "forward_script": str(self.layout.forward_script_path()),
            "inverse_script": str(self.layout.inverse_script_path()),
            "geometry": "elliptical",
            "num_locations": 256,
            "space_shape": [500, 370],
            "space_extra": [50, 50],
            "space_absorbing": [40, 40],
            "space_spacing_m": [0.5e-3, 0.5e-3],
            "time_start_s": 0.0,
            "time_step_s": 0.08e-6,
            "time_num": 2500,
            "source_centre_frequency_hz": 0.25e6,
            "source_cycles": 3,
            "inverse_num_blocks": 3,
            "inverse_max_freqs_hz": [0.1e6, 0.2e6, 0.3e6],
            "inverse_num_iters_per_block": 8,
            "inverse_num_shots_per_iter": 32,
            "inverse_step_size": 5.0,
            "platform": "cpu",
            "kernel": "OT4",
            "interpolation_type": "hicks",
            "fw3d_mode": True,
        }
