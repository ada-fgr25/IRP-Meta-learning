"""Tests for the lightweight Stride benchmark wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fwi.backends import build_backend
from fwi.stride_benchmark import StrideBenchmarkLayout, StrideBenchmarkRunner


class StrideBenchmarkTests(unittest.TestCase):
    """Regression tests for benchmark orchestration helpers."""

    def test_runner_builds_expected_commands_for_local_resource_layout(self):
        """The wrapper should launch the bundled scripts by filename."""

        runner = StrideBenchmarkRunner(python_executable="python")

        self.assertEqual(runner.forward_command(), ["python", "01_script_forward.py"])
        self.assertEqual(runner.inverse_command(), ["python", "02_script_inverse.py"])
        self.assertEqual(
            runner.layout.resource_dir,
            Path("experiments/stride_brain_reference"),
        )

    def test_runner_lists_zero_padded_velocity_snapshots_in_order(self):
        """Snapshot discovery should preserve the benchmark iteration order."""

        with tempfile.TemporaryDirectory() as tmpdir:
            resource_dir = Path(tmpdir)
            (resource_dir / "01_script_forward.py").write_text("", encoding="utf-8")
            (resource_dir / "02_script_inverse.py").write_text("", encoding="utf-8")
            (resource_dir / "alpha2D-Vp-00010.h5").write_text("", encoding="utf-8")
            (resource_dir / "alpha2D-Vp-00002.h5").write_text("", encoding="utf-8")
            (resource_dir / "alpha2D-Vp-00001.h5").write_text("", encoding="utf-8")

            runner = StrideBenchmarkRunner(
                layout=StrideBenchmarkLayout(resource_dir=resource_dir),
            )
            snapshots = runner.list_velocity_snapshots()

            self.assertEqual(
                [path.name for path in snapshots],
                [
                    "alpha2D-Vp-00001.h5",
                    "alpha2D-Vp-00002.h5",
                    "alpha2D-Vp-00010.h5",
                ],
            )

    def test_stride_backend_exposes_shared_acquisition_metadata(self):
        """The benchmark path should fit the shared acquisition API surface."""

        acquisition = build_backend("stride").build_acquisition(config=None)

        self.assertEqual(acquisition.geometry_type, "elliptical")
        self.assertEqual(acquisition.n_transducers, 256)
        self.assertEqual(acquisition.n_time_samples, 2500)


if __name__ == "__main__":
    unittest.main()
