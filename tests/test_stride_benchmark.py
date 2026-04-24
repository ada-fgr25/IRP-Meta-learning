"""Tests for the lightweight Stride benchmark wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.stride_brain_benchmark import (
    _load_stride_scalar_field,
    _save_model_snapshot_figure,
)
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

    def test_scalar_field_loader_reads_stride_h5_data_dataset(self):
        """Benchmark scalar loader should read the standard `data` dataset."""

        with tempfile.TemporaryDirectory() as tmpdir:
            field_path = Path(tmpdir) / "field.h5"
            expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            with h5py.File(field_path, "w") as handle:
                handle.create_dataset("data", data=expected)

            loaded = _load_stride_scalar_field(field_path)
            self.assertTrue(np.allclose(loaded, expected))

    def test_snapshot_figure_writer_saves_reconstruction_png(self):
        """Stride benchmark helper should write a model-comparison figure."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            resource_dir = root / "stride_resource"
            output_dir = root / "outputs"
            resource_dir.mkdir()
            output_dir.mkdir()

            # Minimal script placeholders required by layout validation.
            (resource_dir / "01_script_forward.py").write_text("", encoding="utf-8")
            (resource_dir / "02_script_inverse.py").write_text("", encoding="utf-8")

            # Build tiny model fields for true/start/recovered snapshots.
            true_model_path = root / "true.h5"
            starting_model_path = root / "start.h5"
            recovered_snapshot_path = resource_dir / "alpha2D-Vp-00001.h5"

            true_model = np.asarray(
                [[2000.0, 2100.0], [2200.0, 2300.0]], dtype=np.float32
            )
            start_model = np.asarray(
                [[1500.0, 1500.0], [1500.0, 1500.0]], dtype=np.float32
            )
            recovered_model = np.asarray(
                [[1800.0, 1850.0], [1900.0, 1950.0]],
                dtype=np.float32,
            )

            for path, data in (
                (true_model_path, true_model),
                (starting_model_path, start_model),
                (recovered_snapshot_path, recovered_model),
            ):
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("data", data=data)

            runner = StrideBenchmarkRunner(
                layout=StrideBenchmarkLayout(resource_dir=resource_dir),
            )
            figure_outputs = _save_model_snapshot_figure(
                runner,
                output_dir,
                true_model_path=true_model_path,
                starting_model_path=starting_model_path,
            )

            self.assertIsNotNone(figure_outputs["stride_reconstruction_png"])
            self.assertTrue(Path(figure_outputs["stride_reconstruction_png"]).exists())
            self.assertEqual(
                Path(figure_outputs["recovered_model_path"]).resolve(),
                recovered_snapshot_path.resolve(),
            )

    def test_snapshot_figure_writer_handles_missing_recovered_snapshot(self):
        """Figure output metadata should be explicit when no snapshot exists."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            resource_dir = root / "stride_resource"
            output_dir = root / "outputs"
            resource_dir.mkdir()
            output_dir.mkdir()

            (resource_dir / "01_script_forward.py").write_text("", encoding="utf-8")
            (resource_dir / "02_script_inverse.py").write_text("", encoding="utf-8")

            true_model_path = root / "true.h5"
            starting_model_path = root / "start.h5"
            with h5py.File(true_model_path, "w") as handle:
                handle.create_dataset("data", data=np.ones((2, 2), dtype=np.float32))
            with h5py.File(starting_model_path, "w") as handle:
                handle.create_dataset("data", data=np.ones((2, 2), dtype=np.float32))

            runner = StrideBenchmarkRunner(
                layout=StrideBenchmarkLayout(resource_dir=resource_dir),
            )
            figure_outputs = _save_model_snapshot_figure(
                runner,
                output_dir,
                true_model_path=true_model_path,
                starting_model_path=starting_model_path,
            )

            self.assertIsNone(figure_outputs["stride_reconstruction_png"])
            self.assertIsNone(figure_outputs["recovered_model_path"])


if __name__ == "__main__":
    unittest.main()
