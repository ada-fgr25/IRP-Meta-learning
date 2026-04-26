"""Tests for the lightweight Stride benchmark wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.stride_brain_benchmark import (
    _load_stride_scalar_field,
    _parse_stride_iteration_losses,
    _save_model_snapshot_figure,
    _save_stride_history_figure,
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

    def test_loss_parser_extracts_stagewise_totals_from_head_log(self):
        """Stride head-log parsing should recover per-stage total-loss series."""

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "head.log"
            log_path.write_text(
                "\n".join(
                    [
                        "Done iteration 1 (out of 2), block 1 (out of 2) - Total loss 1.000000e+02",
                        "Done iteration 2 (out of 2), block 1 (out of 2) - Total loss 8.000000e+01",
                        "Done iteration 1 (out of 2), block 2 (out of 2) - Total loss 6.000000e+01",
                        "Done iteration 2 (out of 2), block 2 (out of 2) - Total loss 4.000000e+01",
                    ]
                ),
                encoding="utf-8",
            )

            losses = _parse_stride_iteration_losses(
                log_path,
                num_blocks=2,
                num_iters_per_block=2,
            )

            self.assertEqual(losses, [[100.0, 80.0], [60.0, 40.0]])

    def test_history_figure_writer_saves_stride_history_png(self):
        """History writer should emit a Stride-stage loss/RMSE figure."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            resource_dir = root / "stride_resource"
            output_dir = root / "outputs"
            mosaic_workspace = resource_dir / "mosaic-workspace"
            resource_dir.mkdir()
            output_dir.mkdir()
            mosaic_workspace.mkdir()

            (resource_dir / "01_script_forward.py").write_text("", encoding="utf-8")
            (resource_dir / "02_script_inverse.py").write_text("", encoding="utf-8")

            # Create 24 snapshots to match the default 3 blocks x 8 iters layout.
            for idx in range(1, 25):
                snapshot_path = resource_dir / f"alpha2D-Vp-{idx:05d}.h5"
                value = 1500.0 + idx
                with h5py.File(snapshot_path, "w") as handle:
                    handle.create_dataset(
                        "data",
                        data=np.full((2, 2), value, dtype=np.float32),
                    )

            # Minimal head.log with one loss per iteration.
            loss_lines = []
            for block in range(1, 4):
                for iteration in range(1, 9):
                    total_loss = float(1000 - (block - 1) * 100 - iteration)
                    loss_lines.append(
                        "Done iteration "
                        f"{iteration} (out of 8), block {block} (out of 3) - "
                        f"Total loss {total_loss:.6e}"
                    )
            (mosaic_workspace / "head.log").write_text(
                "\n".join(loss_lines),
                encoding="utf-8",
            )

            true_model_path = root / "true.h5"
            with h5py.File(true_model_path, "w") as handle:
                handle.create_dataset(
                    "data",
                    data=np.full((2, 2), 1500.0, dtype=np.float32),
                )

            runner = StrideBenchmarkRunner(
                layout=StrideBenchmarkLayout(resource_dir=resource_dir),
            )
            history = _save_stride_history_figure(
                runner,
                output_dir,
                true_model_path=true_model_path,
            )

            self.assertTrue(Path(history["stride_history_png"]).exists())
            self.assertEqual(len(history["loss_by_stage"]), 3)
            self.assertEqual(len(history["loss_by_stage"][0]), 8)
            self.assertEqual(len(history["rmse_by_stage"]), 3)
            self.assertEqual(len(history["rmse_by_stage"][0]), 8)


if __name__ == "__main__":
    unittest.main()
