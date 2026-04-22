"""Smoke tests for the Phase 1 JAX FWI baseline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp

from experiments.phase1_brain_fwi import _build_random_shot_schedule
from fwi.acoustics import (
    _apply_stride_like_time_window,
    _process_trace_pair_for_stride_misfit,
    _build_boundary_terms,
    _prepare_source_wavelet,
    _source_scale,
    _stride_like_shift_traces,
    _stride_like_source_window,
    loss_and_grad,
    shot_loss_from_traces,
    simulate_survey,
    simulate_survey_forward_only,
)
from fwi.filtering import bandlimit_traces
from fwi.acoustics import _pad_model_for_solver
from fwi.backends import build_backend
from fwi.config import (
    AcquisitionConfig,
    BrainFWIConfig,
    GridConfig,
    ModelConfig,
    SolverConfig,
    TimeConfig,
)
from fwi.medium import build_acoustic_medium
from fwi.optimisers import process_global_gradient_stride_like
from fwi.problem import (
    build_brain_fwi_problem,
    dldx,
    forward,
    init_params,
    loss,
    smooth_traces,
)
from fwi.run_utils import (
    clear_run_outputs,
    format_shot_ids_for_log,
    select_final_metric_shot_positions,
    write_run_complete_marker,
    write_run_state_marker,
)


def _tiny_config() -> BrainFWIConfig:
    """Return a small configuration that keeps tests lightweight."""

    return BrainFWIConfig(
        grid=GridConfig(nx=32, ny=24),
        time=TimeConfig(nt=40),
        acquisition=AcquisitionConfig(n_transducers=12, n_shots=3),
        model=ModelConfig(source="procedural"),
        solver=SolverConfig(extra_cells_x=6, extra_cells_y=6, damping_cells=4),
    )


def _tiny_medium_config() -> BrainFWIConfig:
    """Return a small configuration with density and attenuation enabled."""

    base = _tiny_config()
    return BrainFWIConfig(
        grid=base.grid,
        time=base.time,
        acquisition=base.acquisition,
        model=replace(
            base.model,
            density_model="piecewise",
            attenuation_model="piecewise",
        ),
        solver=base.solver,
    )


class Phase1BrainFWITests(unittest.TestCase):
    """Minimal regression tests for the differentiable FWI baseline."""

    def test_forward_problem_shapes(self):
        """Observed data should have the expected [shot, time, receiver] shape."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        y_obs = params["y_obs"]

        self.assertEqual(y_obs.shape, (3, 40, 12))
        self.assertEqual(params["acquisition"].n_shots, 3)
        self.assertEqual(params["acquisition"].n_receivers, 12)
        self.assertIn("medium", params)

    def test_forward_only_survey_matches_checkpointed_forward(self):
        """Forward-only mode should reproduce checkpointed forward traces."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        baseline = simulate_survey(
            params["x0"],
            params["acquisition"],
            params["config"],
            medium=params["medium"],
        )
        forward_only = simulate_survey_forward_only(
            params["x0"],
            params["acquisition"],
            params["config"],
            medium=params["medium"],
            shot_batch_size=1,
        )

        self.assertEqual(forward_only.shape, baseline.shape)
        self.assertTrue(
            bool(jnp.allclose(forward_only, baseline, rtol=1.0e-5, atol=1.0e-6))
        )

    def test_forward_only_shot_batching_is_numerically_stable(self):
        """Shot batching should preserve forward-only survey results."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        sequential = simulate_survey_forward_only(
            params["x_exact"],
            params["acquisition"],
            params["config"],
            medium=params["medium"],
            shot_batch_size=1,
        )
        batched = simulate_survey_forward_only(
            params["x_exact"],
            params["acquisition"],
            params["config"],
            medium=params["medium"],
            shot_batch_size=2,
        )

        self.assertEqual(batched.shape, sequential.shape)
        self.assertTrue(
            bool(jnp.allclose(batched, sequential, rtol=1.0e-5, atol=1.0e-6))
        )

    def test_forward_only_rejects_non_positive_shot_batch_size(self):
        """Forward-only survey batching should validate input arguments."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        with self.assertRaises(ValueError):
            simulate_survey_forward_only(
                params["x0"],
                params["acquisition"],
                params["config"],
                medium=params["medium"],
                shot_batch_size=0,
            )

    def test_piecewise_medium_builder_returns_density_and_attenuation_fields(self):
        """Optional fixed medium fields should be constructible from velocity."""

        base = _tiny_config()
        config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=replace(
                base.model,
                density_model="piecewise",
                attenuation_model="piecewise",
            ),
            solver=base.solver,
        )
        velocity = jnp.asarray(
            [
                [config.model.background_velocity, config.model.brain_velocity],
                [config.model.skull_velocity, config.model.lesion_velocity],
            ],
            dtype=jnp.float32,
        )

        medium = build_acoustic_medium(config, velocity)

        self.assertIsNotNone(medium.density)
        self.assertIsNotNone(medium.attenuation)
        self.assertEqual(medium.density.shape, velocity.shape)
        self.assertEqual(medium.attenuation.shape, velocity.shape)
        self.assertTrue(bool(jnp.all(medium.density > 0.0)))
        self.assertTrue(bool(jnp.all(medium.attenuation >= 0.0)))

    def test_hicks_acquisition_builds_precomputed_coefficients(self):
        """Hicks mode should materialise Stride-like interpolation tensors."""

        base = _tiny_config()
        hicks_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=replace(base.acquisition, interpolation_type="hicks"),
            model=base.model,
            solver=base.solver,
        )
        params = init_params(jax.random.PRNGKey(0), config=hicks_config)
        acquisition = params["acquisition"]

        self.assertEqual(acquisition.interpolation_type, "hicks")
        self.assertIsNotNone(acquisition.source_reference_gridpoints)
        self.assertIsNotNone(acquisition.source_coefficients)
        self.assertIsNotNone(acquisition.receiver_reference_gridpoints)
        self.assertIsNotNone(acquisition.receiver_coefficients)
        self.assertEqual(acquisition.source_coefficients.shape, (12, 2, 8))
        self.assertEqual(acquisition.receiver_coefficients.shape, (12, 2, 8))

        # Regression guard: Hicks reference points should stay in grid-index
        # coordinates. If coordinate units drift (for example dividing by grid
        # spacing twice), these values become extremely large and collapse
        # interpolation to boundary-clipped samples.
        src_refs = acquisition.source_reference_gridpoints
        rec_refs = acquisition.receiver_reference_gridpoints
        self.assertGreaterEqual(int(jnp.min(src_refs[:, 0])), 0)
        self.assertGreaterEqual(int(jnp.min(src_refs[:, 1])), 0)
        self.assertLess(int(jnp.max(src_refs[:, 0])), hicks_config.grid.nx)
        self.assertLess(int(jnp.max(src_refs[:, 1])), hicks_config.grid.ny)
        self.assertGreaterEqual(int(jnp.min(rec_refs[:, 0])), 0)
        self.assertGreaterEqual(int(jnp.min(rec_refs[:, 1])), 0)
        self.assertLess(int(jnp.max(rec_refs[:, 0])), hicks_config.grid.nx)
        self.assertLess(int(jnp.max(rec_refs[:, 1])), hicks_config.grid.ny)

    def test_hicks_coordinate_epsilon_changes_reference_gridpoints(self):
        """Stride-like coordinate epsilon should affect Hicks reference points."""

        base = _tiny_config()
        # Use an exaggerated epsilon scale so this regression test is robust:
        # if epsilon handling is removed, the references collapse back to the
        # zero-epsilon case and this test catches the parity drift immediately.
        epsilon_on = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=replace(
                base.acquisition,
                interpolation_type="hicks",
                apply_coordinate_epsilon=True,
                coordinate_epsilon_scale=0.5,
            ),
            model=base.model,
            solver=base.solver,
        )
        epsilon_off = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=replace(
                base.acquisition,
                interpolation_type="hicks",
                apply_coordinate_epsilon=False,
                coordinate_epsilon_scale=0.5,
            ),
            model=base.model,
            solver=base.solver,
        )

        refs_on = init_params(jax.random.PRNGKey(0), config=epsilon_on)["acquisition"]
        refs_off = init_params(jax.random.PRNGKey(0), config=epsilon_off)["acquisition"]

        self.assertFalse(
            bool(
                jnp.allclose(
                    refs_on.source_reference_gridpoints,
                    refs_off.source_reference_gridpoints,
                )
            )
        )

    def test_stride_like_source_window_respects_time_bounds(self):
        """Source window should be non-zero only inside requested bounds."""

        window = _stride_like_source_window(
            n_time=16,
            start=3,
            stop=10,
            alpha=1.0e-3,
            dtype=jnp.float32,
        )

        self.assertEqual(window.shape, (16,))
        self.assertTrue(bool(jnp.allclose(window[:3], 0.0)))
        self.assertTrue(bool(jnp.allclose(window[10:], 0.0)))
        self.assertGreater(float(jnp.sum(window[3:10])), 0.0)

    def test_prepare_source_wavelet_applies_stride_like_windowing(self):
        """Source preparation should apply Tukey window when enabled."""

        base = _tiny_config()
        config_with_window = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                source_window_enabled=True,
                source_window_alpha=1.0e-3,
                source_window_start=4,
                source_window_stop=12,
                diff_source=False,
            ),
        )
        config_no_window = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                source_window_enabled=False,
                diff_source=False,
            ),
        )
        wavelet = jnp.ones((base.time.nt,), dtype=jnp.float32)

        with_window = _prepare_source_wavelet(wavelet, config_with_window)
        without_window = _prepare_source_wavelet(wavelet, config_no_window)

        self.assertEqual(with_window.shape, wavelet.shape)
        self.assertTrue(bool(jnp.allclose(without_window, wavelet)))
        self.assertTrue(bool(jnp.allclose(with_window[:4], 0.0)))
        self.assertTrue(bool(jnp.allclose(with_window[12:], 0.0)))
        self.assertGreater(float(jnp.sum(with_window[4:12])), 0.0)

    def test_prepare_source_wavelet_applies_stride_like_fw3d_shift(self):
        """FW3D mode should apply Stride's quarter-period wavelet shift."""

        base = _tiny_config()
        config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                source_window_enabled=False,
                stride_trace_processing=True,
                fw3d_mode=True,
                trace_filter_relaxation=0.75,
            ),
        )
        wavelet = jnp.arange(base.time.nt, dtype=jnp.float32)
        shifted = _prepare_source_wavelet(wavelet, config, f_max_hz=1.0e6)

        filtered = bandlimit_traces(
            wavelet,
            config.time.dt,
            1.0e6,
            axis=0,
            filter_type=config.solver.trace_filter_type,
            relaxation=config.solver.trace_filter_relaxation,
            order=config.solver.trace_filter_order,
            zero_phase=config.solver.trace_filter_zero_phase,
        )
        expected = _stride_like_shift_traces(filtered, config, 1.0e6, axis=0)
        self.assertTrue(bool(jnp.allclose(shifted, expected)))

    def test_shot_loss_trace_processing_path_is_differentiable(self):
        """Stride-like trace conditioning should keep shot loss differentiable."""

        base = _tiny_config()
        config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                stride_trace_processing=True,
                source_window_enabled=False,
            ),
        )
        modelled = jnp.ones(
            (base.time.nt, base.acquisition.n_transducers), dtype=jnp.float32
        )
        observed = jnp.zeros_like(modelled)
        observed = observed.at[:, : base.acquisition.n_transducers // 2].set(1.0)

        loss_value = shot_loss_from_traces(modelled, observed, config, f_max_hz=2.0e5)
        grad = jax.grad(
            lambda traces: shot_loss_from_traces(
                traces, observed, config, f_max_hz=2.0e5
            )
        )(modelled)

        self.assertGreater(float(loss_value), 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertEqual(grad.shape, modelled.shape)

    def test_scale_per_shot_path_accepts_raw_observed_reference(self):
        """ScalePerShot branch should execute with an explicit raw reference."""

        base = _tiny_config()
        config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                stride_trace_processing=True,
                stride_trace_scale_per_shot=True,
                source_window_enabled=False,
            ),
        )
        modelled = jnp.ones(
            (base.time.nt, base.acquisition.n_transducers), dtype=jnp.float32
        )
        observed_processed = jnp.full_like(modelled, 0.5)
        observed_raw = jnp.full_like(modelled, 2.0)

        with_raw_ref = _process_trace_pair_for_stride_misfit(
            modelled,
            observed_processed,
            observed_raw,
            config,
            f_max_hz=2.0e5,
        )
        without_raw_ref = _process_trace_pair_for_stride_misfit(
            modelled,
            observed_processed,
            None,
            config,
            f_max_hz=2.0e5,
        )

        self.assertEqual(with_raw_ref[0].shape, modelled.shape)
        self.assertEqual(with_raw_ref[1].shape, modelled.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(with_raw_ref[0]))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(with_raw_ref[1]))))
        # With Stride's default `relative_scale=True`, ScalePerShot is
        # effectively neutral in amplitude, so both reference choices should
        # remain numerically close in this synthetic setup.
        self.assertTrue(bool(jnp.allclose(with_raw_ref[0], without_raw_ref[0])))
        self.assertTrue(bool(jnp.allclose(with_raw_ref[1], without_raw_ref[1])))

    def test_stride_like_time_window_applies_across_trace_axis(self):
        """Configured source window should also apply to adjoint/source traces."""

        base = _tiny_config()
        config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                source_window_enabled=True,
                source_window_alpha=1.0e-3,
                source_window_start=2,
                source_window_stop=7,
            ),
        )
        traces = jnp.ones((10, 3), dtype=jnp.float32)
        windowed = _apply_stride_like_time_window(traces, config, axis=0)

        self.assertTrue(bool(jnp.allclose(windowed[:2], 0.0)))
        self.assertTrue(bool(jnp.allclose(windowed[7:], 0.0)))
        self.assertGreater(float(jnp.sum(windowed[2:7])), 0.0)

    def test_gradient_is_finite(self):
        """The differentiable solver should provide a finite adjoint signal."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        loss_value, grad = dldx(params, params["x0"], (params["y_obs"],))

        self.assertEqual(loss_value.shape, (1,))
        self.assertEqual(grad.shape, params["x0"].shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_hicks_forward_and_gradient_are_finite(self):
        """The explicit forward/adjoint path should remain stable in Hicks mode."""

        base = _tiny_config()
        hicks_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=replace(base.acquisition, interpolation_type="hicks"),
            model=base.model,
            solver=base.solver,
        )
        params = init_params(jax.random.PRNGKey(0), config=hicks_config)
        traces = forward(params, params["x_exact"])
        loss_value, grad = dldx(params, params["x0"], (params["y_obs"],))

        self.assertEqual(traces.shape, (3, 40, 12))
        self.assertTrue(bool(jnp.all(jnp.isfinite(traces))))
        self.assertEqual(loss_value.shape, (1,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        # The Hicks path should produce a meaningful residual signal for the
        # initial model, not a degenerate zero-loss/zero-gradient trajectory.
        self.assertGreater(float(loss_value.squeeze()), 0.0)
        self.assertGreater(float(jnp.linalg.norm(grad)), 0.0)

    def test_explicit_adjoint_matches_autodiff_gradient(self):
        """The explicit adjoint should agree with a direct autodiff reference."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        auxs = (params["y_obs"],)
        backend = build_backend("jax")

        explicit_value, explicit_grad = backend.loss_grad(params, params["x0"], auxs)
        autodiff_value, autodiff_grad = jax.value_and_grad(
            lambda model: loss(params, model, auxs).sum()
        )(params["x0"])

        self.assertTrue(
            bool(
                jnp.allclose(
                    explicit_value.squeeze(),
                    autodiff_value,
                    rtol=1.0e-4,
                    atol=1.0e-6,
                )
            )
        )
        self.assertTrue(
            bool(jnp.allclose(explicit_grad, autodiff_grad, rtol=5.0e-3, atol=5.0e-4))
        )

    def test_adjoint_shot_batching_matches_sequential_gradient(self):
        """Shot-batched adjoint accumulation should preserve loss and gradient."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())

        loss_seq, grad_seq = loss_and_grad(
            params["x0"],
            params["acquisition"],
            params["config"],
            params["medium"],
            params["y_obs"],
            shot_batch_size=1,
        )
        loss_batch, grad_batch = loss_and_grad(
            params["x0"],
            params["acquisition"],
            params["config"],
            params["medium"],
            params["y_obs"],
            shot_batch_size=2,
        )

        self.assertTrue(
            bool(jnp.allclose(loss_batch, loss_seq, rtol=1.0e-6, atol=1.0e-8))
        )
        self.assertTrue(
            bool(jnp.allclose(grad_batch, grad_seq, rtol=1.0e-5, atol=1.0e-6))
        )

    def test_explicit_adjoint_supports_higher_order_differentiation(self):
        """Meta-gradients should stay available through the explicit adjoint."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        backend = build_backend("jax")

        def squared_grad_norm(model):
            _, grad = backend.loss_grad(params, model, (params["y_obs"],))
            return jnp.sum(grad**2)

        meta_grad = jax.grad(squared_grad_norm)(params["x0"])
        self.assertEqual(meta_grad.shape, params["x0"].shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(meta_grad))))

    def test_medium_physics_change_the_simulated_wavefield(self):
        """Density/attenuation support should materially affect traces."""

        baseline_params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        medium_params = init_params(jax.random.PRNGKey(0), config=_tiny_medium_config())

        baseline_traces = forward(baseline_params, baseline_params["x_exact"])
        medium_traces = forward(medium_params, medium_params["x_exact"])

        self.assertTrue(bool(jnp.all(jnp.isfinite(medium_traces))))
        self.assertFalse(bool(jnp.allclose(baseline_traces, medium_traces)))

    def test_explicit_adjoint_matches_autodiff_with_medium_physics(self):
        """The explicit adjoint should stay correct with fixed medium fields."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_medium_config())
        auxs = (params["y_obs"],)
        backend = build_backend("jax")

        explicit_value, explicit_grad = backend.loss_grad(params, params["x0"], auxs)
        autodiff_value, autodiff_grad = jax.value_and_grad(
            lambda model: loss(params, model, auxs).sum()
        )(params["x0"])

        self.assertTrue(
            bool(
                jnp.allclose(
                    explicit_value.squeeze(),
                    autodiff_value,
                    rtol=1.0e-4,
                    atol=1.0e-6,
                )
            )
        )
        self.assertTrue(
            bool(jnp.allclose(explicit_grad, autodiff_grad, rtol=7.5e-3, atol=7.5e-4))
        )

    def test_stride_source_scale_matches_reference_formula(self):
        """The source scaling should follow the tracked Stride expression."""

        config = _tiny_config()
        velocity_at_source = jnp.array(1500.0)
        expected = (
            2.0
            * (config.time.dt**2)
            * velocity_at_source
            / max(config.grid.dx, config.grid.dy)
            / config.time.dt
        )

        self.assertTrue(
            bool(jnp.isclose(_source_scale(velocity_at_source, config), expected))
        )

    def test_ot2_and_ot4_produce_distinct_finite_wavefields(self):
        """The kernel switch should be active and remain numerically stable."""

        base = _tiny_config()
        ot2_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(base.solver, kernel="OT2"),
        )
        ot4_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(base.solver, kernel="OT4"),
        )

        ot2_params = init_params(jax.random.PRNGKey(0), config=ot2_config)
        ot4_params = init_params(jax.random.PRNGKey(0), config=ot4_config)
        traces_ot2 = forward(ot2_params, ot2_params["x_exact"])
        traces_ot4 = forward(ot4_params, ot4_params["x_exact"])

        self.assertTrue(bool(jnp.all(jnp.isfinite(traces_ot2))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(traces_ot4))))
        self.assertFalse(bool(jnp.allclose(traces_ot2, traces_ot4)))

    def test_second_and_tenth_order_stencils_produce_distinct_wavefields(self):
        """Higher-order spatial derivatives should materially change the traces."""

        base = _tiny_config()
        so2_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(base.solver, space_order=2),
        )
        so10_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(base.solver, space_order=10),
        )

        so2_params = init_params(jax.random.PRNGKey(0), config=so2_config)
        so10_params = init_params(jax.random.PRNGKey(0), config=so10_config)
        traces_so2 = forward(so2_params, so2_params["x_exact"])
        traces_so10 = forward(so10_params, so10_params["x_exact"])

        self.assertTrue(bool(jnp.all(jnp.isfinite(traces_so2))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(traces_so10))))
        self.assertFalse(bool(jnp.allclose(traces_so2, traces_so10)))

    def test_stride_like_gradient_processing_masks_normalises_and_smooths(self):
        """Gradient preprocessing should follow the configured stride-like steps."""

        grad = jnp.zeros((8, 8), dtype=jnp.float32)
        grad = grad.at[4, 4].set(5.0)
        grad = grad.at[1, 1].set(2.0)

        processed = process_global_gradient_stride_like(
            grad,
            damping_cells=1,
            mask_grad=True,
            smooth_grad=True,
            smooth_radius=1,
            norm_grad=True,
        )

        # A boundary-only impulse should be removed by masking.
        boundary_grad = jnp.zeros((8, 8), dtype=jnp.float32).at[0, 0].set(3.0)
        masked_boundary = process_global_gradient_stride_like(
            boundary_grad,
            damping_cells=1,
            mask_grad=True,
            smooth_grad=True,
            smooth_radius=1,
            norm_grad=False,
        )
        unmasked_boundary = process_global_gradient_stride_like(
            boundary_grad,
            damping_cells=1,
            mask_grad=False,
            smooth_grad=True,
            smooth_radius=1,
            norm_grad=False,
        )
        self.assertTrue(bool(jnp.allclose(masked_boundary, 0.0)))
        self.assertGreater(float(jnp.linalg.norm(unmasked_boundary)), 0.0)

        # Normalisation keeps amplitudes in [-1, 1].
        self.assertLessEqual(float(jnp.max(jnp.abs(processed))), 1.0 + 1.0e-6)

        # Smoothing spreads the centre impulse to neighbouring cells.
        self.assertGreater(float(processed[4, 3]), 0.0)
        self.assertGreater(float(processed[3, 4]), 0.0)

    def test_stride_like_gradient_processing_applies_norm_guess_change(self):
        """NormField parity should apply Stride's model-dependent scaling."""

        grad = jnp.zeros((4, 4), dtype=jnp.float32).at[2, 2].set(5.0)
        model = jnp.full((4, 4), 2000.0, dtype=jnp.float32)

        processed = process_global_gradient_stride_like(
            grad,
            damping_cells=0,
            model=model,
            mask_grad=False,
            smooth_grad=False,
            norm_grad=True,
            norm_guess_change=0.5,
        )

        # With max |grad| = 5 and Stride-like var_corr = 2000 * 0.5 / 100 = 10,
        # the peak amplitude should be scaled to 10.
        self.assertTrue(bool(jnp.isclose(jnp.max(jnp.abs(processed)), 10.0)))

    def test_stride_like_gradient_processing_supports_legacy_box_smoothing(self):
        """Non-positive smooth_sigma should trigger box-filter fallback."""

        grad = jnp.zeros((9, 9), dtype=jnp.float32).at[4, 4].set(1.0)
        gaussian = process_global_gradient_stride_like(
            grad,
            damping_cells=0,
            mask_grad=False,
            smooth_grad=True,
            smooth_sigma=0.25,
            smooth_radius=1,
            norm_grad=False,
        )
        box = process_global_gradient_stride_like(
            grad,
            damping_cells=0,
            mask_grad=False,
            smooth_grad=True,
            smooth_sigma=0.0,
            smooth_radius=1,
            norm_grad=False,
        )

        self.assertTrue(bool(jnp.all(jnp.isfinite(gaussian))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(box))))
        # Regression guard: the sigma<=0 branch should be an actual alternate
        # smoothing path, not accidentally identical to the Gaussian one.
        self.assertFalse(bool(jnp.allclose(gaussian, box)))

    def test_stride_like_gradient_mask_uses_soft_rampoff(self):
        """MaskField parity should apply a cosine ramp near domain edges."""

        grad = jnp.ones((9, 9), dtype=jnp.float32)
        processed = process_global_gradient_stride_like(
            grad,
            damping_cells=0,
            mask_grad=True,
            mask_rampoff=4,
            smooth_grad=False,
            norm_grad=False,
        )

        self.assertEqual(processed.shape, grad.shape)
        self.assertTrue(bool(jnp.isclose(processed[0, 0], 0.0)))
        self.assertTrue(bool(jnp.isclose(processed[4, 4], 1.0)))
        self.assertGreater(float(processed[1, 1]), 0.0)
        self.assertLess(float(processed[1, 1]), 1.0)

    def test_stride_like_boundary_mask_damps_edges_more_than_interior(self):
        """Stride-like damping should attenuate edge cells more than the centre."""

        config = _tiny_config()
        velocity = jnp.full((config.grid.nx, config.grid.ny), 1500.0, dtype=jnp.float32)
        mask, sponge_damp = _build_boundary_terms(config, velocity)

        self.assertEqual(mask.shape, velocity.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(mask))))
        self.assertTrue(bool(jnp.all(mask >= 0.0)))
        self.assertTrue(bool(jnp.all(mask <= 1.0)))
        self.assertTrue(bool(jnp.allclose(sponge_damp, 0.0)))

        centre = float(mask[config.grid.nx // 2, config.grid.ny // 2])
        edge = float(mask[1, config.grid.ny // 2])
        self.assertGreaterEqual(centre, edge)

    def test_sponge2_boundary_mode_exposes_positive_damping_field(self):
        """Sponge2 mode should return a non-zero damping field near boundaries."""

        base = _tiny_config()
        config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(base.solver, damping_mode="sponge2", damping_cells=4),
        )
        velocity = jnp.full((config.grid.nx, config.grid.ny), 1500.0, dtype=jnp.float32)
        mask, sponge_damp = _build_boundary_terms(config, velocity)

        # Stride's sponge boundary path applies damping terms directly rather
        # than masking edge rows/columns to zero, so the update mask should be
        # fully active.
        self.assertTrue(bool(jnp.allclose(mask, 1.0)))
        self.assertGreater(float(jnp.max(sponge_damp)), 0.0)
        self.assertTrue(
            bool(
                jnp.allclose(sponge_damp[config.grid.nx // 2, config.grid.ny // 2], 0.0)
            )
        )

    def test_sponge2_default_reflection_matches_stride_width_rule(self):
        """Default reflection coefficient should follow Stride's width heuristic."""

        base = _tiny_config()
        cells = 12
        reflection = 10.0 ** (-(jnp.log10(float(cells)) - 1.0) / jnp.log10(2.0) - 3.0)

        auto_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                damping_mode="sponge2",
                damping_cells=cells,
                damping_reflection_coefficient=None,
            ),
        )
        explicit_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                damping_mode="sponge2",
                damping_cells=cells,
                damping_reflection_coefficient=float(reflection),
            ),
        )
        velocity = jnp.full((base.grid.nx, base.grid.ny), 1500.0, dtype=jnp.float32)

        _, damp_auto = _build_boundary_terms(auto_config, velocity)
        _, damp_explicit = _build_boundary_terms(explicit_config, velocity)
        self.assertTrue(
            bool(jnp.allclose(damp_auto, damp_explicit, rtol=1.0e-6, atol=1.0e-8))
        )

    def test_sponge2_with_zero_absorbing_cells_yields_zero_damping(self):
        """Sponge2 should reduce to zero boundary damping when width is zero."""

        base = _tiny_config()
        config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                damping_mode="sponge2",
                damping_cells=0,
            ),
        )
        velocity = jnp.full((base.grid.nx, base.grid.ny), 1500.0, dtype=jnp.float32)
        mask, sponge_damp = _build_boundary_terms(config, velocity)
        self.assertTrue(bool(jnp.allclose(mask, 1.0)))
        self.assertTrue(bool(jnp.allclose(sponge_damp, 0.0)))

    def test_sponge2_damping_uses_local_velocity_scaling(self):
        """Sponge2 damping should scale pointwise with local velocity."""

        base = _tiny_config()
        scaled_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                damping_mode="sponge2",
                damping_cells=4,
                damping_velocity_scale=True,
            ),
        )
        unscaled_config = BrainFWIConfig(
            grid=base.grid,
            time=base.time,
            acquisition=base.acquisition,
            model=base.model,
            solver=replace(
                base.solver,
                damping_mode="sponge2",
                damping_cells=4,
                damping_velocity_scale=False,
            ),
        )
        velocity = jnp.full((base.grid.nx, base.grid.ny), 1500.0, dtype=jnp.float32)
        velocity = velocity.at[0, base.grid.ny // 2].set(3000.0)

        _, damp_scaled = _build_boundary_terms(scaled_config, velocity)
        _, damp_unscaled = _build_boundary_terms(unscaled_config, velocity)
        edge_i = 0
        edge_j = base.grid.ny // 2

        self.assertGreater(float(damp_unscaled[edge_i, edge_j]), 0.0)
        ratio = float(damp_scaled[edge_i, edge_j] / damp_unscaled[edge_i, edge_j])
        self.assertTrue(bool(jnp.isfinite(ratio)))
        self.assertAlmostEqual(ratio, 3000.0, delta=5.0)

    def test_solver_domain_padding_wraps_physical_model_in_extra_halo(self):
        """The solver should run on a larger padded grid than the inversion model."""

        config = _tiny_config()
        velocity = jnp.ones((config.grid.nx, config.grid.ny), dtype=jnp.float32)
        padded = _pad_model_for_solver(velocity, config)

        self.assertEqual(
            padded.shape,
            (
                config.grid.nx + 2 * config.solver.extra_cells_x,
                config.grid.ny + 2 * config.solver.extra_cells_y,
            ),
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    padded[
                        config.solver.extra_cells_x : config.solver.extra_cells_x
                        + config.grid.nx,
                        config.solver.extra_cells_y : config.solver.extra_cells_y
                        + config.grid.ny,
                    ],
                    velocity,
                )
            )
        )

    def test_shot_progress_formatter_compacts_long_batches(self):
        """Shot progress logging should keep long source lists readable."""

        compact = format_shot_ids_for_log(jnp.array([1, 2, 3], dtype=jnp.int32))
        self.assertEqual(compact, "[1, 2, 3]")

        long_preview = format_shot_ids_for_log(
            jnp.arange(16, dtype=jnp.int32),
            max_items=6,
        )
        self.assertIn("(total=16)", long_preview)
        self.assertIn("...", long_preview)

    def test_stride_like_random_shot_schedule_avoids_repeats_before_wrap(self):
        """Random shot scheduling should consume one permutation before repeats."""

        schedule = _build_random_shot_schedule(
            available_shots=jnp.arange(8, dtype=jnp.int32),
            stage_steps=(1, 2),
            shots_per_iter=3,
            seed=7,
        )

        first_cycle = jnp.concatenate((schedule[0][0], schedule[1][0], schedule[1][1]))
        self.assertEqual(first_cycle.shape[0], 8)
        self.assertEqual(int(jnp.unique(first_cycle).shape[0]), 8)

    def test_stride_like_random_shot_schedule_allows_short_boundary_batches(self):
        """Stride-like queue consumption should allow short batches at wrap boundaries."""

        schedule = _build_random_shot_schedule(
            available_shots=jnp.arange(8, dtype=jnp.int32),
            stage_steps=(4,),
            shots_per_iter=3,
            seed=123,
        )
        batch_lengths = [int(batch.shape[0]) for batch in schedule[0]]

        # With 8 shots and batch size 3, Stride's queue logic yields lengths:
        # 3, 3, 2, then a new cycle starts.
        self.assertEqual(batch_lengths[:3], [3, 3, 2])

    def test_run_complete_marker_writes_expected_artifact(self):
        """Completion marker should be created with artifact references."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            marker = write_run_complete_marker(
                output_dir,
                "sgd",
                steps=24,
                max_freqs_hz=(100000.0, 200000.0, 300000.0),
                metrics_path=output_dir / "sgd_metrics.json",
                history_path=output_dir / "sgd_history.json",
                reconstruction_path=output_dir / "sgd_reconstruction.png",
                history_plot_path=output_dir / "sgd_history.png",
            )

            self.assertTrue(marker.exists())
            payload = marker.read_text(encoding="utf-8")
            self.assertIn('"status": "completed"', payload)
            self.assertIn('"optimizer": "sgd"', payload)
            self.assertIn('"steps": 24', payload)

    def test_run_state_marker_writes_running_payload(self):
        """Running markers should make in-progress runs easy to identify."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            marker = write_run_state_marker(
                output_dir,
                "sgd",
                state="RUNNING",
                steps=24,
                max_freqs_hz=(100000.0, 200000.0, 300000.0),
                message="Run started.",
                artifacts={"metrics_json": str(output_dir / "sgd_metrics.json")},
            )

            self.assertTrue(marker.exists())
            payload = marker.read_text(encoding="utf-8")
            self.assertIn('"state": "RUNNING"', payload)
            self.assertIn('"status": "running"', payload)
            self.assertIn('"message": "Run started."', payload)

    def test_clear_run_outputs_removes_only_matching_optimizer_artifacts(self):
        """Output cleanup should not delete artifacts from other optimisers."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            matching = output_dir / "sgd_metrics.json"
            matching.write_text("{}", encoding="utf-8")
            matching_diag = output_dir / "sgd_stage01_first_step001_diagnostics.json"
            matching_diag.write_text("{}", encoding="utf-8")
            matching_running = output_dir / "sgd_RUNNING.json"
            matching_running.write_text("{}", encoding="utf-8")
            other = output_dir / "adam_metrics.json"
            other.write_text("{}", encoding="utf-8")

            removed = clear_run_outputs(output_dir, "sgd")

            self.assertFalse(matching.exists())
            self.assertFalse(matching_diag.exists())
            self.assertFalse(matching_running.exists())
            self.assertTrue(other.exists())
            self.assertEqual(
                {path.name for path in removed},
                {matching.name, matching_diag.name, matching_running.name},
            )

    def test_final_metric_shot_selection_is_deterministic_and_bounded(self):
        """Final-metric shot subset selection should be stable and valid."""

        all_shots = jnp.arange(12, dtype=jnp.int32)
        subset_a = select_final_metric_shot_positions(all_shots, final_shots=5, seed=42)
        subset_b = select_final_metric_shot_positions(all_shots, final_shots=5, seed=42)
        self.assertTrue(bool(jnp.array_equal(subset_a, subset_b)))
        self.assertEqual(subset_a.shape[0], 5)
        self.assertTrue(bool(jnp.all(subset_a >= 0)))
        self.assertTrue(bool(jnp.all(subset_a < 12)))

        full = select_final_metric_shot_positions(all_shots, final_shots=None, seed=42)
        self.assertTrue(bool(jnp.array_equal(full, jnp.arange(12, dtype=jnp.int32))))

    def test_trace_smoothing_preserves_shape(self):
        """Continuation smoothing should not change the survey tensor shape."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        y_obs = params["y_obs"]
        y_smooth = smooth_traces(y_obs, radius=3)

        self.assertEqual(y_smooth.shape, y_obs.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(y_smooth))))

    def test_problem_builder_returns_shared_problem_object(self):
        """The explicit problem builder should expose the shared API object."""

        problem = build_brain_fwi_problem(jax.random.PRNGKey(0), config=_tiny_config())

        self.assertEqual(problem.backend_name, "jax")
        self.assertEqual(problem.acquisition.n_shots, 3)
        self.assertEqual(problem.y_obs.shape, (3, 40, 12))


if __name__ == "__main__":
    unittest.main()
