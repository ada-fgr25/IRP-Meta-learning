"""Smoke tests for the Phase 1 JAX FWI baseline."""

from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp

from fwi.acoustics import _build_boundary_terms, _source_scale
from fwi.backends import build_backend
from fwi.config import (
    AcquisitionConfig,
    BrainFWIConfig,
    GridConfig,
    ModelConfig,
    TimeConfig,
)
from fwi.optimisers import process_global_gradient_stride_like
from fwi.problem import (
    build_brain_fwi_problem,
    dldx,
    forward,
    init_params,
    loss,
    smooth_traces,
)


def _tiny_config() -> BrainFWIConfig:
    """Return a small configuration that keeps tests lightweight."""

    return BrainFWIConfig(
        grid=GridConfig(nx=32, ny=24),
        time=TimeConfig(nt=40),
        acquisition=AcquisitionConfig(n_transducers=12, n_shots=3),
        model=ModelConfig(source="procedural"),
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

        self.assertTrue(bool(jnp.all(mask >= 0.0)))
        self.assertTrue(bool(jnp.all(mask <= 1.0)))
        self.assertGreater(float(jnp.max(sponge_damp)), 0.0)
        self.assertTrue(
            bool(
                jnp.allclose(sponge_damp[config.grid.nx // 2, config.grid.ny // 2], 0.0)
            )
        )

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
