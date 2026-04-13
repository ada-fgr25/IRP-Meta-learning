"""Smoke tests for the Phase 1 JAX FWI baseline."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from fwi.config import (
    AcquisitionConfig,
    BrainFWIConfig,
    GridConfig,
    ModelConfig,
    TimeConfig,
)
from fwi.problem import dldx, init_params, smooth_traces


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

    def test_gradient_is_finite(self):
        """The differentiable solver should provide a finite adjoint signal."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        loss_value, grad = dldx(params, params["x0"], (params["y_obs"],))

        self.assertEqual(loss_value.shape, (1,))
        self.assertEqual(grad.shape, params["x0"].shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_trace_smoothing_preserves_shape(self):
        """Continuation smoothing should not change the survey tensor shape."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        y_obs = params["y_obs"]
        y_smooth = smooth_traces(y_obs, radius=3)

        self.assertEqual(y_smooth.shape, y_obs.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(y_smooth))))


if __name__ == "__main__":
    unittest.main()
