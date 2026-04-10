"""Smoke tests for the Phase 1 JAX FWI baseline."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from irp_meta_learning.fwi.config import (
    AcquisitionConfig,
    BrainFWIConfig,
    GridConfig,
    TimeConfig,
)
from irp_meta_learning.fwi.problem import dldx, init_params


def _tiny_config() -> BrainFWIConfig:
    """Return a small configuration that keeps tests lightweight."""

    return BrainFWIConfig(
        grid=GridConfig(nx=32, ny=24),
        time=TimeConfig(nt=40),
        acquisition=AcquisitionConfig(n_transducers=12, n_shots=3),
    )


class Phase1BrainFWITests(unittest.TestCase):
    """Minimal regression tests for the differentiable FWI baseline."""

    def test_forward_problem_shapes(self):
        """Observed data should have the expected [shot, time, receiver] shape."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        y_obs = params["y_obs"]

        self.assertEqual(y_obs.shape, (3, 40, 12))

    def test_gradient_is_finite(self):
        """The differentiable solver should provide a finite adjoint signal."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        loss_value, grad = dldx(params, params["x0"], (params["y_obs"],))

        self.assertEqual(loss_value.shape, (1,))
        self.assertEqual(grad.shape, params["x0"].shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))


if __name__ == "__main__":
    unittest.main()
