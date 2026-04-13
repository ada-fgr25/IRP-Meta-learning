"""Smoke tests for the Phase 1 JAX FWI baseline."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from fwi.backends import build_backend
from fwi.config import (
    AcquisitionConfig,
    BrainFWIConfig,
    GridConfig,
    ModelConfig,
    TimeConfig,
)
from fwi.problem import (
    build_brain_fwi_problem,
    dldx,
    forward,
    init_params,
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

    def test_gradient_is_finite(self):
        """The differentiable solver should provide a finite adjoint signal."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        loss_value, grad = dldx(params, params["x0"], (params["y_obs"],))

        self.assertEqual(loss_value.shape, (1,))
        self.assertEqual(grad.shape, params["x0"].shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_explicit_adjoint_matches_autodiff_gradient(self):
        """The explicit adjoint should agree with a direct autodiff reference."""

        params = init_params(jax.random.PRNGKey(0), config=_tiny_config())
        auxs = (params["y_obs"],)
        backend = build_backend("jax")

        explicit_value, explicit_grad = backend.loss_grad(params, params["x0"], auxs)
        autodiff_value, autodiff_grad = jax.value_and_grad(
            lambda model: jnp.sum((forward(params, model) - auxs[0]) ** 2)
            / auxs[0].size
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
