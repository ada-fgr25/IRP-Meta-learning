"""Problem definition for a differentiable brain FWI baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .acoustics import build_geometry
from .backends import build_backend
from .config import BrainFWIConfig
from .phantoms import build_initial_velocity, build_true_brain_velocity


def init_params(key, config: BrainFWIConfig | None = None, backend_name: str = "jax"):
    """Initialise a brain imaging FWI problem and generate observed data."""

    del key
    config = config or BrainFWIConfig()
    geometry = build_geometry(config)
    backend = build_backend(backend_name)
    x_exact = build_true_brain_velocity(config)
    x0 = build_initial_velocity(config)
    y_obs = backend.forward(x_exact, geometry, config)

    return {
        "config": config,
        "geometry": geometry,
        "backend_name": backend_name,
        "x0": x0,
        "x_exact": x_exact,
        "y_obs": y_obs,
        "auxs_shapes": (y_obs.shape,),
    }


def build_brain_fwi_problem(
    key, config: BrainFWIConfig | None = None, backend_name: str = "jax"
):
    """Compatibility wrapper that mirrors the style used in `descend`."""

    return init_params(key, config=config, backend_name=backend_name)


def forward(params, x):
    """Run the configured forward model for a candidate velocity field."""

    backend = build_backend(params["backend_name"])
    return backend.forward(x, params["geometry"], params["config"])


def loss(params, x, auxs):
    """Data-fidelity term used for classical FWI optimisation."""

    y_obs = auxs[0]
    y = forward(params, x)
    return jnp.mean((y - y_obs) ** 2).reshape((1,))


def dldx(params, x, auxs):
    """Return the loss value and gradient with respect to the velocity field."""

    value, grad = jax.value_and_grad(lambda model: loss(params, model, auxs).sum())(x)
    return value.reshape((1,)), grad


def sample(params, key):
    """Return a deterministic sample to stay close to the `descend` API."""

    del key
    return params["x0"], (params["y_obs"],), params["x_exact"]


def sample_batch(params, key, batch_size):
    """Replicate the same synthetic problem to provide a batch-shaped interface."""

    x0, auxs, x_exact = sample(params, key)
    return (
        jnp.repeat(x0[None, ...], batch_size, axis=0),
        (jnp.repeat(auxs[0][None, ...], batch_size, axis=0),),
        jnp.repeat(x_exact[None, ...], batch_size, axis=0),
    )
