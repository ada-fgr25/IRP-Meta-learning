"""Classical optimisation runners for the Phase 1 baseline."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import optimize


Array = jnp.ndarray
LossFn = Callable[[Array], Array]


def _project_velocity(model: Array, bounds: tuple[float, float]) -> Array:
    """Keep velocity estimates within physically plausible limits."""

    return jnp.clip(model, bounds[0], bounds[1])


def _step_metrics(
    step: int,
    model: Array,
    loss_value: Array,
    true_model: Array | None = None,
) -> dict[str, float]:
    """Build a compact history record for logging and plotting."""

    scalar_loss = float(jnp.asarray(loss_value).reshape(()))
    metrics = {
        "step": float(step),
        "loss": scalar_loss,
        "velocity_mean": float(jnp.mean(model)),
    }
    if true_model is not None:
        metrics["model_rmse"] = float(jnp.sqrt(jnp.mean((model - true_model) ** 2)))
    return metrics


def run_optax(
    model_init: Array,
    loss_grad_fn: Callable[[Array], tuple[Array, Array]],
    optimiser: optax.GradientTransformation,
    n_steps: int,
    bounds: tuple[float, float],
    true_model: Array | None = None,
):
    """Run a standard Optax optimiser on the FWI objective."""

    model = model_init
    state = optimiser.init(model)
    history = []

    for step in range(n_steps):
        loss_value, grad = loss_grad_fn(model)
        updates, state = optimiser.update(grad, state, model)
        model = optax.apply_updates(model, updates)
        model = _project_velocity(model, bounds)
        history.append(_step_metrics(step, model, loss_value, true_model))

    final_loss, _ = loss_grad_fn(model)
    return model, history, float(jnp.asarray(final_loss).reshape(()))


def run_sgd(
    model_init: Array,
    loss_grad_fn: Callable[[Array], tuple[Array, Array]],
    learning_rate: float,
    n_steps: int,
    bounds: tuple[float, float],
    true_model: Array | None = None,
):
    """Run SGD on the FWI objective."""

    optimiser = optax.sgd(learning_rate=learning_rate)
    return run_optax(model_init, loss_grad_fn, optimiser, n_steps, bounds, true_model)


def run_adam(
    model_init: Array,
    loss_grad_fn: Callable[[Array], tuple[Array, Array]],
    learning_rate: float,
    n_steps: int,
    bounds: tuple[float, float],
    true_model: Array | None = None,
):
    """Run Adam on the FWI objective."""

    optimiser = optax.adam(learning_rate=learning_rate)
    return run_optax(model_init, loss_grad_fn, optimiser, n_steps, bounds, true_model)


def run_lbfgsb(
    model_init: Array,
    loss_grad_fn: Callable[[Array], tuple[Array, Array]],
    maxiter: int,
    bounds: tuple[float, float],
    true_model: Array | None = None,
):
    """Run SciPy L-BFGS-B using JAX-generated gradients."""

    shape = model_init.shape
    history = []

    def objective(flat_model: np.ndarray):
        model = jnp.asarray(flat_model.reshape(shape))
        loss_value, grad = loss_grad_fn(model)
        scalar_loss = float(jnp.asarray(loss_value).reshape(()))
        return scalar_loss, np.asarray(grad, dtype=np.float64).ravel()

    def callback(flat_model: np.ndarray):
        model = jnp.asarray(flat_model.reshape(shape))
        loss_value, _ = loss_grad_fn(model)
        history.append(_step_metrics(len(history), model, loss_value, true_model))

    scipy_bounds = [bounds] * int(np.prod(shape))
    result = optimize.minimize(
        fun=objective,
        x0=np.asarray(model_init, dtype=np.float64).ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=scipy_bounds,
        callback=callback,
        options={"maxiter": maxiter},
    )

    model = jnp.asarray(result.x.reshape(shape))
    final_loss, _ = loss_grad_fn(model)
    if not history:
        history.append(_step_metrics(0, model, final_loss, true_model))
    return model, history, float(jnp.asarray(final_loss).reshape(()))
