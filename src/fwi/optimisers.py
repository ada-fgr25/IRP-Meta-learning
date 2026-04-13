"""Classical optimisation runners for the Phase 1 baseline."""

from __future__ import annotations

from typing import Callable

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
    snapshot_steps: tuple[int, ...] = (0, 10, 20),
):
    """Run a standard Optax optimiser on the FWI objective."""

    model = model_init
    state = optimiser.init(model)
    history = []
    snapshots = {0: jnp.array(model_init)}

    for step in range(n_steps):
        loss_value, grad = loss_grad_fn(model)
        updates, state = optimiser.update(grad, state, model)
        model = optax.apply_updates(model, updates)
        model = _project_velocity(model, bounds)
        history.append(_step_metrics(step, model, loss_value, true_model))
        iteration = step + 1
        if iteration in snapshot_steps:
            snapshots[iteration] = jnp.array(model)

    final_loss, _ = loss_grad_fn(model)
    return model, history, float(jnp.asarray(final_loss).reshape(())), snapshots


def run_sgd(
    model_init: Array,
    loss_grad_fn: Callable[[Array], tuple[Array, Array]],
    learning_rate: float,
    n_steps: int,
    bounds: tuple[float, float],
    true_model: Array | None = None,
    snapshot_steps: tuple[int, ...] = (0, 10, 20),
):
    """Run SGD on the FWI objective."""

    optimiser = optax.sgd(learning_rate=learning_rate)
    return run_optax(
        model_init,
        loss_grad_fn,
        optimiser,
        n_steps,
        bounds,
        true_model,
        snapshot_steps,
    )


def run_adam(
    model_init: Array,
    loss_grad_fn: Callable[[Array], tuple[Array, Array]],
    learning_rate: float,
    n_steps: int,
    bounds: tuple[float, float],
    true_model: Array | None = None,
    snapshot_steps: tuple[int, ...] = (0, 10, 20),
):
    """Run Adam on the FWI objective."""

    optimiser = optax.adam(learning_rate=learning_rate)
    return run_optax(
        model_init,
        loss_grad_fn,
        optimiser,
        n_steps,
        bounds,
        true_model,
        snapshot_steps,
    )


def run_lbfgsb(
    model_init: Array,
    loss_grad_fn: Callable[[Array], tuple[Array, Array]],
    maxiter: int,
    bounds: tuple[float, float],
    true_model: Array | None = None,
    snapshot_steps: tuple[int, ...] = (0, 10, 20),
):
    """Run SciPy L-BFGS-B using JAX-generated gradients."""

    shape = model_init.shape
    history = []
    snapshots = {0: jnp.array(model_init)}

    def objective(flat_model: np.ndarray):
        model = jnp.asarray(flat_model.reshape(shape))
        loss_value, grad = loss_grad_fn(model)
        scalar_loss = float(jnp.asarray(loss_value).reshape(()))
        return scalar_loss, np.asarray(grad, dtype=np.float64).ravel()

    def callback(flat_model: np.ndarray):
        model = jnp.asarray(flat_model.reshape(shape))
        loss_value, _ = loss_grad_fn(model)
        history.append(_step_metrics(len(history), model, loss_value, true_model))
        iteration = len(history)
        if iteration in snapshot_steps:
            snapshots[iteration] = jnp.array(model)

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
    if maxiter in snapshot_steps:
        snapshots[maxiter] = jnp.array(model)
    return model, history, float(jnp.asarray(final_loss).reshape(())), snapshots


def run_stagewise_optax(
    model_init: Array,
    make_loss_grad_fn: Callable[[int], Callable[[Array], tuple[Array, Array]]],
    optimiser_factory: Callable[[], optax.GradientTransformation],
    stage_steps: tuple[int, ...],
    bounds: tuple[float, float],
    true_model: Array | None = None,
    snapshot_steps: tuple[int, ...] = (0, 10, 20),
    make_step_loss_grad_fn: Callable[
        [int, int], Callable[[Array], tuple[Array, Array]]
    ]
    | None = None,
):
    """Run an Optax optimiser across multiple continuation stages.

    Each stage gets its own loss/gradient function, which lets the experiment
    progressively relax a coarse objective into the full waveform objective.
    """

    model = model_init
    history = []
    snapshots = {0: jnp.array(model_init)}
    global_step = 0

    for stage_index, n_steps in enumerate(stage_steps):
        if n_steps <= 0:
            continue

        loss_grad_fn = make_loss_grad_fn(stage_index)
        optimiser = optimiser_factory()
        state = optimiser.init(model)

        for step_in_stage in range(n_steps):
            # The default path reuses one loss for the whole stage. When a
            # caller provides `make_step_loss_grad_fn`, each iteration can swap
            # in a different stochastic shot subset while still sharing the
            # same optimiser loop and history bookkeeping.
            active_loss_grad_fn = (
                loss_grad_fn
                if make_step_loss_grad_fn is None
                else make_step_loss_grad_fn(stage_index, step_in_stage)
            )
            loss_value, grad = active_loss_grad_fn(model)
            updates, state = optimiser.update(grad, state, model)
            model = optax.apply_updates(model, updates)
            model = _project_velocity(model, bounds)

            metrics = _step_metrics(global_step, model, loss_value, true_model)
            metrics["stage"] = float(stage_index)
            history.append(metrics)

            global_step += 1
            if global_step in snapshot_steps:
                snapshots[global_step] = jnp.array(model)

    final_loss, _ = make_loss_grad_fn(max(len(stage_steps) - 1, 0))(model)
    return model, history, float(jnp.asarray(final_loss).reshape(())), snapshots
