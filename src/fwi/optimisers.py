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


def _edge_padded_rolling_mean_1d(values: Array, radius: int) -> Array:
    """Return a 1D rolling mean with edge padding.

    A compact box smoother is a practical stand-in for Stride's default
    `smooth_field` preprocessing step. We apply it separably in 2D later.
    """

    radius = max(int(radius), 0)
    if radius == 0:
        return values

    window = 2 * radius + 1
    kernel = jnp.ones((window,), dtype=values.dtype) / window
    padded = jnp.pad(values, (radius, radius), mode="edge")
    return jnp.convolve(padded, kernel, mode="valid")


def _box_smooth_2d(field: Array, radius: int) -> Array:
    """Apply a separable box filter to a 2D field."""

    row_smoothed = jax.vmap(lambda row: _edge_padded_rolling_mean_1d(row, radius))(
        field
    )
    col_smoothed_t = jax.vmap(lambda col: _edge_padded_rolling_mean_1d(col, radius))(
        row_smoothed.T
    )
    return col_smoothed_t.T


def process_global_gradient_stride_like(
    grad: Array,
    *,
    damping_cells: int,
    mask_grad: bool = True,
    smooth_grad: bool = True,
    smooth_radius: int = 2,
    norm_grad: bool = True,
) -> Array:
    """Approximate Stride's default `ProcessGlobalGradient` pipeline.

    Stride's default stack is:
    - `mask_field`
    - `smooth_field`
    - `norm_field`

    We reproduce the same high-level behaviour in pure JAX:
    - mask: zero gradient in the outer absorbing frame
    - smooth: apply a lightweight separable box filter
    - norm: scale by max absolute amplitude to stabilise step magnitudes
    """

    processed = grad

    if mask_grad:
        cells = max(int(damping_cells), 0)
        if cells > 0:
            mask = jnp.ones_like(processed)
            mask = mask.at[:cells, :].set(0.0)
            mask = mask.at[-cells:, :].set(0.0)
            mask = mask.at[:, :cells].set(0.0)
            mask = mask.at[:, -cells:].set(0.0)
            processed = processed * mask

    if smooth_grad:
        processed = _box_smooth_2d(processed, radius=smooth_radius)

    if norm_grad:
        max_abs = jnp.max(jnp.abs(processed))
        processed = processed / jnp.maximum(max_abs, 1.0e-12)

    return processed


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
    make_step_loss_grad_fn: (
        Callable[[int, int], Callable[[Array], tuple[Array, Array]]] | None
    ) = None,
    process_grad_fn: Callable[[Array, Array, int, int], Array] | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    step_callback: Callable[[dict[str, object]], None] | None = None,
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

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "stage_start",
                    "stage": float(stage_index),
                    "n_steps": float(n_steps),
                    "global_step": float(global_step),
                }
            )

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
            model_before = model
            loss_value, grad = active_loss_grad_fn(model)
            processed_grad = (
                grad
                if process_grad_fn is None
                else process_grad_fn(model, grad, stage_index, step_in_stage)
            )
            updates, state = optimiser.update(processed_grad, state, model)
            model = optax.apply_updates(model, updates)
            model = _project_velocity(model, bounds)

            metrics = _step_metrics(global_step, model, loss_value, true_model)
            metrics["stage"] = float(stage_index)
            metrics["step_in_stage"] = float(step_in_stage)
            metrics["n_steps_in_stage"] = float(n_steps)
            history.append(metrics)

            if progress_callback is not None:
                progress_callback(dict(metrics))
            if step_callback is not None:
                step_callback(
                    {
                        "stage_index": stage_index,
                        "step_in_stage": step_in_stage,
                        "n_steps_in_stage": n_steps,
                        "global_step": global_step,
                        "model_before": model_before,
                        "model_after": model,
                        "gradient": grad,
                        "processed_gradient": processed_grad,
                        "loss": loss_value,
                    }
                )

            global_step += 1
            if global_step in snapshot_steps:
                snapshots[global_step] = jnp.array(model)

    final_loss, _ = make_loss_grad_fn(max(len(stage_steps) - 1, 0))(model)
    return model, history, float(jnp.asarray(final_loss).reshape(())), snapshots
