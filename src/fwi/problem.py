"""Problem definition for a differentiable brain FWI baseline.

This module gives the FWI setup a compact API close to the style used by the
`descend` reference code: initialise parameters, run the forward model, compute
loss, and obtain gradients with respect to the model.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from .backends import build_backend
from .config import BrainFWIConfig
from .phantoms import build_initial_velocity, build_true_brain_velocity


def _load_stride_field(
    path: str, downsample: int
) -> tuple[jnp.ndarray, tuple[float, float]]:
    """Load and optionally downsample a Stride HDF5 scalar field.

    The Stride example stores both the grid spacing and the model samples in the
    file. We preserve that metadata here so the JAX solver sees a geometry that
    matches the loaded model rather than the old hard-coded toy grid. The
    default experiment path now uses the native Stride resolution, and this
    downsampling hook is only retained as an escape hatch for cheaper debugging.
    """

    # Import lazily so procedural-only workflows do not need the HDF5 runtime
    # just to import the generic problem module.
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle["data"][()]
        spacing = tuple(float(v) for v in handle["space/spacing"][()])

    stride = max(int(downsample), 1)
    data = data[::stride, ::stride]
    spacing = (spacing[0] * stride, spacing[1] * stride)
    return jnp.asarray(data), spacing


def _initialise_models(
    config: BrainFWIConfig,
) -> tuple[BrainFWIConfig, jnp.ndarray, jnp.ndarray]:
    """Build the true and starting models for the selected data source."""

    if config.model.source == "procedural":
        x_exact = build_true_brain_velocity(config)
        x0 = build_initial_velocity(config)
        return config, x_exact, x0

    if config.model.source != "stride":
        raise ValueError(f"Unknown model source '{config.model.source}'.")

    x_exact, spacing = _load_stride_field(
        config.model.true_model_path, config.model.stride_downsample
    )
    x0, _ = _load_stride_field(
        config.model.starting_model_path, config.model.stride_downsample
    )

    grid = replace(
        config.grid,
        nx=int(x_exact.shape[0]),
        ny=int(x_exact.shape[1]),
        dx=float(spacing[0]),
        dy=float(spacing[1]),
    )
    config = replace(config, grid=grid)
    return config, x_exact, x0


def init_params(key, config: BrainFWIConfig | None = None, backend_name: str = "jax"):
    """Initialise a brain imaging FWI problem and generate observed data.

    We generate:
    - `x_exact`: the synthetic ground-truth velocity model
    - `x0`: the deliberately smoother starting model
    - `y_obs`: the observed data cube produced by the true model

    In a later data-driven stage, this is the natural place to swap in measured
    data or phantoms loaded from disk instead of procedural ones.
    """

    del key
    config = config or BrainFWIConfig()
    config, x_exact, x0 = _initialise_models(config)
    backend = build_backend(backend_name)
    acquisition = backend.build_acquisition(config)
    y_obs = backend.forward(x_exact, acquisition, config)

    return {
        "config": config,
        "acquisition": acquisition,
        "geometry": acquisition,
        "backend_name": backend_name,
        "x0": x0,
        "x_exact": x_exact,
        "y_obs": y_obs,
        "auxs_shapes": (y_obs.shape,),
    }


def build_brain_fwi_problem(
    key, config: BrainFWIConfig | None = None, backend_name: str = "jax"
):
    """Compatibility wrapper that mirrors the style used in `descend`.

    Keeping this alias means future meta-learning experiments can call the FWI
    problem through a small, stable interface rather than depending directly on
    all of the underlying helper modules.
    """

    return init_params(key, config=config, backend_name=backend_name)


def forward(params, x):
    """Run the configured forward model for a candidate velocity field.

    `x` is a velocity model candidate and the result is a full simulated data
    cube with shape `[shot, time, receiver]`.
    """

    backend = build_backend(params["backend_name"])
    return backend.forward(x, params["acquisition"], params["config"])


def loss(params, x, auxs):
    """Data-fidelity term used for classical FWI optimisation.

    For Phase 1 we use a plain mean-squared error between predicted and observed
    traces. That keeps the baseline simple before we explore more robust losses
    or multi-scale strategies.
    """

    y_obs = auxs[0]
    y = forward(params, x)
    return jnp.mean((y - y_obs) ** 2).reshape((1,))


def smooth_traces(traces: jnp.ndarray, radius: int) -> jnp.ndarray:
    """Apply a simple low-pass smoothing along the time axis of each trace.

    This is a lightweight stand-in for frequency continuation: large smoothing
    radii suppress high-frequency waveform detail early in optimisation, and the
    radius can then be reduced over stages until we recover the unsmoothed loss.
    """

    radius = int(radius)
    if radius <= 0:
        return traces

    kernel = jnp.ones((2 * radius + 1,), dtype=traces.dtype) / (2 * radius + 1)

    def smooth_trace(trace: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.pad(trace, (radius, radius), mode="edge")
        return jnp.convolve(padded, kernel, mode="valid")

    traces = jnp.swapaxes(traces, 1, 2)
    traces = jax.vmap(jax.vmap(smooth_trace))(traces)
    return jnp.swapaxes(traces, 1, 2)


def dldx(params, x, auxs):
    """Return the loss value and gradient with respect to the velocity field.

    The "adjoint" in this baseline is obtained implicitly through
    `jax.value_and_grad`, which differentiates through the full forward solve.
    """

    value, grad = jax.value_and_grad(lambda model: loss(params, model, auxs).sum())(x)
    return value.reshape((1,)), grad


def sample(params, key):
    """Return a deterministic sample to stay close to the `descend` API.

    The current baseline uses a single fixed synthetic problem instance. The key
    is therefore unused for now, but keeping it in the signature makes later
    stochastic sampling or dataset-based generation easier to add.
    """

    del key
    return params["x0"], (params["y_obs"],), params["x_exact"]


def sample_batch(params, key, batch_size):
    """Replicate the same synthetic problem to provide a batch-shaped interface.

    This is mainly an integration convenience for future meta-learning code,
    where batched problems are more natural than single examples.
    """

    x0, auxs, x_exact = sample(params, key)
    return (
        jnp.repeat(x0[None, ...], batch_size, axis=0),
        (jnp.repeat(auxs[0][None, ...], batch_size, axis=0),),
        jnp.repeat(x_exact[None, ...], batch_size, axis=0),
    )
