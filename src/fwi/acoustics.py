"""Differentiable 2D acoustic wave propagation in JAX.

This module focuses on the time-domain solver itself. Acquisition construction
now lives in :mod:`fwi.acquisition` so both the JAX and Stride workflows can
share one experiment-facing acquisition API.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .acquisition import AcquisitionGeometry
from .config import BrainFWIConfig


def _laplacian(field: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Second-order finite-difference Laplacian with fixed zero edges.

    We only update the interior cells explicitly and then pad the result back to
    the full grid shape. The boundary cells themselves are clamped later.
    """

    centre = field[1:-1, 1:-1]
    lap = (field[2:, 1:-1] - 2.0 * centre + field[:-2, 1:-1]) / dx**2 + (
        field[1:-1, 2:] - 2.0 * centre + field[1:-1, :-2]
    ) / dy**2
    return jnp.pad(lap, ((1, 1), (1, 1)))


def _build_damping_mask(config: BrainFWIConfig) -> jnp.ndarray:
    """Create a simple absorbing frame to reduce edge reflections.

    This is a pragmatic substitute for a full absorbing boundary model. The mask
    smoothly damps wave amplitudes near the edges where artificial reflections
    would otherwise contaminate the inversion objective.
    """

    nx = config.grid.nx
    ny = config.grid.ny
    cells = config.solver.damping_cells
    strength = config.solver.damping_strength

    ix = jnp.minimum(jnp.arange(nx), jnp.arange(nx)[::-1])
    iy = jnp.minimum(jnp.arange(ny), jnp.arange(ny)[::-1])
    dist = jnp.minimum(ix[:, None], iy[None, :]).astype(jnp.float32)
    taper = jnp.clip((cells - dist) / jnp.maximum(cells, 1), 0.0, 1.0)
    return 1.0 - strength * taper**2


def _build_boundary_mask(config: BrainFWIConfig) -> jnp.ndarray:
    """Combine damping and fixed-edge clamping into one linear mask.

    Writing the boundary treatment this way keeps the forward and adjoint
    implementations aligned: both simply apply the same self-adjoint diagonal
    mask rather than having to mirror a sequence of in-place edge updates.
    """

    damping = _build_damping_mask(config)
    interior = jnp.ones_like(damping)
    interior = interior.at[0, :].set(0.0)
    interior = interior.at[-1, :].set(0.0)
    interior = interior.at[:, 0].set(0.0)
    interior = interior.at[:, -1].set(0.0)
    return damping * interior


def _inject_source(
    source_index: jnp.ndarray,
    source_value: jnp.ndarray,
    shape: tuple[int, int],
) -> jnp.ndarray:
    """Place one source sample onto the full simulation grid."""

    return jnp.zeros(shape).at[source_index[0], source_index[1]].set(source_value)


def _inject_receivers(
    receiver_i: jnp.ndarray,
    receiver_j: jnp.ndarray,
    receiver_values: jnp.ndarray,
    shape: tuple[int, int],
) -> jnp.ndarray:
    """Scatter receiver-domain cotangents back onto the grid.

    The injection uses `add` rather than `set` so the code remains correct even
    if a future geometry reuses the same grid cell for multiple receivers.
    """

    return jnp.zeros(shape).at[receiver_i, receiver_j].add(receiver_values)


def _simulate_shot_with_history(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    shot_index: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one shot while storing the forward states needed by the adjoint.

    We record the current wavefield and its Laplacian at each time step. The
    explicit adjoint later reuses those arrays when it accumulates the velocity
    gradient and propagates wavefield sensitivities backwards in time.
    """

    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    receivers, _, wavelet = acquisition.require_solver_arrays()
    source = receivers[shot_index]
    receiver_i = receivers[:, 0]
    receiver_j = receivers[:, 1]
    boundary_mask = _build_boundary_mask(config)
    grid_shape = velocity.shape

    def step(carry, source_value):
        """Advance one time step and emit all adjoint-side history tensors."""

        u_prev, u_curr = carry
        lap_curr = _laplacian(u_curr, dx, dy)
        source_term = _inject_source(source, source_value, grid_shape)
        u_next = boundary_mask * (
            2.0 * u_curr
            - u_prev
            + (dt**2) * ((velocity**2) * lap_curr + source_term)
        )
        traces = u_next[receiver_i, receiver_j]
        return (u_curr, u_next), (traces, u_curr, lap_curr)

    init = (jnp.zeros_like(velocity), jnp.zeros_like(velocity))
    _, (traces, curr_fields, laplacians) = jax.lax.scan(step, init, wavelet)
    return traces, curr_fields, laplacians


def simulate_shot(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    shot_index: jnp.ndarray,
) -> jnp.ndarray:
    """Simulate one transmit event and record traces at every transducer.

    The returned tensor has shape `[time, receiver]`. Internally we carry the
    previous and current wavefields so the second-order time update can generate
    the next wavefield.
    """

    traces, _, _ = _simulate_shot_with_history(velocity, acquisition, config, shot_index)
    return traces


def simulate_survey(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Simulate all configured shots in the acquisition.

    The returned tensor has shape `[shot, time, receiver]`, which is the data
    cube used by the FWI loss.
    """

    return jax.vmap(
        lambda shot_idx: simulate_shot(velocity, acquisition, config, shot_idx)
    )(acquisition.require_solver_arrays()[1])
