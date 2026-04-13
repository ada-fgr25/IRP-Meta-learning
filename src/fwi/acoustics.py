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

    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    receivers, _, wavelet = acquisition.require_solver_arrays()
    source = receivers[shot_index]
    damping = _build_damping_mask(config)
    receiver_i = receivers[:, 0]
    receiver_j = receivers[:, 1]

    def step(carry, source_value):
        """Advance the wavefield by one time step and sample all receivers."""

        u_prev, u_curr = carry
        lap = _laplacian(u_curr, dx, dy)

        # Inject the source pulse at a single transducer location. Because the
        # solver update is multiplied by `dt**2`, the source amplitude in the
        # configuration is intentionally large enough to keep the observed data
        # numerically meaningful at this coarse baseline scale.
        source_term = jnp.zeros_like(u_curr).at[source[0], source[1]].set(source_value)
        u_next = 2.0 * u_curr - u_prev + (dt**2) * ((velocity**2) * lap + source_term)

        # Apply a light damping frame and keep the outermost cells fixed at zero.
        u_next = damping * u_next
        u_next = u_next.at[0, :].set(0.0)
        u_next = u_next.at[-1, :].set(0.0)
        u_next = u_next.at[:, 0].set(0.0)
        u_next = u_next.at[:, -1].set(0.0)

        traces = u_next[receiver_i, receiver_j]
        return (u_curr, u_next), traces

    init = (
        jnp.zeros_like(velocity),
        jnp.zeros_like(velocity),
    )
    _, traces = jax.lax.scan(step, init, wavelet)
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
