"""Differentiable 2D acoustic wave propagation in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import BrainFWIConfig


def _ricker_wavelet(nt: int, dt: float, frequency_hz: float) -> jnp.ndarray:
    """Generate a compact source pulse for ultrasound transmission."""

    t = jnp.arange(nt) * dt
    t0 = 1.5 / frequency_hz
    arg = jnp.pi * frequency_hz * (t - t0)
    return (1.0 - 2.0 * arg**2) * jnp.exp(-arg**2)


def build_geometry(config: BrainFWIConfig) -> dict[str, jnp.ndarray]:
    """Create an elliptical transducer ring inspired by the Stride brain setup."""

    grid = config.grid
    acq = config.acquisition
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, acq.n_transducers, endpoint=False)
    centre = jnp.array([(grid.nx - 1) / 2.0, (grid.ny - 1) / 2.0])
    radius = jnp.array(
        [
            acq.ellipse_scale_x * (grid.nx - 1) / 2.0,
            acq.ellipse_scale_y * (grid.ny - 1) / 2.0,
        ]
    )
    coords = jnp.stack(
        [
            centre[0] + radius[0] * jnp.cos(angles),
            centre[1] + radius[1] * jnp.sin(angles),
        ],
        axis=-1,
    )
    indices = jnp.rint(coords).astype(jnp.int32)

    # We use a fixed subset of transmitters to keep the baseline inexpensive.
    shot_indices = jnp.linspace(0, acq.n_transducers - 1, acq.n_shots, dtype=jnp.int32)

    return {
        "transducer_indices": indices,
        "shot_indices": shot_indices,
        "source_wavelet": _ricker_wavelet(
            config.time.nt, config.time.dt, acq.source_frequency_hz
        )
        * acq.source_amplitude,
    }


def _laplacian(field: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Second-order finite-difference Laplacian with fixed zero edges."""

    centre = field[1:-1, 1:-1]
    lap = (
        (field[2:, 1:-1] - 2.0 * centre + field[:-2, 1:-1]) / dx**2
        + (field[1:-1, 2:] - 2.0 * centre + field[1:-1, :-2]) / dy**2
    )
    return jnp.pad(lap, ((1, 1), (1, 1)))


def _build_damping_mask(config: BrainFWIConfig) -> jnp.ndarray:
    """Create a simple absorbing frame to reduce edge reflections."""

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
    geometry: dict[str, jnp.ndarray],
    config: BrainFWIConfig,
    shot_index: jnp.ndarray,
) -> jnp.ndarray:
    """Simulate one transmit event and record traces at every transducer."""

    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    receivers = geometry["transducer_indices"]
    source = receivers[shot_index]
    wavelet = geometry["source_wavelet"]
    damping = _build_damping_mask(config)
    receiver_i = receivers[:, 0]
    receiver_j = receivers[:, 1]

    def step(carry, source_value):
        u_prev, u_curr = carry
        lap = _laplacian(u_curr, dx, dy)

        # Inject the source pulse at a single transducer location.
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
    geometry: dict[str, jnp.ndarray],
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Simulate all configured shots in the acquisition."""

    return jax.vmap(
        lambda shot_idx: simulate_shot(velocity, geometry, config, shot_idx)
    )(geometry["shot_indices"])
