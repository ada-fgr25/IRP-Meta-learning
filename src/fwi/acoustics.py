"""Differentiable 2D acoustic wave propagation in JAX.

The implementation here is intentionally simple:
- second-order finite differences in space
- an explicit second-order update in time
- a lightweight damping frame at the boundary

That keeps the entire forward solve differentiable under JAX so the adjoint
needed for FWI comes from automatic differentiation of the time-stepping loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import BrainFWIConfig


def _ricker_wavelet(nt: int, dt: float, frequency_hz: float) -> jnp.ndarray:
    """Generate a compact source pulse for ultrasound transmission.

    A Ricker wavelet is a standard simple source model in seismic and acoustic
    inversion. It is not a perfect description of a medical transducer, but it
    gives us a clean band-limited pulse for a baseline experiment.
    """

    t = jnp.arange(nt) * dt
    t0 = 1.5 / frequency_hz
    arg = jnp.pi * frequency_hz * (t - t0)
    return (1.0 - 2.0 * arg**2) * jnp.exp(-(arg**2))


def _select_shot_indices(n_transducers: int, n_shots: int) -> jnp.ndarray:
    """Choose an evenly spaced, unique subset of transmitters.

    The first prototype used a direct `linspace(..., dtype=int)` call, which is
    concise but can become awkward when `n_shots` approaches `n_transducers`.
    This helper makes the intent explicit and prevents duplicate indices.
    """

    if n_transducers <= 0:
        raise ValueError("`n_transducers` must be positive.")
    if n_shots <= 0:
        raise ValueError("`n_shots` must be positive.")

    capped_n_shots = min(n_shots, n_transducers)
    shot_positions = (
        jnp.arange(capped_n_shots, dtype=jnp.float32) * n_transducers / capped_n_shots
    )
    shot_indices = jnp.floor(shot_positions).astype(jnp.int32)
    return shot_indices


def build_geometry(config: BrainFWIConfig) -> dict[str, jnp.ndarray]:
    """Create an elliptical transducer ring inspired by the Stride brain setup.

    Returns integer grid indices for all transducers, the subset used as
    transmitters, and the source wavelet shared across shots.
    """

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

    # We use a fixed subset of transmitters to keep the baseline inexpensive,
    # but we make the selection explicit so denser surveys remain well defined.
    shot_indices = _select_shot_indices(acq.n_transducers, acq.n_shots)

    return {
        "transducer_indices": indices,
        "shot_indices": shot_indices,
        "source_wavelet": _ricker_wavelet(
            config.time.nt, config.time.dt, acq.source_frequency_hz
        )
        * acq.source_amplitude,
    }


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
    geometry: dict[str, jnp.ndarray],
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
    receivers = geometry["transducer_indices"]
    source = receivers[shot_index]
    wavelet = geometry["source_wavelet"]
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
    geometry: dict[str, jnp.ndarray],
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Simulate all configured shots in the acquisition.

    The returned tensor has shape `[shot, time, receiver]`, which is the data
    cube used by the FWI loss.
    """

    return jax.vmap(
        lambda shot_idx: simulate_shot(velocity, geometry, config, shot_idx)
    )(geometry["shot_indices"])
