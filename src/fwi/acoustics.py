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
from .filtering import bandlimit_traces


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


def _step_shot_state(
    carry: tuple[jnp.ndarray, jnp.ndarray],
    source_value: jnp.ndarray,
    active: jnp.ndarray,
    source: jnp.ndarray,
    receiver_i: jnp.ndarray,
    receiver_j: jnp.ndarray,
    velocity_sq: jnp.ndarray,
    boundary_mask: jnp.ndarray,
    grid_shape: tuple[int, int],
    dt: float,
    dx: float,
    dy: float,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
]:
    """Advance one time step, optionally turning padded steps into no-ops.

    Fixed-length checkpoint segments are easier for JAX to compile than a
    variable-length last segment. To keep the maths exact, padded steps past the
    physical recording window are turned into explicit no-ops rather than extra
    wave-equation updates.
    """

    u_prev, u_curr = carry
    lap_curr = _laplacian(u_curr, dx, dy)
    source_term = _inject_source(source, source_value, grid_shape)
    proposed_u_next = boundary_mask * (
        2.0 * u_curr - u_prev + (dt**2) * (velocity_sq * lap_curr + source_term)
    )

    u_next = jnp.where(active, proposed_u_next, u_curr)
    next_prev = jnp.where(active, u_curr, u_prev)
    traces = jnp.where(active, u_next[receiver_i, receiver_j], 0.0)
    return (next_prev, u_next), (traces, u_curr, lap_curr)


def _segment_wavelet(
    wavelet: jnp.ndarray,
    checkpoint_interval: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split the source wavelet into fixed-size segments plus an active mask."""

    interval = max(int(checkpoint_interval), 1)
    pad = (-int(wavelet.shape[0])) % interval
    padded_wavelet = jnp.pad(wavelet, (0, pad))
    active = jnp.arange(padded_wavelet.shape[0]) < wavelet.shape[0]
    return padded_wavelet.reshape((-1, interval)), active.reshape((-1, interval))


def _pad_traces_to_segments(
    traces: jnp.ndarray,
    n_segments: int,
    segment_length: int,
) -> jnp.ndarray:
    """Pad a `[time, receiver]` tensor so it reshapes into segment blocks."""

    target_length = int(n_segments) * int(segment_length)
    pad = target_length - int(traces.shape[0])
    return jnp.pad(traces, ((0, pad), (0, 0))).reshape(
        n_segments,
        segment_length,
        traces.shape[1],
    )


def _simulate_shot_with_checkpoints(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    shot_index: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one shot while storing sparse checkpoints for later recomputation.

    Instead of keeping every wavefield in memory, we save only the `(u_{n-1},
    u_n)` pair at the start of each checkpoint segment and keep the full trace
    history. The adjoint later replays each segment on demand, which trades
    extra compute for a much smaller peak memory footprint.
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
    velocity_sq = velocity**2
    wavelet_segments, active_segments = _segment_wavelet(
        wavelet,
        config.solver.checkpoint_interval,
    )

    def run_segment(carry, xs):
        """Advance one checkpoint segment and save its starting state."""

        source_block, active_block = xs
        segment_start_prev, segment_start_curr = carry
        carry, (segment_traces, _, _) = jax.lax.scan(
            lambda state, step_xs: _step_shot_state(
                state,
                step_xs[0],
                step_xs[1],
                source,
                receiver_i,
                receiver_j,
                velocity_sq,
                boundary_mask,
                grid_shape,
                dt,
                dx,
                dy,
            ),
            carry,
            (source_block, active_block),
        )
        return carry, (segment_start_prev, segment_start_curr, segment_traces)

    init = (jnp.zeros_like(velocity), jnp.zeros_like(velocity))
    _, (checkpoint_prevs, checkpoint_currs, segment_traces) = jax.lax.scan(
        run_segment,
        init,
        (wavelet_segments, active_segments),
    )
    traces = segment_traces.reshape((-1, receivers.shape[0]))[: wavelet.shape[0]]
    return traces, checkpoint_prevs, checkpoint_currs, wavelet_segments, active_segments


def _replay_segment_history(
    start_carry: tuple[jnp.ndarray, jnp.ndarray],
    source: jnp.ndarray,
    receiver_i: jnp.ndarray,
    receiver_j: jnp.ndarray,
    velocity_sq: jnp.ndarray,
    boundary_mask: jnp.ndarray,
    grid_shape: tuple[int, int],
    dt: float,
    dx: float,
    dy: float,
    source_block: jnp.ndarray,
    active_block: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recompute the current fields and Laplacians for one checkpoint segment."""

    _, (_, curr_fields, laplacians) = jax.lax.scan(
        lambda state, step_xs: _step_shot_state(
            state,
            step_xs[0],
            step_xs[1],
            source,
            receiver_i,
            receiver_j,
            velocity_sq,
            boundary_mask,
            grid_shape,
            dt,
            dx,
            dy,
        ),
        start_carry,
        (source_block, active_block),
    )
    return curr_fields, laplacians


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

    traces, _, _, _, _ = _simulate_shot_with_checkpoints(
        velocity, acquisition, config, shot_index
    )
    return traces


def simulate_survey(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    shot_indices: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Simulate all configured shots in the acquisition.

    The returned tensor has shape `[shot, time, receiver]`, which is the data
    cube used by the FWI loss.
    """

    active_shot_indices = (
        acquisition.require_solver_arrays()[1] if shot_indices is None else shot_indices
    )
    return jax.vmap(
        lambda shot_idx: simulate_shot(velocity, acquisition, config, shot_idx)
    )(active_shot_indices)


def loss_and_grad(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    observed_data: jnp.ndarray,
    f_max_hz: float | None = None,
    shot_indices: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the survey loss and explicit adjoint gradient in JAX.

    The gradient is produced by an explicit reverse-time adjoint
    implementation. The data misfit itself now mirrors the tracked Stride
    benchmark more closely by combining a Stride-style `0.5 * sum(r^2)` loss
    with an optional `f_max` low-pass filter applied in the trace domain.
    """

    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    receivers, acquisition_shot_indices, _ = acquisition.require_solver_arrays()
    active_shot_indices = (
        acquisition_shot_indices if shot_indices is None else shot_indices
    )
    receiver_i = receivers[:, 0]
    receiver_j = receivers[:, 1]
    velocity_sq = velocity**2
    boundary_mask = _build_boundary_mask(config)
    grid_shape = velocity.shape

    def shot_loss_grad(shot_index: jnp.ndarray, observed_shot: jnp.ndarray):
        """Run one forward/adjoint pair and return its loss contribution."""

        source = receivers[shot_index]
        (
            traces,
            checkpoint_prevs,
            checkpoint_currs,
            wavelet_segments,
            active_segments,
        ) = _simulate_shot_with_checkpoints(
            velocity,
            acquisition,
            config,
            shot_index,
        )
        residual = traces - observed_shot

        # The FFT mask defines a linear zero-phase operator. Applying it to the
        # residual gives us the band-limited misfit used for the current stage,
        # and because the filter is symmetric in time the same filtered residual
        # also acts as the trace-domain cotangent for the adjoint.
        residual = bandlimit_traces(residual, dt, f_max_hz, axis=0)
        shot_loss = 0.5 * jnp.sum(residual**2)
        data_cotangents = residual
        padded_data_cotangents = _pad_traces_to_segments(
            data_cotangents,
            wavelet_segments.shape[0],
            wavelet_segments.shape[1],
        )

        def reverse_step(carry, xs):
            """Reverse one time step of the discrete wave equation."""

            cotangent_curr, cotangent_next, grad_velocity = carry
            curr_field, laplacian_curr, data_cotangent, active = xs

            def active_reverse(state):
                active_cotangent_curr, active_cotangent_next, active_grad_velocity = (
                    state
                )

                # The observation operator samples the next wavefield at receiver
                # points, so its adjoint scatters those trace-domain cotangents back
                # onto the full grid before the wave-equation reverse step.
                active_cotangent_next = active_cotangent_next + _inject_receivers(
                    receiver_i,
                    receiver_j,
                    data_cotangent,
                    grid_shape,
                )

                # The boundary operator is just a diagonal mask, so it is its own
                # adjoint. Applying it here mirrors the boundary treatment used in
                # the forward step.
                masked_cotangent = boundary_mask * active_cotangent_next

                # The gradient with respect to velocity comes from differentiating
                # `velocity**2 * laplacian(u_n)` pointwise at each time step.
                active_grad_velocity = (
                    active_grad_velocity
                    + 2.0 * velocity * (dt**2) * masked_cotangent * laplacian_curr
                )

                # Reverse propagation through the explicit second-order update:
                # - `u_{n-1}` receives `-masked_cotangent`
                # - `u_n` receives the direct `2 * masked_cotangent` term
                # - plus the adjoint of the spatial operator
                cotangent_prev = -masked_cotangent
                active_cotangent_curr = (
                    active_cotangent_curr
                    + 2.0 * masked_cotangent
                    + (dt**2) * (_laplacian(velocity_sq * masked_cotangent, dx, dy))
                )
                return (cotangent_prev, active_cotangent_curr, active_grad_velocity)

            return (
                jax.lax.cond(
                    active,
                    active_reverse,
                    lambda state: state,
                    carry,
                ),
                None,
            )

        def reverse_segment(carry, xs):
            """Replay one segment then sweep its steps in reverse."""

            checkpoint_prev, checkpoint_curr, source_block, active_block, data_block = (
                xs
            )
            curr_fields, laplacians = _replay_segment_history(
                (checkpoint_prev, checkpoint_curr),
                source,
                receiver_i,
                receiver_j,
                velocity_sq,
                boundary_mask,
                grid_shape,
                dt,
                dx,
                dy,
                source_block,
                active_block,
            )

            carry, _ = jax.lax.scan(
                reverse_step,
                carry,
                (curr_fields, laplacians, data_block, active_block),
                reverse=True,
            )
            return carry, None

        init_carry = (
            jnp.zeros_like(velocity),
            jnp.zeros_like(velocity),
            jnp.zeros_like(velocity),
        )
        (cotangent_initial_prev, cotangent_initial_curr, shot_grad), _ = jax.lax.scan(
            reverse_segment,
            init_carry,
            (
                checkpoint_prevs,
                checkpoint_currs,
                wavelet_segments,
                active_segments,
                padded_data_cotangents,
            ),
            reverse=True,
        )

        # The initial wavefields are fixed zeros rather than optimisation
        # variables, so their cotangents are intentionally ignored.
        del cotangent_initial_prev, cotangent_initial_curr
        return shot_loss, shot_grad

    # Accumulate shots sequentially rather than with `vmap`. This keeps only
    # one shot's forward/adjoint history live at a time, which is much more
    # memory-friendly on large benchmark-sized problems.
    def accumulate_shot(carry, xs):
        total_loss, total_grad = carry
        shot_index, observed_shot = xs
        shot_loss, shot_grad = shot_loss_grad(shot_index, observed_shot)
        return (total_loss + shot_loss, total_grad + shot_grad), None

    init = (jnp.array(0.0, dtype=velocity.dtype), jnp.zeros_like(velocity))
    (total_loss, total_grad), _ = jax.lax.scan(
        accumulate_shot,
        init,
        (active_shot_indices, observed_data),
    )
    return total_loss, total_grad
