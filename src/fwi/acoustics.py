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


def _build_stride_like_damping_sigma(
    config: BrainFWIConfig,
    velocity: jnp.ndarray,
) -> jnp.ndarray:
    """Build a Stride-inspired damping coefficient field (`sigma`).

    Stride's boundary helper creates a per-dimension damping profile in the
    absorbing frame and sums it across dimensions. We replicate the same
    high-level shape control (`sine`/`power`) and reflection-coefficient-based
    scaling, then convert `sigma` into a multiplicative mask for our explicit
    time stepping.
    """

    nx = int(config.grid.nx)
    ny = int(config.grid.ny)
    dx = jnp.asarray(config.grid.dx, dtype=velocity.dtype)
    dy = jnp.asarray(config.grid.dy, dtype=velocity.dtype)
    cells = max(int(config.solver.damping_cells), 0)
    if cells == 0:
        return jnp.zeros((nx, ny), dtype=velocity.dtype)

    damping_type = config.solver.damping_type
    power_degree = max(int(config.solver.damping_power_degree), 1)
    reflection = jnp.asarray(
        max(float(config.solver.damping_reflection_coefficient), 1.0e-12),
        dtype=velocity.dtype,
    )

    def dimension_coefficient(cell_width: int, spacing: jnp.ndarray) -> jnp.ndarray:
        custom_coeff = config.solver.damping_max_coefficient
        if custom_coeff is not None:
            return jnp.asarray(custom_coeff, dtype=velocity.dtype)

        if cell_width > 15:
            return (
                jnp.asarray((power_degree + 1.0) / 2.0, dtype=velocity.dtype)
                * jnp.log(1.0 / reflection)
                / (jnp.asarray(cell_width, dtype=velocity.dtype) * spacing)
            )
        return jnp.asarray(0.67, dtype=velocity.dtype) / spacing

    coeff_x = dimension_coefficient(cells, dx)
    coeff_y = dimension_coefficient(cells, dy)

    # Distance in cells from each edge.
    ix = jnp.minimum(jnp.arange(nx), jnp.arange(nx)[::-1]).astype(velocity.dtype)
    iy = jnp.minimum(jnp.arange(ny), jnp.arange(ny)[::-1]).astype(velocity.dtype)

    # Convert to Stride-style profile coordinate:
    # - 1 at the outer edge
    # - 0 at the inner edge of the absorbing frame
    denom = max(cells - 1, 1)
    px = jnp.clip((cells - 1 - ix) / denom, 0.0, 1.0)
    py = jnp.clip((cells - 1 - iy) / denom, 0.0, 1.0)

    if damping_type == "sine":
        px = px - jnp.sin(2.0 * jnp.pi * px) / (2.0 * jnp.pi)
        py = py - jnp.sin(2.0 * jnp.pi * py) / (2.0 * jnp.pi)
    elif damping_type == "power":
        px = px**power_degree
        py = py**power_degree
    else:
        raise ValueError(
            f"Unsupported damping_type '{damping_type}'. Use 'sine' or 'power'."
        )

    sigma = coeff_x * px[:, None] + coeff_y * py[None, :]
    if config.solver.damping_velocity_scale:
        sigma = sigma * jnp.max(velocity)

    return sigma.astype(velocity.dtype)


def _build_interior_clamp_mask(config: BrainFWIConfig, shape, dtype) -> jnp.ndarray:
    """Build a fixed-edge clamp mask shared by all boundary modes."""

    interior = jnp.ones(shape, dtype=dtype)
    interior = interior.at[0, :].set(0.0)
    interior = interior.at[-1, :].set(0.0)
    interior = interior.at[:, 0].set(0.0)
    interior = interior.at[:, -1].set(0.0)
    return interior


def _build_boundary_terms(
    config: BrainFWIConfig,
    velocity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return `(boundary_mask, sponge_damp)` for the selected boundary mode.

    The current JAX solver supports:
    - `legacy`: simple quadratic taper mask
    - `stride_like`: Stride-inspired absorbing profile converted to mask
    - `sponge2`: Stride-like second-order sponge damping coefficient
    """

    interior = _build_interior_clamp_mask(config, velocity.shape, velocity.dtype)

    if config.solver.damping_mode == "legacy":
        damping = _build_damping_mask(config).astype(velocity.dtype)
        return damping * interior, jnp.zeros_like(velocity)

    if config.solver.damping_mode == "stride_like":
        sigma = _build_stride_like_damping_sigma(config, velocity)
        damping = jnp.exp(-sigma * config.time.dt)
        return damping * interior, jnp.zeros_like(velocity)

    if config.solver.damping_mode == "sponge2":
        sigma = _build_stride_like_damping_sigma(config, velocity)
        # Stride's SpongeBoundary2 scales damping by 7*dt before injecting it in
        # the second-order damped update equation.
        sponge_damp = jnp.asarray(7.0, dtype=velocity.dtype) * sigma * config.time.dt
        return interior, sponge_damp

    raise ValueError(
        f"Unsupported damping_mode '{config.solver.damping_mode}'. "
        "Use 'legacy', 'stride_like', or 'sponge2'."
    )


def _inject_linear_point(
    point_index: jnp.ndarray,
    point_value: jnp.ndarray,
    shape: tuple[int, int],
) -> jnp.ndarray:
    """Inject a scalar at one nearest-gridpoint location."""

    return jnp.zeros(shape).at[point_index[0], point_index[1]].set(point_value)


def _hicks_offsets() -> jnp.ndarray:
    """Return the fixed Hicks stencil offsets used by the Stride reference."""

    return jnp.arange(-3, 4, dtype=jnp.int32)


def _hicks_weights_2d(coefficients: jnp.ndarray) -> jnp.ndarray:
    """Build separable 2D Hicks weights from `[dim, coeff]` arrays.

    Stride stores an extra trailing coefficient slot (`r+1`), so we keep parity
    by consuming only the first seven populated taps.
    """

    wx = coefficients[0, :7]
    wy = coefficients[1, :7]
    return wx[:, None] * wy[None, :]


def _clip_hicks_indices(indices: jnp.ndarray, max_index: int) -> jnp.ndarray:
    """Clamp Hicks support indices to valid grid bounds."""

    return jnp.clip(indices, 0, max_index)


def _sample_hicks_point(
    field: jnp.ndarray,
    reference_gridpoint: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> jnp.ndarray:
    """Sample one point from the grid with Stride-like Hicks interpolation."""

    offsets = _hicks_offsets()
    ii = _clip_hicks_indices(reference_gridpoint[0] + offsets, field.shape[0] - 1)
    jj = _clip_hicks_indices(reference_gridpoint[1] + offsets, field.shape[1] - 1)
    patch = field[ii[:, None], jj[None, :]]
    return jnp.sum(_hicks_weights_2d(coefficients) * patch)


def _inject_hicks_point(
    reference_gridpoint: jnp.ndarray,
    coefficients: jnp.ndarray,
    point_value: jnp.ndarray,
    shape: tuple[int, int],
) -> jnp.ndarray:
    """Scatter one point value onto the grid with Hicks interpolation weights."""

    offsets = _hicks_offsets()
    ii = _clip_hicks_indices(reference_gridpoint[0] + offsets, shape[0] - 1)
    jj = _clip_hicks_indices(reference_gridpoint[1] + offsets, shape[1] - 1)
    weighted = point_value * _hicks_weights_2d(coefficients)
    return jnp.zeros(shape).at[ii[:, None], jj[None, :]].add(weighted)


def _inject_source(
    source_index: jnp.ndarray,
    source_reference_gridpoint: jnp.ndarray,
    source_coefficients: jnp.ndarray,
    source_value: jnp.ndarray,
    shape: tuple[int, int],
    use_hicks: bool,
) -> jnp.ndarray:
    """Inject one source sample according to the configured interpolation mode."""

    if use_hicks:
        return _inject_hicks_point(
            source_reference_gridpoint,
            source_coefficients,
            source_value,
            shape,
        )
    return _inject_linear_point(source_index, source_value, shape)


def _sample_receivers(
    field: jnp.ndarray,
    receiver_i: jnp.ndarray,
    receiver_j: jnp.ndarray,
    receiver_reference_gridpoints: jnp.ndarray,
    receiver_coefficients: jnp.ndarray,
    use_hicks: bool,
) -> jnp.ndarray:
    """Sample receiver traces from the wavefield in linear or Hicks mode."""

    if use_hicks:
        return jax.vmap(_sample_hicks_point, in_axes=(None, 0, 0))(
            field,
            receiver_reference_gridpoints,
            receiver_coefficients,
        )
    return field[receiver_i, receiver_j]


def _inject_receivers(
    receiver_i: jnp.ndarray,
    receiver_j: jnp.ndarray,
    receiver_reference_gridpoints: jnp.ndarray,
    receiver_coefficients: jnp.ndarray,
    receiver_values: jnp.ndarray,
    shape: tuple[int, int],
    use_hicks: bool,
) -> jnp.ndarray:
    """Scatter receiver-domain cotangents back onto the grid.

    The injection uses additive scatter so repeated point contributions are
    accumulated correctly for both nearest-point and Hicks interpolation.
    """

    if use_hicks:
        per_receiver = jax.vmap(_inject_hicks_point, in_axes=(0, 0, 0, None))(
            receiver_reference_gridpoints,
            receiver_coefficients,
            receiver_values,
            shape,
        )
        return jnp.sum(per_receiver, axis=0)

    return jnp.zeros(shape).at[receiver_i, receiver_j].add(receiver_values)


def _time_derivative(samples: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Approximate the first time derivative the way Stride prepares sources.

    Stride uses `np.gradient(..., dt)` when `diff_source=True`. Reproducing the
    same central-difference stencil here keeps the JAX source preparation close
    to the reference while remaining compatible with JIT compilation.
    """

    if samples.shape[0] <= 1:
        return jnp.zeros_like(samples)

    # `np.gradient` uses one-sided differences at the ends and centred
    # differences in the interior. The explicit construction keeps that
    # behaviour readable and avoids depending on NumPy-only helpers.
    first = (samples[1] - samples[0]) / dt
    last = (samples[-1] - samples[-2]) / dt
    middle = (samples[2:] - samples[:-2]) / (2.0 * dt)
    return jnp.concatenate((first[None], middle, last[None]))


def _prepare_source_wavelet(
    wavelet: jnp.ndarray, config: BrainFWIConfig
) -> jnp.ndarray:
    """Convert the acquisition wavelet into the injected source samples.

    The default path mirrors Stride's source preparation:
    - optionally replace the raw wavelet by its first time derivative
    - scale the injected sample later using `2 * dt**2 * vp / max(dx, dy)`
    - divide once more by `dt` when injecting the undifferentiated source
    """

    if config.solver.diff_source:
        return _time_derivative(wavelet, config.time.dt)
    return wavelet


def _source_scale(
    velocity_at_source: jnp.ndarray,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Return the per-shot source scaling used by the Stride Devito kernel."""

    if config.solver.source_scale_mode != "stride":
        raise ValueError(
            "Unsupported source scaling mode " f"'{config.solver.source_scale_mode}'."
        )

    dt = config.time.dt
    h_max = max(config.grid.dx, config.grid.dy)
    scale = 2.0 * (dt**2) * velocity_at_source / h_max
    if not config.solver.diff_source:
        scale = scale / dt
    return scale


def _spatial_operator(
    field: jnp.ndarray,
    velocity_sq: jnp.ndarray,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Evaluate the discrete spatial operator used by the current kernel.

    For `OT2`, this is the familiar `vp**2 * Lap(u)`. For `OT4`, we mirror the
    Stride `IsoAcousticDevito` correction term and add
    `dt**2 / 12 * vp**2 * Lap(vp**2 * Lap(u))`.
    """

    laplacian_2 = velocity_sq * _laplacian(field, config.grid.dx, config.grid.dy)
    if config.solver.kernel == "OT2":
        return laplacian_2
    if config.solver.kernel != "OT4":
        raise ValueError(f"Unsupported solver kernel '{config.solver.kernel}'.")

    laplacian_4 = velocity_sq * _laplacian(
        laplacian_2,
        config.grid.dx,
        config.grid.dy,
    )
    return laplacian_2 + ((config.time.dt**2) / 12.0) * laplacian_4


def _propose_next_field(
    u_prev: jnp.ndarray,
    u_curr: jnp.ndarray,
    velocity: jnp.ndarray,
    source_index: jnp.ndarray,
    source_reference_gridpoint: jnp.ndarray,
    source_coefficients: jnp.ndarray,
    source_sample: jnp.ndarray,
    boundary_mask: jnp.ndarray,
    sponge_damp: jnp.ndarray,
    use_hicks: bool,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Apply one discrete wave-equation step before receiver sampling.

    Structuring the forward update as one explicit pure function lets the
    reverse-time implementation reuse its exact linearisation via `jax.vjp`.
    That keeps the explicit adjoint aligned with whatever discrete physics the
    forward solver is using, including Stride-like source scaling and `OT4`.
    """

    velocity_sq = velocity**2
    pressure_update = (config.time.dt**2) * _spatial_operator(
        u_curr, velocity_sq, config
    )
    velocity_at_source = (
        _sample_hicks_point(
            velocity,
            source_reference_gridpoint,
            source_coefficients,
        )
        if use_hicks
        else velocity[source_index[0], source_index[1]]
    )
    scaled_source_sample = source_sample * _source_scale(velocity_at_source, config)
    source_update = _inject_source(
        source_index,
        source_reference_gridpoint,
        source_coefficients,
        scaled_source_sample,
        velocity.shape,
        use_hicks,
    )

    if config.solver.damping_mode == "sponge2":
        # Discrete analogue of Stride's second-order sponge damping term:
        #   u_tt - L + 2*damp*u_t + damp^2*u = source
        # with centred u_t approximation. Solving explicitly for u_{n+1} gives:
        #   u_{n+1} = [2u_n - (1-d)u_{n-1} + dt^2*L(u_n) + src - d^2*u_n] / (1+d)
        # where `d` is the pre-scaled damping coefficient field.
        numerator = (
            2.0 * u_curr
            - (1.0 - sponge_damp) * u_prev
            + pressure_update
            + source_update
            - (sponge_damp**2) * u_curr
        )
        return boundary_mask * (numerator / (1.0 + sponge_damp))

    return boundary_mask * (2.0 * u_curr - u_prev + pressure_update + source_update)


def _advance_state(
    u_prev: jnp.ndarray,
    u_curr: jnp.ndarray,
    velocity: jnp.ndarray,
    source_index: jnp.ndarray,
    source_reference_gridpoint: jnp.ndarray,
    source_coefficients: jnp.ndarray,
    source_sample: jnp.ndarray,
    active: jnp.ndarray,
    boundary_mask: jnp.ndarray,
    sponge_damp: jnp.ndarray,
    use_hicks: bool,
    config: BrainFWIConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Advance the solver state by one step or keep it unchanged if inactive."""

    proposed_u_next = _propose_next_field(
        u_prev,
        u_curr,
        velocity,
        source_index,
        source_reference_gridpoint,
        source_coefficients,
        source_sample,
        boundary_mask,
        sponge_damp,
        use_hicks,
        config,
    )
    u_next = jnp.where(active, proposed_u_next, u_curr)
    next_prev = jnp.where(active, u_curr, u_prev)
    return next_prev, u_next


def _step_shot_state(
    carry: tuple[jnp.ndarray, jnp.ndarray],
    source_value: jnp.ndarray,
    active: jnp.ndarray,
    source_index: jnp.ndarray,
    source_reference_gridpoint: jnp.ndarray,
    source_coefficients: jnp.ndarray,
    receiver_i: jnp.ndarray,
    receiver_j: jnp.ndarray,
    receiver_reference_gridpoints: jnp.ndarray,
    receiver_coefficients: jnp.ndarray,
    velocity: jnp.ndarray,
    boundary_mask: jnp.ndarray,
    sponge_damp: jnp.ndarray,
    use_hicks: bool,
    config: BrainFWIConfig,
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
    next_prev, u_next = _advance_state(
        u_prev,
        u_curr,
        velocity,
        source_index,
        source_reference_gridpoint,
        source_coefficients,
        source_value,
        active,
        boundary_mask,
        sponge_damp,
        use_hicks,
        config,
    )
    traces = jnp.where(
        active,
        _sample_receivers(
            u_next,
            receiver_i,
            receiver_j,
            receiver_reference_gridpoints,
            receiver_coefficients,
            use_hicks,
        ),
        0.0,
    )
    return (next_prev, u_next), (traces, u_prev, u_curr)


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

    receivers, _, wavelet = acquisition.require_solver_arrays()
    source_index = receivers[shot_index]
    receiver_i = receivers[:, 0]
    receiver_j = receivers[:, 1]
    use_hicks = acquisition.interpolation_type == "hicks"

    if use_hicks:
        source_reference_gridpoint = acquisition.source_reference_gridpoints[shot_index]
        source_coefficients = acquisition.source_coefficients[shot_index]
        receiver_reference_gridpoints = acquisition.receiver_reference_gridpoints
        receiver_coefficients = acquisition.receiver_coefficients
    else:
        # Keep lightweight placeholders so the linear path remains unchanged and
        # all scan-carried shapes stay static under JIT compilation.
        source_reference_gridpoint = source_index
        source_coefficients = jnp.zeros((2, 8), dtype=velocity.dtype)
        receiver_reference_gridpoints = receivers
        receiver_coefficients = jnp.zeros(
            (receivers.shape[0], 2, 8),
            dtype=velocity.dtype,
        )
    boundary_mask, sponge_damp = _build_boundary_terms(config, velocity)
    source_wavelet = _prepare_source_wavelet(wavelet, config)
    wavelet_segments, active_segments = _segment_wavelet(
        source_wavelet,
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
                source_index,
                source_reference_gridpoint,
                source_coefficients,
                receiver_i,
                receiver_j,
                receiver_reference_gridpoints,
                receiver_coefficients,
                velocity,
                boundary_mask,
                sponge_damp,
                use_hicks,
                config,
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
    source_index: jnp.ndarray,
    source_reference_gridpoint: jnp.ndarray,
    source_coefficients: jnp.ndarray,
    receiver_i: jnp.ndarray,
    receiver_j: jnp.ndarray,
    receiver_reference_gridpoints: jnp.ndarray,
    receiver_coefficients: jnp.ndarray,
    velocity: jnp.ndarray,
    boundary_mask: jnp.ndarray,
    sponge_damp: jnp.ndarray,
    use_hicks: bool,
    config: BrainFWIConfig,
    source_block: jnp.ndarray,
    active_block: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recompute the input states for one checkpoint segment.

    The explicit adjoint only needs the `(u_{n-1}, u_n)` pairs that feed each
    step. Replaying them from sparse checkpoints is much cheaper than storing
    every wavefield outright.
    """

    _, (_, prev_fields, curr_fields) = jax.lax.scan(
        lambda state, step_xs: _step_shot_state(
            state,
            step_xs[0],
            step_xs[1],
            source_index,
            source_reference_gridpoint,
            source_coefficients,
            receiver_i,
            receiver_j,
            receiver_reference_gridpoints,
            receiver_coefficients,
            velocity,
            boundary_mask,
            sponge_damp,
            use_hicks,
            config,
        ),
        start_carry,
        (source_block, active_block),
    )
    return prev_fields, curr_fields


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
    receivers, acquisition_shot_indices, _ = acquisition.require_solver_arrays()
    active_shot_indices = (
        acquisition_shot_indices if shot_indices is None else shot_indices
    )
    receiver_i = receivers[:, 0]
    receiver_j = receivers[:, 1]
    boundary_mask, sponge_damp = _build_boundary_terms(config, velocity)
    grid_shape = velocity.shape

    def shot_loss_grad(shot_index: jnp.ndarray, observed_shot: jnp.ndarray):
        """Run one forward/adjoint pair and return its loss contribution."""

        source_index = receivers[shot_index]
        use_hicks = acquisition.interpolation_type == "hicks"
        if use_hicks:
            source_reference_gridpoint = acquisition.source_reference_gridpoints[
                shot_index
            ]
            source_coefficients = acquisition.source_coefficients[shot_index]
            receiver_reference_gridpoints = acquisition.receiver_reference_gridpoints
            receiver_coefficients = acquisition.receiver_coefficients
        else:
            source_reference_gridpoint = source_index
            source_coefficients = jnp.zeros((2, 8), dtype=velocity.dtype)
            receiver_reference_gridpoints = receivers
            receiver_coefficients = jnp.zeros(
                (receivers.shape[0], 2, 8),
                dtype=velocity.dtype,
            )
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

            cotangent_next_prev, cotangent_next_curr, grad_velocity = carry
            prev_field, curr_field, data_cotangent, active, source_block_value = xs

            def active_reverse(state):
                active_cotangent_next_prev, active_cotangent_next_curr, active_grad = (
                    state
                )

                # The observation operator samples the next wavefield at receiver
                # points, so its adjoint adds the trace-domain cotangents to the
                # second component of the output state `(u_n, u_{n+1})`.
                active_cotangent_next_curr = (
                    active_cotangent_next_curr
                    + _inject_receivers(
                        receiver_i,
                        receiver_j,
                        receiver_reference_gridpoints,
                        receiver_coefficients,
                        data_cotangent,
                        grid_shape,
                        use_hicks,
                    )
                )

                # The state transition is treated as a pure function, so we can
                # reuse JAX's exact linearisation of the discrete forward model.
                # This keeps the explicit adjoint in lockstep with the solver even
                # as we change kernels or source scaling details.
                def state_transition(
                    step_prev: jnp.ndarray,
                    step_curr: jnp.ndarray,
                    step_velocity: jnp.ndarray,
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
                    return (
                        step_curr,
                        _propose_next_field(
                            step_prev,
                            step_curr,
                            step_velocity,
                            source_index,
                            source_reference_gridpoint,
                            source_coefficients,
                            source_block_value,
                            boundary_mask,
                            sponge_damp,
                            use_hicks,
                            config,
                        ),
                    )

                (_, _), state_vjp = jax.vjp(
                    state_transition,
                    prev_field,
                    curr_field,
                    velocity,
                )
                cotangent_prev, cotangent_curr, velocity_cotangent = state_vjp(
                    (active_cotangent_next_prev, active_cotangent_next_curr)
                )
                return (
                    cotangent_prev,
                    cotangent_curr,
                    active_grad + velocity_cotangent,
                )

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
            prev_fields, curr_fields = _replay_segment_history(
                (checkpoint_prev, checkpoint_curr),
                source_index,
                source_reference_gridpoint,
                source_coefficients,
                receiver_i,
                receiver_j,
                receiver_reference_gridpoints,
                receiver_coefficients,
                velocity,
                boundary_mask,
                sponge_damp,
                use_hicks,
                config,
                source_block,
                active_block,
            )

            carry, _ = jax.lax.scan(
                reverse_step,
                carry,
                (
                    prev_fields,
                    curr_fields,
                    data_block,
                    active_block,
                    source_block,
                ),
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
