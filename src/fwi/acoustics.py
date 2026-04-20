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
from .medium import AcousticMedium


_SECOND_DERIVATIVE_WEIGHTS = {
    2: (-2.0, 1.0),
    10: (
        -5269.0 / 1800.0,
        5.0 / 3.0,
        -5.0 / 21.0,
        5.0 / 126.0,
        -5.0 / 1008.0,
        1.0 / 3150.0,
    ),
}

_FIRST_DERIVATIVE_WEIGHTS = {
    2: (0.5,),
    10: (
        5.0 / 6.0,
        -5.0 / 21.0,
        5.0 / 84.0,
        -5.0 / 504.0,
        1.0 / 630.0,
    ),
}


def _first_derivative_1d(
    field: jnp.ndarray,
    spacing: float,
    axis: int,
    space_order: int,
) -> jnp.ndarray:
    """Apply a central finite-difference first derivative along one axis."""

    weights = _FIRST_DERIVATIVE_WEIGHTS.get(space_order)
    if weights is None:
        raise ValueError(
            f"Unsupported space_order '{space_order}'. Use one of "
            f"{sorted(_FIRST_DERIVATIVE_WEIGHTS)}."
        )

    radius = len(weights)
    centre_slices = [slice(None)] * field.ndim
    centre_slices[axis] = slice(radius, field.shape[axis] - radius)
    derivative = jnp.zeros_like(field[tuple(centre_slices)])

    for offset, weight in enumerate(weights, start=1):
        plus_slices = [slice(None)] * field.ndim
        minus_slices = [slice(None)] * field.ndim
        plus_slices[axis] = slice(radius + offset, field.shape[axis] - radius + offset)
        minus_slices[axis] = slice(radius - offset, field.shape[axis] - radius - offset)
        derivative = derivative + weight * (
            field[tuple(plus_slices)] - field[tuple(minus_slices)]
        )

    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (radius, radius)
    return jnp.pad(derivative / spacing, pad_width)


def _second_derivative_1d(
    field: jnp.ndarray,
    spacing: float,
    axis: int,
    space_order: int,
) -> jnp.ndarray:
    """Apply a central finite-difference second derivative along one axis.

    The repository originally used only the three-point stencil. To approach
    Stride's `space_order=10` behaviour more closely we now support the same
    higher-order central derivative family on the interior of the padded solver
    domain, while keeping the outer stencil radius zeroed where insufficient
    neighbours are available.
    """

    weights = _SECOND_DERIVATIVE_WEIGHTS.get(space_order)
    if weights is None:
        raise ValueError(
            f"Unsupported space_order '{space_order}'. Use one of "
            f"{sorted(_SECOND_DERIVATIVE_WEIGHTS)}."
        )

    radius = len(weights) - 1
    centre_slices = [slice(None)] * field.ndim
    centre_slices[axis] = slice(radius, field.shape[axis] - radius)
    derivative = weights[0] * field[tuple(centre_slices)]

    for offset, weight in enumerate(weights[1:], start=1):
        plus_slices = [slice(None)] * field.ndim
        minus_slices = [slice(None)] * field.ndim
        plus_slices[axis] = slice(radius + offset, field.shape[axis] - radius + offset)
        minus_slices[axis] = slice(radius - offset, field.shape[axis] - radius - offset)
        derivative = derivative + weight * (
            field[tuple(plus_slices)] + field[tuple(minus_slices)]
        )

    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (radius, radius)
    return jnp.pad(derivative / (spacing**2), pad_width)


def _laplacian(
    field: jnp.ndarray,
    dx: float,
    dy: float,
    space_order: int,
) -> jnp.ndarray:
    """Finite-difference Laplacian with configurable interior accuracy."""

    return _second_derivative_1d(field, dx, axis=0, space_order=space_order) + (
        _second_derivative_1d(field, dy, axis=1, space_order=space_order)
    )


def _solver_padding(config: BrainFWIConfig) -> tuple[int, int]:
    """Return the Stride-like extra halo width on each spatial axis."""

    return (
        max(int(config.solver.extra_cells_x), 0),
        max(int(config.solver.extra_cells_y), 0),
    )


def _pad_optional_field_for_solver(
    field: jnp.ndarray | None,
    config: BrainFWIConfig,
) -> jnp.ndarray | None:
    """Embed an optional medium field in the solver-domain halo."""

    if field is None:
        return None
    pad_x, pad_y = _solver_padding(config)
    return jnp.pad(field, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge")


def _pad_model_for_solver(
    velocity: jnp.ndarray,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Embed the physical model in the larger solver domain.

    Stride solves the wave equation on an extended grid where the physical model
    sits away from the absorbing frame. We mimic that by edge-padding the
    inversion model before stepping the wave equation.
    """

    pad_x, pad_y = _solver_padding(config)
    return jnp.pad(velocity, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge")


def _buoyancy_field(density: jnp.ndarray | None) -> jnp.ndarray | None:
    """Convert density to buoyancy while guarding against division by zero."""

    if density is None:
        return None
    return 1.0 / jnp.maximum(density, 1.0e-8)


def _shift_indices_for_solver(
    indices: jnp.ndarray, config: BrainFWIConfig
) -> jnp.ndarray:
    """Shift grid indices from the physical model into the padded solver grid."""

    pad_x, pad_y = _solver_padding(config)
    shift = jnp.asarray((pad_x, pad_y), dtype=indices.dtype)
    return indices + shift


def _solver_acquisition_views(
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    dtype,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    bool,
]:
    """Return acquisition arrays expressed in solver-domain coordinates."""

    receivers, _, wavelet = acquisition.require_solver_arrays()
    solver_receivers = _shift_indices_for_solver(receivers, config)
    use_hicks = acquisition.interpolation_type == "hicks"
    if use_hicks:
        source_reference_gridpoints = _shift_indices_for_solver(
            acquisition.source_reference_gridpoints,
            config,
        )
        receiver_reference_gridpoints = _shift_indices_for_solver(
            acquisition.receiver_reference_gridpoints,
            config,
        )
        source_coefficients = acquisition.source_coefficients
        receiver_coefficients = acquisition.receiver_coefficients
    else:
        source_reference_gridpoints = solver_receivers
        receiver_reference_gridpoints = solver_receivers
        source_coefficients = jnp.zeros((receivers.shape[0], 2, 8), dtype=dtype)
        receiver_coefficients = jnp.zeros((receivers.shape[0], 2, 8), dtype=dtype)
    return (
        solver_receivers,
        source_reference_gridpoints,
        source_coefficients,
        receiver_reference_gridpoints,
        receiver_coefficients,
        wavelet,
        use_hicks,
    )


def _build_damping_mask(config: BrainFWIConfig, shape: tuple[int, int]) -> jnp.ndarray:
    """Create a simple absorbing frame to reduce edge reflections.

    This is a pragmatic substitute for a full absorbing boundary model. The mask
    smoothly damps wave amplitudes near the edges where artificial reflections
    would otherwise contaminate the inversion objective.
    """

    nx, ny = shape
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

    nx = int(velocity.shape[0])
    ny = int(velocity.shape[1])
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
        damping = _build_damping_mask(config, velocity.shape).astype(velocity.dtype)
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


def _spatial_operator_ot2(
    field: jnp.ndarray,
    velocity_sq: jnp.ndarray,
    density: jnp.ndarray | None,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Evaluate the second-order-in-time spatial operator."""

    if density is None:
        return velocity_sq * _laplacian(
            field,
            config.grid.dx,
            config.grid.dy,
            config.solver.space_order,
        )

    buoyancy = _buoyancy_field(density)
    grad_x = _first_derivative_1d(
        field,
        config.grid.dx,
        axis=0,
        space_order=config.solver.space_order,
    )
    grad_y = _first_derivative_1d(
        field,
        config.grid.dy,
        axis=1,
        space_order=config.solver.space_order,
    )
    div_x = _first_derivative_1d(
        buoyancy * grad_x,
        config.grid.dx,
        axis=0,
        space_order=config.solver.space_order,
    )
    div_y = _first_derivative_1d(
        buoyancy * grad_y,
        config.grid.dy,
        axis=1,
        space_order=config.solver.space_order,
    )
    return velocity_sq * density * (div_x + div_y)


def _spatial_operator(
    field: jnp.ndarray,
    velocity_sq: jnp.ndarray,
    density: jnp.ndarray | None,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Evaluate the discrete spatial operator used by the current kernel.

    For constant density this reduces to `vp**2 * Lap(u)`. When density is
    provided we approximate Stride's `vp**2 * rho * div(buoy * grad(u))`
    operator on the collocated JAX grid. For `OT4`, we mirror the
    `IsoAcousticDevito` correction term by applying the same discrete operator
    twice.
    """

    operator_2 = _spatial_operator_ot2(field, velocity_sq, density, config)

    if config.solver.kernel == "OT2":
        return operator_2
    if config.solver.kernel != "OT4":
        raise ValueError(f"Unsupported solver kernel '{config.solver.kernel}'.")

    operator_4 = _spatial_operator_ot2(operator_2, velocity_sq, density, config)
    return operator_2 + ((config.time.dt**2) / 12.0) * operator_4


def _attenuation_update(
    u_prev: jnp.ndarray,
    u_curr: jnp.ndarray,
    velocity: jnp.ndarray,
    attenuation: jnp.ndarray | None,
    config: BrainFWIConfig,
) -> jnp.ndarray:
    """Return the explicit attenuation contribution for one time step.

    Stride supports attenuation powers `0` and `2`. We mirror those options in
    the JAX update while keeping the fields fixed. The returned tensor is the
    additive contribution that should be combined with the usual pressure and
    source updates.
    """

    if attenuation is None:
        return jnp.zeros_like(u_curr)

    power = int(config.model.attenuation_power)
    if power == 0:
        quantity_prev = u_prev
        quantity_curr = u_curr
    elif power == 2:
        quantity_prev = -_laplacian(
            u_prev,
            config.grid.dx,
            config.grid.dy,
            config.solver.space_order,
        )
        quantity_curr = -_laplacian(
            u_curr,
            config.grid.dx,
            config.grid.dy,
            config.solver.space_order,
        )
    else:
        raise ValueError(f"Unsupported attenuation_power '{power}'. Use 0 or 2.")

    coeff = 2.0 * attenuation * (velocity ** (power + 1))
    return -config.time.dt * coeff * (quantity_curr - quantity_prev)


def _propose_next_field(
    u_prev: jnp.ndarray,
    u_curr: jnp.ndarray,
    velocity: jnp.ndarray,
    density: jnp.ndarray | None,
    attenuation: jnp.ndarray | None,
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
        u_curr, velocity_sq, density, config
    )
    attenuation_update = _attenuation_update(
        u_prev,
        u_curr,
        velocity,
        attenuation,
        config,
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
            + attenuation_update
            + source_update
            - (sponge_damp**2) * u_curr
        )
        return boundary_mask * (numerator / (1.0 + sponge_damp))

    return boundary_mask * (
        2.0 * u_curr - u_prev + pressure_update + attenuation_update + source_update
    )


def _advance_state(
    u_prev: jnp.ndarray,
    u_curr: jnp.ndarray,
    velocity: jnp.ndarray,
    density: jnp.ndarray | None,
    attenuation: jnp.ndarray | None,
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
        density,
        attenuation,
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
    density: jnp.ndarray | None,
    attenuation: jnp.ndarray | None,
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
        density,
        attenuation,
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
    medium: AcousticMedium | None,
    shot_index: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one shot while storing sparse checkpoints for later recomputation.

    Instead of keeping every wavefield in memory, we save only the `(u_{n-1},
    u_n)` pair at the start of each checkpoint segment and keep the full trace
    history. The adjoint later replays each segment on demand, which trades
    extra compute for a much smaller peak memory footprint.
    """

    padded_velocity = _pad_model_for_solver(velocity, config)
    padded_density = _pad_optional_field_for_solver(
        None if medium is None else medium.density,
        config,
    )
    padded_attenuation = _pad_optional_field_for_solver(
        None if medium is None else medium.attenuation,
        config,
    )
    (
        solver_receivers,
        solver_source_reference_gridpoints,
        solver_source_coefficients,
        solver_receiver_reference_gridpoints,
        solver_receiver_coefficients,
        wavelet,
        use_hicks,
    ) = _solver_acquisition_views(acquisition, config, padded_velocity.dtype)
    source_index = solver_receivers[shot_index]
    source_reference_gridpoint = solver_source_reference_gridpoints[shot_index]
    source_coefficients = solver_source_coefficients[shot_index]
    receiver_i = solver_receivers[:, 0]
    receiver_j = solver_receivers[:, 1]
    boundary_mask, sponge_damp = _build_boundary_terms(config, padded_velocity)
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
                solver_receiver_reference_gridpoints,
                solver_receiver_coefficients,
                padded_velocity,
                padded_density,
                padded_attenuation,
                boundary_mask,
                sponge_damp,
                use_hicks,
                config,
            ),
            carry,
            (source_block, active_block),
        )
        return carry, (segment_start_prev, segment_start_curr, segment_traces)

    init = (jnp.zeros_like(padded_velocity), jnp.zeros_like(padded_velocity))
    _, (checkpoint_prevs, checkpoint_currs, segment_traces) = jax.lax.scan(
        run_segment,
        init,
        (wavelet_segments, active_segments),
    )
    traces = segment_traces.reshape((-1, solver_receivers.shape[0]))[: wavelet.shape[0]]
    return traces, checkpoint_prevs, checkpoint_currs, wavelet_segments, active_segments


def _simulate_shot_forward_only(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    medium: AcousticMedium | None,
    shot_index: jnp.ndarray,
) -> jnp.ndarray:
    """Run one shot without storing any checkpoint replay tensors.

    This path is intended for forward-only use cases such as:
    - metrics after optimisation
    - diagnostic residual plots
    - synthetic observation generation

    It keeps only the current two wavefields plus the output traces, which is
    materially cheaper than the checkpointed adjoint path for memory-heavy runs.
    """

    padded_velocity = _pad_model_for_solver(velocity, config)
    padded_density = _pad_optional_field_for_solver(
        None if medium is None else medium.density,
        config,
    )
    padded_attenuation = _pad_optional_field_for_solver(
        None if medium is None else medium.attenuation,
        config,
    )
    (
        solver_receivers,
        solver_source_reference_gridpoints,
        solver_source_coefficients,
        solver_receiver_reference_gridpoints,
        solver_receiver_coefficients,
        wavelet,
        use_hicks,
    ) = _solver_acquisition_views(acquisition, config, padded_velocity.dtype)
    source_index = solver_receivers[shot_index]
    source_reference_gridpoint = solver_source_reference_gridpoints[shot_index]
    source_coefficients = solver_source_coefficients[shot_index]
    receiver_i = solver_receivers[:, 0]
    receiver_j = solver_receivers[:, 1]
    boundary_mask, sponge_damp = _build_boundary_terms(config, padded_velocity)
    source_wavelet = _prepare_source_wavelet(wavelet, config)

    # Keep an explicit active mask so this scan remains robust even if future
    # wavelet construction introduces padded/no-op samples.
    active_mask = jnp.arange(source_wavelet.shape[0]) < wavelet.shape[0]

    def step_forward_only(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        xs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Advance one time step and record receiver samples."""

        u_prev, u_curr = carry
        source_value, active = xs
        next_prev, u_next = _advance_state(
            u_prev,
            u_curr,
            padded_velocity,
            padded_density,
            padded_attenuation,
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
                solver_receiver_reference_gridpoints,
                solver_receiver_coefficients,
                use_hicks,
            ),
            0.0,
        )
        return (next_prev, u_next), traces

    init = (jnp.zeros_like(padded_velocity), jnp.zeros_like(padded_velocity))
    _, traces = jax.lax.scan(step_forward_only, init, (source_wavelet, active_mask))
    return traces


def _normalise_shot_batch_size(
    total_shots: int,
    shot_batch_size: int | None,
    *,
    argument_name: str,
) -> int:
    """Resolve and validate a generic shot-batch-size argument."""

    if shot_batch_size is None:
        return 1

    batch_size = int(shot_batch_size)
    if batch_size <= 0:
        raise ValueError(f"{argument_name} must be a positive integer.")
    return min(batch_size, max(total_shots, 1))


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
    density: jnp.ndarray | None,
    attenuation: jnp.ndarray | None,
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
            density,
            attenuation,
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
    medium: AcousticMedium | None,
    shot_index: jnp.ndarray,
) -> jnp.ndarray:
    """Simulate one transmit event and record traces at every transducer.

    The returned tensor has shape `[time, receiver]`. Internally we carry the
    previous and current wavefields so the second-order time update can generate
    the next wavefield.
    """

    traces, _, _, _, _ = _simulate_shot_with_checkpoints(
        velocity, acquisition, config, medium, shot_index
    )
    return traces


def simulate_survey_forward_only(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    medium: AcousticMedium | None = None,
    shot_indices: jnp.ndarray | None = None,
    shot_batch_size: int | None = None,
) -> jnp.ndarray:
    """Simulate a survey using the no-checkpoint forward-only path.

    `shot_batch_size` controls how many shots are vmapped together in one
    compile/execution unit. A value of `1` runs fully sequentially with the
    lowest memory footprint; larger values can improve throughput when memory
    headroom exists.
    """

    active_shot_indices = (
        acquisition.require_solver_arrays()[1] if shot_indices is None else shot_indices
    )
    total_shots = int(active_shot_indices.shape[0])
    batch_size = _normalise_shot_batch_size(
        total_shots,
        shot_batch_size,
        argument_name="shot_batch_size",
    )

    if total_shots == 0:
        # Keep shape parity with the regular survey path even for an empty shot
        # selection.
        return jnp.zeros(
            (0, config.time.nt, acquisition.n_receivers),
            dtype=velocity.dtype,
        )

    if batch_size == 1:
        _, traces = jax.lax.scan(
            lambda carry, shot_idx: (
                carry,
                _simulate_shot_forward_only(
                    velocity,
                    acquisition,
                    config,
                    medium,
                    shot_idx,
                ),
            ),
            None,
            active_shot_indices,
        )
        return traces

    batched_outputs = []
    for start in range(0, total_shots, batch_size):
        stop = min(start + batch_size, total_shots)
        shot_batch = active_shot_indices[start:stop]
        batch_traces = jax.vmap(
            lambda shot_idx: _simulate_shot_forward_only(
                velocity,
                acquisition,
                config,
                medium,
                shot_idx,
            )
        )(shot_batch)
        batched_outputs.append(batch_traces)
    return jnp.concatenate(batched_outputs, axis=0)


def simulate_survey(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    medium: AcousticMedium | None = None,
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
        lambda shot_idx: simulate_shot(velocity, acquisition, config, medium, shot_idx)
    )(active_shot_indices)


def loss_and_grad(
    velocity: jnp.ndarray,
    acquisition: AcquisitionGeometry,
    config: BrainFWIConfig,
    medium: AcousticMedium | None,
    observed_data: jnp.ndarray,
    f_max_hz: float | None = None,
    shot_indices: jnp.ndarray | None = None,
    shot_batch_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the survey loss and explicit adjoint gradient in JAX.

    The gradient is produced by an explicit reverse-time adjoint
    implementation. The data misfit itself now mirrors the tracked Stride
    benchmark more closely by combining a Stride-style `0.5 * sum(r^2)` loss
    with an optional `f_max` low-pass filter applied in the trace domain.
    """

    dt = config.time.dt
    (
        receivers,
        acquisition_shot_indices,
        _,
    ) = acquisition.require_solver_arrays()
    active_shot_indices = (
        acquisition_shot_indices if shot_indices is None else shot_indices
    )
    total_shots = int(active_shot_indices.shape[0])
    grad_batch_size = _normalise_shot_batch_size(
        total_shots,
        shot_batch_size,
        argument_name="shot_batch_size",
    )
    padded_velocity = _pad_model_for_solver(velocity, config)
    padded_density = _pad_optional_field_for_solver(
        None if medium is None else medium.density,
        config,
    )
    padded_attenuation = _pad_optional_field_for_solver(
        None if medium is None else medium.attenuation,
        config,
    )
    (
        solver_receivers,
        solver_source_reference_gridpoints,
        solver_source_coefficients,
        solver_receiver_reference_gridpoints,
        solver_receiver_coefficients,
        _,
        use_hicks,
    ) = _solver_acquisition_views(acquisition, config, padded_velocity.dtype)
    receiver_i = solver_receivers[:, 0]
    receiver_j = solver_receivers[:, 1]
    boundary_mask, sponge_damp = _build_boundary_terms(config, padded_velocity)

    def shot_loss_grad(shot_index: jnp.ndarray, observed_shot: jnp.ndarray):
        """Run one forward/adjoint pair and return its loss contribution."""

        source_index = solver_receivers[shot_index]
        source_reference_gridpoint = solver_source_reference_gridpoints[shot_index]
        source_coefficients = solver_source_coefficients[shot_index]
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
            medium,
            shot_index,
        )
        residual = traces - observed_shot

        # The FFT mask defines a linear zero-phase operator. Applying it to the
        # residual gives us the band-limited misfit used for the current stage,
        # and because the filter is symmetric in time the same filtered residual
        # also acts as the trace-domain cotangent for the adjoint.
        residual = bandlimit_traces(
            residual,
            dt,
            f_max_hz,
            axis=0,
            filter_type=config.solver.trace_filter_type,
            relaxation=config.solver.trace_filter_relaxation,
            order=config.solver.trace_filter_order,
            zero_phase=config.solver.trace_filter_zero_phase,
        )
        shot_loss = 0.5 * jnp.sum(residual**2)
        data_cotangents = bandlimit_traces(
            residual,
            dt,
            f_max_hz,
            axis=0,
            filter_type=config.solver.trace_filter_type,
            relaxation=config.solver.trace_filter_relaxation,
            order=config.solver.trace_filter_order,
            zero_phase=config.solver.trace_filter_zero_phase,
            adjoint=not config.solver.trace_filter_zero_phase,
        )
        padded_data_cotangents = _pad_traces_to_segments(
            data_cotangents,
            wavelet_segments.shape[0],
            wavelet_segments.shape[1],
        )

        def segment_transition(
            segment_start_prev: jnp.ndarray,
            segment_start_curr: jnp.ndarray,
            segment_velocity: jnp.ndarray,
            source_block: jnp.ndarray,
            active_block: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Advance one full checkpoint segment and return segment outputs.

            This fuses the reverse-time sensitivity into one VJP call per
            segment instead of one VJP per time step. The discrete forward
            logic is identical to `_step_shot_state`, so gradients remain
            aligned with the exact solver update while reducing adjoint overhead.
            """

            padded_segment_velocity = _pad_model_for_solver(segment_velocity, config)

            def segment_step(
                carry: tuple[jnp.ndarray, jnp.ndarray],
                step_xs: tuple[jnp.ndarray, jnp.ndarray],
            ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
                source_value, active = step_xs
                next_prev, next_curr = _advance_state(
                    carry[0],
                    carry[1],
                    padded_segment_velocity,
                    padded_density,
                    padded_attenuation,
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
                        next_curr,
                        receiver_i,
                        receiver_j,
                        solver_receiver_reference_gridpoints,
                        solver_receiver_coefficients,
                        use_hicks,
                    ),
                    0.0,
                )
                return (next_prev, next_curr), traces

            (
                segment_end_prev,
                segment_end_curr,
            ), segment_traces = jax.lax.scan(
                segment_step,
                (segment_start_prev, segment_start_curr),
                (source_block, active_block),
            )
            return segment_end_prev, segment_end_curr, segment_traces

        def reverse_segment(carry, xs):
            """Run one fused segment-level VJP for reverse propagation."""

            cotangent_end_prev, cotangent_end_curr, grad_velocity = carry
            checkpoint_prev, checkpoint_curr, source_block, active_block, data_block = (
                xs
            )

            # Inactive padded samples are no-ops in the forward segment scan.
            # Zeroing their cotangents keeps the transposed trace injection
            # exactly aligned with that behaviour.
            active_trace_mask = active_block[:, None].astype(data_block.dtype)
            segment_data_cotangent = data_block * active_trace_mask

            (
                _segment_end_prev,
                _segment_end_curr,
                _segment_traces,
            ), segment_vjp = jax.vjp(
                lambda step_prev, step_curr, step_velocity: segment_transition(
                    step_prev,
                    step_curr,
                    step_velocity,
                    source_block,
                    active_block,
                ),
                checkpoint_prev,
                checkpoint_curr,
                velocity,
            )
            (
                cotangent_start_prev,
                cotangent_start_curr,
                velocity_cotangent,
            ) = segment_vjp(
                (
                    cotangent_end_prev,
                    cotangent_end_curr,
                    segment_data_cotangent,
                )
            )
            return (
                (
                    cotangent_start_prev,
                    cotangent_start_curr,
                    grad_velocity + velocity_cotangent,
                ),
                None,
            )

        init_carry = (
            jnp.zeros_like(padded_velocity),
            jnp.zeros_like(padded_velocity),
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

    if total_shots == 0:
        return jnp.array(0.0, dtype=velocity.dtype), jnp.zeros_like(velocity)

    if grad_batch_size == 1:
        # Sequential accumulation keeps only one shot's forward/adjoint history
        # live at a time, which is the most memory-conservative mode.
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

    # Batched shot accumulation improves throughput by evaluating several
    # independent shot forward+adjoint solves together, while masking the tail
    # of the final partial batch so objective and gradient remain unchanged.
    n_batches = (total_shots + grad_batch_size - 1) // grad_batch_size
    padded_size = n_batches * grad_batch_size
    pad = padded_size - total_shots

    batched_shot_indices = jnp.pad(active_shot_indices, (0, pad)).reshape(
        n_batches, grad_batch_size
    )
    batched_observed = jnp.pad(
        observed_data,
        ((0, pad), (0, 0), (0, 0)),
    ).reshape(
        n_batches,
        grad_batch_size,
        observed_data.shape[1],
        observed_data.shape[2],
    )
    batched_active_mask = jnp.pad(
        jnp.ones((total_shots,), dtype=jnp.bool_),
        (0, pad),
        constant_values=False,
    ).reshape(n_batches, grad_batch_size)

    def accumulate_shot_batch(carry, xs):
        total_loss, total_grad = carry
        shot_batch, observed_batch, active_mask = xs
        batch_losses, batch_grads = jax.vmap(shot_loss_grad)(shot_batch, observed_batch)

        active_loss = active_mask.astype(batch_losses.dtype)
        active_grad = active_mask.astype(batch_grads.dtype)[:, None, None]
        return (
            total_loss + jnp.sum(batch_losses * active_loss),
            total_grad + jnp.sum(batch_grads * active_grad, axis=0),
        ), None

    init = (jnp.array(0.0, dtype=velocity.dtype), jnp.zeros_like(velocity))
    (total_loss, total_grad), _ = jax.lax.scan(
        accumulate_shot_batch,
        init,
        (batched_shot_indices, batched_observed, batched_active_mask),
    )
    return total_loss, total_grad
