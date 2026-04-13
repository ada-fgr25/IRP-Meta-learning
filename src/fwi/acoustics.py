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
            2.0 * u_curr - u_prev + (dt**2) * ((velocity**2) * lap_curr + source_term)
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

    traces, _, _ = _simulate_shot_with_history(
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

        traces, curr_fields, laplacians = _simulate_shot_with_history(
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

        def reverse_step(carry, xs):
            """Reverse one time step of the discrete wave equation."""

            cotangent_curr, cotangent_next, grad_velocity = carry
            curr_field, laplacian_curr, data_cotangent = xs

            # The observation operator samples the next wavefield at receiver
            # points, so its adjoint scatters those trace-domain cotangents back
            # onto the full grid before the wave-equation reverse step.
            cotangent_next = cotangent_next + _inject_receivers(
                receiver_i,
                receiver_j,
                data_cotangent,
                grid_shape,
            )

            # The boundary operator is just a diagonal mask, so it is its own
            # adjoint. Applying it here mirrors the boundary treatment used in
            # the forward step.
            masked_cotangent = boundary_mask * cotangent_next

            # The gradient with respect to velocity comes from differentiating
            # `velocity**2 * laplacian(u_n)` pointwise at each time step.
            grad_velocity = (
                grad_velocity
                + 2.0 * velocity * (dt**2) * masked_cotangent * laplacian_curr
            )

            # Reverse propagation through the explicit second-order update:
            # - `u_{n-1}` receives `-masked_cotangent`
            # - `u_n` receives the direct `2 * masked_cotangent` term
            # - plus the adjoint of the spatial operator
            cotangent_prev = -masked_cotangent
            cotangent_curr = (
                cotangent_curr
                + 2.0 * masked_cotangent
                + (dt**2) * (_laplacian(velocity_sq * masked_cotangent, dx, dy))
            )
            return (cotangent_prev, cotangent_curr, grad_velocity), None

        init_carry = (
            jnp.zeros_like(velocity),
            jnp.zeros_like(velocity),
            jnp.zeros_like(velocity),
        )
        (cotangent_initial_prev, cotangent_initial_curr, shot_grad), _ = jax.lax.scan(
            reverse_step,
            init_carry,
            (curr_fields, laplacians, data_cotangents),
            reverse=True,
        )

        # The initial wavefields are fixed zeros rather than optimisation
        # variables, so their cotangents are intentionally ignored.
        del cotangent_initial_prev, cotangent_initial_curr
        return shot_loss, shot_grad

    shot_losses, shot_grads = jax.vmap(shot_loss_grad)(active_shot_indices, observed_data)
    return jnp.sum(shot_losses), jnp.sum(shot_grads, axis=0)
