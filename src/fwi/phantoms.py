"""Synthetic brain-like velocity models used for FWI experiments."""

from __future__ import annotations

import jax.numpy as jnp

from .config import BrainFWIConfig


def _normalised_coordinates(nx: int, ny: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return coordinates scaled to [-1, 1] for procedural phantom creation."""

    x = jnp.linspace(-1.0, 1.0, nx)
    y = jnp.linspace(-1.0, 1.0, ny)
    return jnp.meshgrid(x, y, indexing="ij")


def _ellipse_mask(
    xx: jnp.ndarray,
    yy: jnp.ndarray,
    centre_x: float,
    centre_y: float,
    radius_x: float,
    radius_y: float,
) -> jnp.ndarray:
    """Return a soft binary mask for an ellipse embedded in the grid."""

    return (
        ((xx - centre_x) / radius_x) ** 2 + ((yy - centre_y) / radius_y) ** 2
        <= 1.0
    )


def build_true_brain_velocity(config: BrainFWIConfig) -> jnp.ndarray:
    """Create a simple brain phantom with skull, ventricles, and a lesion."""

    nx = config.grid.nx
    ny = config.grid.ny
    model_cfg = config.model
    xx, yy = _normalised_coordinates(nx, ny)

    # Start from water-like coupling medium outside the skull.
    velocity = jnp.full((nx, ny), model_cfg.background_velocity)

    # Add a thin, high-velocity skull ring.
    skull_outer = _ellipse_mask(xx, yy, 0.0, 0.0, 0.72, 0.88)
    skull_inner = _ellipse_mask(xx, yy, 0.0, 0.0, 0.64, 0.80)
    skull = skull_outer & ~skull_inner
    velocity = jnp.where(skull, model_cfg.skull_velocity, velocity)

    # Fill the interior with softer brain tissue.
    brain = _ellipse_mask(xx, yy, 0.0, 0.0, 0.62, 0.78)
    velocity = jnp.where(brain, model_cfg.brain_velocity, velocity)

    # Add lower-velocity ventricles and a slightly faster focal lesion.
    vent_left = _ellipse_mask(xx, yy, -0.12, 0.05, 0.08, 0.12)
    vent_right = _ellipse_mask(xx, yy, 0.12, 0.05, 0.08, 0.12)
    lesion = _ellipse_mask(xx, yy, 0.18, -0.22, 0.12, 0.10)
    velocity = jnp.where(vent_left | vent_right, model_cfg.brain_velocity - 60.0, velocity)
    velocity = jnp.where(lesion, model_cfg.lesion_velocity, velocity)

    return velocity


def build_initial_velocity(config: BrainFWIConfig) -> jnp.ndarray:
    """Create a smoothed starting model that deliberately misses fine detail."""

    nx = config.grid.nx
    ny = config.grid.ny
    model_cfg = config.model
    xx, yy = _normalised_coordinates(nx, ny)

    velocity = jnp.full((nx, ny), model_cfg.background_velocity)
    skull_outer = _ellipse_mask(xx, yy, 0.0, 0.0, 0.72, 0.88)
    skull_inner = _ellipse_mask(xx, yy, 0.0, 0.0, 0.60, 0.76)
    skull = skull_outer & ~skull_inner
    brain = _ellipse_mask(xx, yy, 0.0, 0.0, 0.58, 0.74)

    # The initial model only captures coarse anatomy, which makes the inverse
    # problem meaningfully non-trivial while still being close enough to converge.
    velocity = jnp.where(skull, 2100.0, velocity)
    velocity = jnp.where(brain, 1540.0, velocity)

    return velocity
