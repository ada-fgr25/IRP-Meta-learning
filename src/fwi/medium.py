"""Helpers for optional density and attenuation medium fields.

The repository still optimises velocity only, but Stride's acoustic operator
also supports fixed density/buoyancy and attenuation fields. This module keeps
the field-construction logic separate from the wave solver so experiments can
enable those physics terms without tangling geometry loading with PDE code.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .config import BrainFWIConfig


@dataclass(frozen=True)
class AcousticMedium:
    """Fixed medium properties used alongside the velocity model.

    The current JAX inversion still treats these as known fields rather than
    optimisation variables. That keeps the explicit adjoint focused on the
    velocity gradient while still allowing the forward physics to include the
    same extra terms that Stride can model.
    """

    density: jnp.ndarray | None = None
    attenuation: jnp.ndarray | None = None


def _nearest_material_index(
    velocity: jnp.ndarray, config: BrainFWIConfig
) -> jnp.ndarray:
    """Classify each cell by the nearest configured anatomical velocity."""

    model = config.model
    references = jnp.asarray(
        (
            model.background_velocity,
            model.brain_velocity,
            model.skull_velocity,
            model.lesion_velocity,
        ),
        dtype=velocity.dtype,
    )
    return jnp.argmin(jnp.abs(velocity[..., None] - references), axis=-1)


def _piecewise_field(
    velocity: jnp.ndarray,
    config: BrainFWIConfig,
    values: tuple[float, float, float, float],
) -> jnp.ndarray:
    """Map each velocity cell to a material property by nearest class."""

    indices = _nearest_material_index(velocity, config)
    table = jnp.asarray(values, dtype=velocity.dtype)
    return table[indices]


def build_acoustic_medium(
    config: BrainFWIConfig,
    velocity: jnp.ndarray,
) -> AcousticMedium:
    """Build the fixed density and attenuation fields requested by the config."""

    model = config.model

    if model.density_model == "none":
        density = None
    elif model.density_model == "homogeneous":
        density = jnp.full_like(velocity, model.background_density)
    elif model.density_model == "piecewise":
        density = _piecewise_field(
            velocity,
            config,
            (
                model.background_density,
                model.brain_density,
                model.skull_density,
                model.lesion_density,
            ),
        )
    else:
        raise ValueError(
            f"Unsupported density_model '{model.density_model}'. Use 'none', "
            "'homogeneous', or 'piecewise'."
        )

    if model.attenuation_model == "none":
        attenuation = None
    elif model.attenuation_model == "homogeneous":
        attenuation = jnp.full_like(velocity, model.background_attenuation)
    elif model.attenuation_model == "piecewise":
        attenuation = _piecewise_field(
            velocity,
            config,
            (
                model.background_attenuation,
                model.brain_attenuation,
                model.skull_attenuation,
                model.lesion_attenuation,
            ),
        )
    else:
        raise ValueError(
            f"Unsupported attenuation_model '{model.attenuation_model}'. Use 'none', "
            "'homogeneous', or 'piecewise'."
        )

    return AcousticMedium(density=density, attenuation=attenuation)
