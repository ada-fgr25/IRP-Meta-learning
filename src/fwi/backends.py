"""Backend selection for FWI forward operators."""

from __future__ import annotations

from dataclasses import dataclass

from .acquisition import AcquisitionGeometry, build_elliptical_acquisition
from .acoustics import simulate_survey


@dataclass(frozen=True)
class JaxBackend:
    """Pure JAX backend used by the Phase 1 differentiable baseline."""

    name: str = "jax"

    def build_acquisition(self, config) -> AcquisitionGeometry:
        """Construct the shared acquisition object from the Python config."""

        return build_elliptical_acquisition(config)

    def forward(self, velocity, acquisition, config):
        return simulate_survey(velocity, acquisition, config)


@dataclass(frozen=True)
class DevitoBackend:
    """Placeholder for a future Devito-backed custom-VJP implementation."""

    name: str = "devito"

    def __post_init__(self) -> None:
        try:
            import devito  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Devito is not installed. Use the default JAX backend for the "
                "Phase 1 baseline, or install the optional `devito` extra."
            ) from exc

    def forward(self, velocity, geometry, config):
        raise NotImplementedError(
            "A Devito-backed differentiable wrapper has not been implemented yet. "
            "The current baseline uses a JAX-native solver to keep the full "
            "forward and adjoint pipeline differentiable."
        )


def build_backend(name: str = "jax"):
    """Return the requested FWI backend."""

    if name == "jax":
        return JaxBackend()
    if name == "devito":
        return DevitoBackend()
    raise ValueError(f"Unknown backend '{name}'.")
