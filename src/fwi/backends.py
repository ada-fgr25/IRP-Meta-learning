"""Backend selection for JAX and Stride-style FWI workflows."""

from __future__ import annotations

from dataclasses import dataclass

from .acquisition import (
    AcquisitionGeometry,
    build_elliptical_acquisition,
    build_stride_acquisition,
)
from .acoustics import simulate_survey
from .stride_benchmark import StrideBenchmarkRunner


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


@dataclass(frozen=True)
class StrideBackend:
    """Shared backend wrapper for the tracked Stride benchmark workflow.

    This backend intentionally exposes the same high-level acquisition surface
    as the JAX path, even though the benchmark itself is still launched via the
    tracked reference scripts instead of a differentiable Python solver.
    """

    name: str = "stride"
    runner: StrideBenchmarkRunner = StrideBenchmarkRunner()

    def build_acquisition(self, config) -> AcquisitionGeometry:
        """Describe the benchmark acquisition through the shared API."""

        del config
        return build_stride_acquisition(self.runner.reference_settings())

    def forward(self, velocity, acquisition, config):
        """The Stride benchmark path does not provide an in-process forward op."""

        del velocity, acquisition, config
        raise NotImplementedError(
            "The Stride backend is benchmark-only in this repository. Use the "
            "tracked scripts through `StrideBenchmarkRunner` rather than an "
            "in-process differentiable forward operator."
        )


def build_backend(name: str = "jax"):
    """Return the requested FWI backend."""

    if name == "jax":
        return JaxBackend()
    if name == "devito":
        return DevitoBackend()
    if name == "stride":
        return StrideBackend()
    raise ValueError(f"Unknown backend '{name}'.")
