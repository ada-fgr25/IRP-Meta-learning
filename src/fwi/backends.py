"""Backend selection for JAX and Stride-style FWI workflows."""

from __future__ import annotations

from dataclasses import dataclass

from .acquisition import (
    AcquisitionGeometry,
    build_elliptical_acquisition,
    build_stride_acquisition,
)
from .acoustics import loss_and_grad, simulate_survey_forward_only
from .stride_benchmark import StrideBenchmarkRunner


@dataclass(frozen=True)
class JaxBackend:
    """Pure JAX backend used by the Phase 1 differentiable baseline."""

    name: str = "jax"

    def build_acquisition(self, config) -> AcquisitionGeometry:
        """Construct the shared acquisition object from the Python config."""

        return build_elliptical_acquisition(config)

    def forward(self, velocity, acquisition, config, medium=None, shot_indices=None):
        return simulate_survey_forward_only(
            velocity,
            acquisition,
            config,
            medium=medium,
            shot_indices=shot_indices,
            shot_batch_size=config.solver.forward_shot_batch_size,
        )

    def loss_grad(self, params, x, auxs):
        """Return the explicit adjoint-state loss gradient in pure JAX."""

        value, grad = loss_and_grad(
            x,
            params["acquisition"],
            params["config"],
            params.get("medium"),
            auxs[0],
            auxs[1] if len(auxs) > 1 else None,
            auxs[2] if len(auxs) > 2 else None,
            auxs[3] if len(auxs) > 3 else None,
        )
        return value.reshape((1,)), grad


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

    def forward(self, velocity, geometry, config, medium=None, shot_indices=None):
        del shot_indices, medium
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

    def forward(self, velocity, acquisition, config, medium=None, shot_indices=None):
        """The Stride benchmark path does not provide an in-process forward op."""

        del velocity, acquisition, config, medium, shot_indices
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
