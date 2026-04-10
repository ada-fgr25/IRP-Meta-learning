"""Configuration objects for the JAX FWI baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:
    """Spatial discretisation for the 2D acoustic model."""

    nx: int = 96
    ny: int = 72
    dx: float = 5.0e-4
    dy: float = 5.0e-4


@dataclass(frozen=True)
class TimeConfig:
    """Temporal discretisation for the wave solver."""

    dt: float = 1.0e-7
    nt: int = 320


@dataclass(frozen=True)
class AcquisitionConfig:
    """Stride-inspired elliptical transducer geometry."""

    n_transducers: int = 24
    n_shots: int = 6
    ellipse_scale_x: float = 0.90
    ellipse_scale_y: float = 0.85
    source_frequency_hz: float = 2.5e5
    source_amplitude: float = 1.0e12


@dataclass(frozen=True)
class ModelConfig:
    """Velocity bounds and phantom controls for the inversion model."""

    background_velocity: float = 1500.0
    brain_velocity: float = 1560.0
    skull_velocity: float = 2400.0
    lesion_velocity: float = 1650.0
    min_velocity: float = 1450.0
    max_velocity: float = 3000.0


@dataclass(frozen=True)
class SolverConfig:
    """Numerical stabilisation controls for the explicit solver."""

    damping_cells: int = 10
    damping_strength: float = 0.015


@dataclass(frozen=True)
class BrainFWIConfig:
    """Complete configuration for the Phase 1 JAX baseline."""

    grid: GridConfig = GridConfig()
    time: TimeConfig = TimeConfig()
    acquisition: AcquisitionConfig = AcquisitionConfig()
    model: ModelConfig = ModelConfig()
    solver: SolverConfig = SolverConfig()
