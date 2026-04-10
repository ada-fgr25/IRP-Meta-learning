"""Configuration objects for the JAX FWI baseline.

These dataclasses deliberately separate the problem into grid, time,
acquisition, model, and solver concerns. That mirrors how FWI setups are
normally described in practice and makes later experiment sweeps easier to
reason about.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:
    """Spatial discretisation for the 2D acoustic model.

    `nx` and `ny` are the number of grid cells.
    `dx` and `dy` are the physical spacing between cells in metres.
    """

    nx: int = 96
    ny: int = 72
    dx: float = 5.0e-4
    dy: float = 5.0e-4


@dataclass(frozen=True)
class TimeConfig:
    """Temporal discretisation for the wave solver.

    `dt` is the time step in seconds and `nt` is the total number of recorded
    time samples. Together they control both solver stability and the recording
    window available to capture arrivals.
    """

    dt: float = 1.0e-7
    nt: int = 320


@dataclass(frozen=True)
class AcquisitionConfig:
    """Stride-inspired elliptical transducer geometry.

    The transducers sit on an ellipse around the head to mimic the broad shape
    of the Stride brain-ultrasound example. `n_shots` selects a subset of those
    transducers as emitters. We now default to a denser survey than the first
    prototype because sparse illumination made the internal anomalies almost
    invisible to the optimiser.
    """

    n_transducers: int = 48
    n_shots: int = 24
    ellipse_scale_x: float = 0.90
    ellipse_scale_y: float = 0.85
    source_frequency_hz: float = 2.5e5
    source_amplitude: float = 1.0e12


@dataclass(frozen=True)
class ModelConfig:
    """Velocity bounds and phantom controls for the inversion model.

    `source` selects whether the experiment uses the lightweight procedural
    phantom or the tracked Stride HDF5 velocity models in `data/`.

    The four velocity values define the synthetic anatomy used by the
    procedural fallback: coupling medium, soft brain tissue, skull, and a
    lesion-like inclusion. `min_velocity` and `max_velocity` are inversion-time
    box constraints regardless of source.
    """

    source: str = "stride"
    true_model_path: str = "data/alpha2D-TrueModel.h5"
    starting_model_path: str = "data/alpha2D-StartingModel.h5"
    stride_downsample: int = 1
    background_velocity: float = 1500.0
    brain_velocity: float = 1560.0
    skull_velocity: float = 2400.0
    lesion_velocity: float = 1650.0
    min_velocity: float = 1450.0
    max_velocity: float = 3000.0


@dataclass(frozen=True)
class SolverConfig:
    """Numerical stabilisation controls for the explicit solver.

    The damping frame is a lightweight absorbing boundary condition used to
    suppress edge reflections without introducing a more elaborate PML.
    """

    damping_cells: int = 10
    damping_strength: float = 0.015


@dataclass(frozen=True)
class BrainFWIConfig:
    """Complete configuration for the Phase 1 JAX baseline.

    This top-level object is what we pass around in experiments so a whole FWI
    setup can be reproduced from one serialisable configuration tree.
    """

    grid: GridConfig = GridConfig()
    time: TimeConfig = TimeConfig()
    acquisition: AcquisitionConfig = AcquisitionConfig()
    model: ModelConfig = ModelConfig()
    solver: SolverConfig = SolverConfig()
