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

    nx: int = 500
    ny: int = 370
    dx: float = 5.0e-4
    dy: float = 5.0e-4


@dataclass(frozen=True)
class TimeConfig:
    """Temporal discretisation for the wave solver.

    `dt` is the time step in seconds and `nt` is the total number of recorded
    time samples. Together they control both solver stability and the recording
    window available to capture arrivals.
    """

    dt: float = 8.0e-8
    nt: int = 2500


@dataclass(frozen=True)
class AcquisitionConfig:
    """Stride-inspired elliptical transducer geometry.

    The transducers sit on an ellipse around the head to mimic the broad shape
    of the Stride brain-ultrasound example. By default the JAX path now adopts
    the same `256` transducer ring as the tracked Stride scripts and uses the
    full ring as the available shot pool. Per-iteration random shot subsets are
    selected later by the optimiser loop rather than being baked into this
    static acquisition description.
    """

    n_transducers: int = 256
    n_shots: int = 256
    ellipse_scale_x: float = 0.90
    ellipse_scale_y: float = 0.85
    source_frequency_hz: float = 2.5e5
    source_cycles: int = 3
    source_amplitude: float = 1.0


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
    `kernel` selects the time-stepping family used by the JAX solver. `OT4`
    mirrors Stride's default more closely by adding the standard fourth-order
    temporal correction on top of the spatial operator, while `OT2` keeps the
    simpler second-order update available for debugging or ablations.
    `source_scale_mode` controls how the source sample is converted into a
    pressure update. The default `stride` mode follows the Stride
    `IsoAcousticDevito` scaling `2 * dt**2 * vp / max(dx, dy)` and, unless
    `diff_source` is enabled, divides once more by `dt` exactly as the Devito
    implementation does.
    `diff_source` mirrors Stride's optional behaviour of injecting the first
    time derivative of the source wavelet instead of the raw wavelet.
    `stride_grad_processing` toggles a JAX approximation of Stride's default
    `ProcessGlobalGradient` pipeline before each optimiser update.
    `mask_grad`, `smooth_grad`, and `norm_grad` mirror Stride's default
    processing switches. `grad_smooth_radius` controls the spatial radius of
    the smoothing kernel when `smooth_grad` is enabled.
    `checkpoint_interval` controls how many time steps of forward history are
    recomputed at once during the explicit adjoint. Smaller values reduce peak
    memory at the cost of more recomputation.
    """

    damping_cells: int = 40
    damping_strength: float = 0.015
    kernel: str = "OT4"
    source_scale_mode: str = "stride"
    diff_source: bool = False
    stride_grad_processing: bool = True
    mask_grad: bool = True
    smooth_grad: bool = True
    norm_grad: bool = True
    grad_smooth_radius: int = 2
    checkpoint_interval: int = 32


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
