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

    `interpolation_type` mirrors Stride's receiver/source interpolation switch:
    - `linear`: simple gridpoint interpolation baseline
    - `hicks`: sinc/Kaiser-style precomputed interpolation coefficients
    `coordinate_epsilon_scale` mirrors Stride's small coordinate perturbation
    before sparse-function setup. The offset is applied as
    `coordinate_epsilon_scale * spacing` per spatial dimension in physical
    coordinates and helps avoid edge-aligned interpolation degeneracies.
    `apply_coordinate_epsilon` keeps this behaviour configurable for ablations.
    """

    n_transducers: int = 256
    n_shots: int = 256
    ellipse_scale_x: float = 0.90
    ellipse_scale_y: float = 0.85
    source_frequency_hz: float = 2.5e5
    source_cycles: int = 3
    source_amplitude: float = 1.0
    interpolation_type: str = "linear"
    coordinate_epsilon_scale: float = 1.0e-3
    apply_coordinate_epsilon: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Velocity bounds and medium-property controls for the inversion model.

    `source` selects whether the experiment uses the lightweight procedural
    phantom or the tracked Stride HDF5 velocity models in `data/`.

    The four velocity values define the synthetic anatomy used by the
    procedural fallback: coupling medium, soft brain tissue, skull, and a
    lesion-like inclusion. `min_velocity` and `max_velocity` are inversion-time
    box constraints regardless of source.

    `density_model` and `attenuation_model` enable optional fixed medium fields
    used by the JAX wave solver in addition to the velocity model. The `piecewise`
    mode maps each cell to the nearest configured anatomical velocity class,
    which lets both procedural and Stride-loaded velocity models reuse the same
    density/attenuation lookup without requiring extra HDF5 inputs.
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
    density_model: str = "none"
    background_density: float = 1000.0
    brain_density: float = 1040.0
    skull_density: float = 1900.0
    lesion_density: float = 1080.0
    attenuation_model: str = "none"
    attenuation_power: int = 0
    background_attenuation: float = 0.0
    brain_attenuation: float = 0.15
    skull_attenuation: float = 2.5
    lesion_attenuation: float = 0.25


@dataclass(frozen=True)
class SolverConfig:
    """Numerical stabilisation controls for the explicit solver.

    The damping frame is a lightweight absorbing boundary condition used to
    suppress edge reflections without introducing a more elaborate PML.
    `damping_mode` selects between:
    - `legacy`: simple quadratic taper mask
    - `stride_like`: Stride-inspired damping field mapped to a multiplicative mask
    - `sponge2`: Stride-inspired second-order sponge damping update
    `damping_type` controls the functional profile used by the Stride-inspired
    mode (`sine` or `power`). `damping_power_degree` is used when
    `damping_type='power'`.
    `damping_reflection_coefficient` mirrors Stride's coefficient used to
    derive a physically motivated damping scale from the absorbing width. When
    omitted (`None`), we follow Stride's default width-dependent heuristic.
    `damping_max_coefficient` can override that derived scale. If it is `None`,
    the coefficient is derived from absorbing width and spacing.
    `damping_velocity_scale` mirrors Stride's optional velocity scaling by using
    the local model velocity when deriving the damping field.
    `extra_cells_x` and `extra_cells_y` add a fixed halo around the inversion
    model before the damping frame is applied. This mirrors Stride's use of an
    extended solver domain, where the absorbing boundary sits outside the
    physical model rather than directly on its edge.
    `space_order` sets the order of the central finite-difference stencil used
    for the spatial derivatives. The default `10` mirrors Stride's Devito
    configuration much more closely than the previous second-order Laplacian.
    `kernel` selects the time-stepping family used by the JAX solver. `OT4`
    mirrors Stride's default more closely by adding the standard fourth-order
    temporal correction on top of the spatial operator, while `OT2` keeps the
    simpler second-order update available for debugging or ablations.
    `source_scale_mode` controls how the source sample is converted into a
    pressure update. The default `stride` mode follows the Stride
    `IsoAcousticDevito` scaling `2 * dt**2 * vp / max(dx, dy)` and, unless
    `diff_source` is enabled, divides once more by `dt` exactly as the Devito
    implementation does.
    `source_window_enabled` applies a Stride-like Tukey source window before
    injection. `source_window_alpha` is the Tukey taper parameter used by
    Stride. `source_window_start` and `source_window_stop` mirror Stride's
    `time_bounds` behavior by defining the active source interval.
    `diff_source` mirrors Stride's optional behaviour of injecting the first
    time derivative of the source wavelet instead of the raw wavelet.
    `stride_grad_processing` toggles a JAX approximation of Stride's default
    `ProcessGlobalGradient` pipeline before each optimiser update.
    `mask_grad`, `smooth_grad`, and `norm_grad` mirror Stride's default
    processing switches. `grad_smooth_sigma` mirrors Stride's Gaussian
    `smooth_sigma` (default `0.25` cells) used by `SmoothField`.
    `grad_smooth_radius` is retained as a legacy fallback for runs that still
    use the previous box-filter approximation.
    `grad_mask_rampoff` mirrors Stride's `MaskField(mask_rampoff=10)` soft edge
    taper used during gradient masking.
    `grad_norm_guess_change` mirrors Stride's `norm_guess_change` used by
    `NormField`: after max-amplitude normalisation, gradients are rescaled by
    `mid_model * grad_norm_guess_change / 100`. This is an important part of
    Stride's default gradient magnitude calibration.
    `grad_global_norm` mirrors Stride's `global_norm` knob. The current JAX
    implementation keeps parity with the default (`False`) and uses per-step
    normalisation values.
    `trace_filter_type`, `trace_filter_order`, and `trace_filter_zero_phase`
    control the trace-domain `f_max` continuation filter.
    `trace_filter_relaxation_wavelets` mirrors Stride's
    `filter_wavelets_relaxation` used by `ProcessWavelets`/`ProcessObserved`.
    `trace_filter_relaxation_traces` mirrors Stride's
    `filter_traces_relaxation` used by `ProcessTraces`.
    `trace_filter_relaxation` is retained as a compatibility fallback and
    defaults to the same `0.75` continuation value.
    `fw3d_mode` toggles Stride's quarter-period trace shift used in the
    benchmark scripts. `stride_trace_processing` enables a Stride-like
    pre-misfit trace-conditioning path (`ProcessObserved` + `ProcessTraces`)
    inside the JAX loss, including mute/filter/norm parity while staying fully
    differentiable.
    `stride_trace_filter_wavelets` controls Stride-like filtering in
    `ProcessWavelets`/`ProcessObserved` before forward modelling.
    `stride_trace_filter_traces` controls Stride-like filtering in
    `ProcessTraces` before the L2 loss.
    `stride_trace_mute_traces`, `stride_trace_norm_per_shot`, and
    `stride_trace_scale_per_shot` mirror the corresponding optional
    `ProcessTraces` steps. The default benchmark path keeps mute+norm enabled
    and scale disabled.
    `stride_trace_time_weighting` adds an optional differentiable time-weight
    stage in the misfit path to emulate Stride's optional `time_weighting`
    pipeline hook. Weighting is controlled by
    `stride_trace_time_weight_power`, `stride_trace_time_weight_start`, and
    `stride_trace_time_weight_stop`.
    `checkpoint_interval` controls how many time steps of forward history are
    recomputed at once during the explicit adjoint. Smaller values reduce peak
    memory at the cost of more recomputation.
    `forward_shot_batch_size` controls shot-level batching for forward-only
    surveys (for example diagnostics and final metrics). `1` is the most
    memory-conservative setting; larger values trade memory for throughput.
    `grad_shot_batch_size` controls shot-level batching for the forward+adjoint
    gradient path. `1` keeps the most conservative sequential accumulation.
    Higher values can improve throughput when memory headroom allows.
    `shot_reduction` controls how multi-shot objectives are reduced before the
    optimiser update:
    - `sum`: Stride-style additive objective (`0.5 * sum(r^2)` per shot, then
      summed across selected shots)
    - `mean`: average over selected shots; useful for experiments where update
      magnitude should be less sensitive to shot-count changes.
    """

    damping_cells: int = 40
    damping_strength: float = 0.015
    damping_mode: str = "stride_like"
    damping_type: str = "sine"
    damping_power_degree: int = 2
    damping_reflection_coefficient: float | None = None
    damping_max_coefficient: float | None = None
    damping_velocity_scale: bool = True
    extra_cells_x: int = 50
    extra_cells_y: int = 50
    space_order: int = 10
    kernel: str = "OT4"
    source_scale_mode: str = "stride"
    source_window_enabled: bool = True
    source_window_alpha: float = 1.0e-3
    source_window_start: int = 0
    source_window_stop: int | None = None
    diff_source: bool = False
    stride_grad_processing: bool = True
    mask_grad: bool = True
    grad_mask_rampoff: int = 10
    smooth_grad: bool = True
    norm_grad: bool = True
    grad_smooth_sigma: float = 0.25
    grad_smooth_radius: int = 2
    grad_norm_guess_change: float = 0.5
    grad_global_norm: bool = False
    trace_filter_type: str = "cos"
    trace_filter_relaxation: float = 0.75
    trace_filter_relaxation_wavelets: float = 0.75
    trace_filter_relaxation_traces: float = 0.75
    trace_filter_order: int = 1
    trace_filter_zero_phase: bool = False
    fw3d_mode: bool = True
    stride_trace_processing: bool = True
    stride_trace_filter_wavelets: bool = True
    stride_trace_filter_traces: bool = True
    stride_trace_mute_traces: bool = True
    stride_trace_norm_per_shot: bool = True
    stride_trace_scale_per_shot: bool = False
    stride_trace_time_weighting: bool = False
    stride_trace_time_weight_power: float = 1.0
    stride_trace_time_weight_start: int = 0
    stride_trace_time_weight_stop: int | None = None
    checkpoint_interval: int = 32
    forward_shot_batch_size: int = 1
    grad_shot_batch_size: int = 1
    shot_reduction: str = "sum"


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
