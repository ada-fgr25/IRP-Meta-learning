"""Shared acquisition objects used by both the JAX and Stride workflows.

The differentiable JAX solver needs explicit transducer coordinates, shot
indices, and a source wavelet. The Stride benchmark path, by contrast, mostly
needs a compact description of the acquisition that the tracked reference
scripts implement. This module gives both paths one common acquisition object
so experiment code can inspect either backend through the same surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.special

from .config import BrainFWIConfig


@dataclass(frozen=True)
class AcquisitionGeometry:
    """Describe an acquisition in a backend-agnostic way.

    The JAX backend populates the solver arrays directly. The Stride benchmark
    populates the counts and metadata while leaving the solver arrays unset,
    because those are encoded inside the tracked Stride scripts rather than
    reconstructed in Python here.
    """

    geometry_type: str
    n_transducers: int
    n_shots: int
    n_time_samples: int
    interpolation_type: str = "linear"
    transducer_coordinates: jnp.ndarray | None = None
    transducer_indices: jnp.ndarray | None = None
    shot_indices: jnp.ndarray | None = None
    source_wavelet: jnp.ndarray | None = None
    source_reference_gridpoints: jnp.ndarray | None = None
    source_coefficients: jnp.ndarray | None = None
    receiver_reference_gridpoints: jnp.ndarray | None = None
    receiver_coefficients: jnp.ndarray | None = None
    metadata: tuple[tuple[str, Any], ...] = ()

    @property
    def n_receivers(self) -> int:
        """Return the number of receivers sampled for each shot."""

        return self.n_transducers

    def require_solver_arrays(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the JAX solver arrays or fail with a backend-focused message."""

        if (
            self.transducer_indices is None
            or self.shot_indices is None
            or self.source_wavelet is None
        ):
            raise ValueError(
                "This acquisition does not carry the explicit arrays required by "
                "the JAX solver."
            )
        return self.transducer_indices, self.shot_indices, self.source_wavelet

    def metadata_dict(self) -> dict[str, Any]:
        """Materialise the immutable metadata pairs as a regular dictionary."""

        return dict(self.metadata)


def _tone_burst_wavelet(
    nt: int,
    dt: float,
    frequency_hz: float,
    n_cycles: int,
) -> jnp.ndarray:
    """Generate a finite-length tone burst closer to the tracked Stride source.

    The Stride benchmark scripts drive each transducer with a `3`-cycle tone
    burst centred at `0.25 MHz`. We mirror that source family here so the JAX
    baseline starts from a waveform that is much closer to the reference
    acquisition before any inversion-time filtering is applied.
    """

    t = jnp.arange(nt) * dt
    burst_duration = n_cycles / frequency_hz

    # The squared-sine taper keeps the pulse compact and symmetric while still
    # being easy to express with JAX primitives. Samples after the active burst
    # window are set to zero so later frequencies come only from propagation.
    phase = jnp.pi * jnp.clip(t / burst_duration, 0.0, 1.0)
    envelope = jnp.sin(phase) ** 2
    carrier = jnp.sin(2.0 * jnp.pi * frequency_hz * t)
    active = (t <= burst_duration).astype(t.dtype)
    return active * envelope * carrier


def _select_shot_indices(n_transducers: int, n_shots: int) -> jnp.ndarray:
    """Choose an evenly spaced, unique subset of transmitters."""

    if n_transducers <= 0:
        raise ValueError("`n_transducers` must be positive.")
    if n_shots <= 0:
        raise ValueError("`n_shots` must be positive.")

    capped_n_shots = min(n_shots, n_transducers)
    shot_positions = (
        jnp.arange(capped_n_shots, dtype=jnp.float32) * n_transducers / capped_n_shots
    )
    return jnp.floor(shot_positions).astype(jnp.int32)


def _calculate_hicks(
    coordinates: np.ndarray,
    *,
    spacing: tuple[float, float],
    origin: tuple[float, float],
    smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate Stride's Hicks coefficient construction for 2D points.

    This mirrors the reference implementation's Kaiser-windowed sinc setup:
    - `kaiser_b = 4.14`
    - half-width `3`, giving support over offsets `[-3, 3]`
    - one extra coefficient slot (`r + 1`) kept for parity with the Devito
      precomputed sparse-function API used by Stride.
    """

    grid_coordinates = (coordinates - np.asarray(origin)) / np.asarray(spacing)
    reference_gridpoints = np.floor(grid_coordinates).astype(np.int32)
    offsets = grid_coordinates - reference_gridpoints

    kaiser_b = 4.14
    kaiser_half_width = 3
    kaiser_den = scipy.special.iv(0, kaiser_b)
    kaiser_extended_width = kaiser_half_width / 0.99

    r = 2 * kaiser_half_width + 1
    num = coordinates.shape[0]
    dim = coordinates.shape[1]
    coefficients = np.zeros((num, dim, r + 1), dtype=np.float32)

    for grid_point in range(-kaiser_half_width, kaiser_half_width + 1):
        index = kaiser_half_width + grid_point
        x = offsets - grid_point
        weights = (x / kaiser_extended_width) ** 2
        weights = np.minimum(weights, 1.0)
        b_weights = scipy.special.iv(0, kaiser_b * np.sqrt(1.0 - weights)) / kaiser_den
        coefficients[:, :, index] = np.sinc(x) * b_weights

    # Stride applies a small optional smoothing tweak for source interpolation.
    if smooth:
        n = kaiser_half_width - 1
        a = coefficients[:, :, n]
        b = coefficients[:, :, n + 1]
        c = coefficients[:, :, n + 2]
        coefficients[:, :, n - 1] = coefficients[:, :, n - 1] + a * 0.01
        coefficients[:, :, n] = a * 0.98 + b * 0.03
        coefficients[:, :, n + 1] = b * 0.94 + (a + c) * 0.01
        coefficients[:, :, n + 2] = c * 0.98 + b * 0.03
        coefficients[:, :, n + 3] = coefficients[:, :, n + 3] + c * 0.01

    return reference_gridpoints, coefficients


def _grid_to_physical_coordinates(
    grid_coordinates: jnp.ndarray,
    *,
    spacing: tuple[float, float],
    origin: tuple[float, float],
    epsilon_scale: float = 0.0,
) -> np.ndarray:
    """Convert grid-index coordinates into physical coordinates.

    `build_elliptical_acquisition` parameterises transducer positions in grid
    index units because that is convenient for plotting and nearest-index
    interpolation. The Hicks helper, however, expects physical coordinates and
    internally maps them back to grid coordinates via `(x - origin) / spacing`.

    If we pass index-space values directly into Hicks while still supplying
    metric spacing, we effectively divide by spacing twice and produce extremely
    large reference indices. Converting to physical coordinates here keeps the
    two coordinate systems consistent.
    """

    spacing_array = np.asarray(spacing, dtype=np.float32)
    origin_array = np.asarray(origin, dtype=np.float32)
    physical_coordinates = (
        np.asarray(grid_coordinates, dtype=np.float32) * spacing_array + origin_array
    )
    if epsilon_scale == 0.0:
        return physical_coordinates

    # Stride applies a tiny spacing-proportional perturbation before sparse
    # interpolation setup (`eps_coords = 1e-3 * spacing`). Keeping this as an
    # explicit coordinate shift improves parity with the benchmark path while
    # remaining deterministic and differentiability-friendly in the JAX setup.
    epsilon = float(epsilon_scale) * spacing_array
    return physical_coordinates + epsilon


def build_elliptical_acquisition(config: BrainFWIConfig) -> AcquisitionGeometry:
    """Create the Stride-inspired elliptical ring used by the JAX solver."""

    grid = config.grid
    acq = config.acquisition
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, acq.n_transducers, endpoint=False)
    centre = jnp.array([(grid.nx - 1) / 2.0, (grid.ny - 1) / 2.0])
    radius = jnp.array(
        [
            acq.ellipse_scale_x * (grid.nx - 1) / 2.0,
            acq.ellipse_scale_y * (grid.ny - 1) / 2.0,
        ]
    )
    coords = jnp.stack(
        [
            centre[0] + radius[0] * jnp.cos(angles),
            centre[1] + radius[1] * jnp.sin(angles),
        ],
        axis=-1,
    )
    indices = jnp.rint(coords).astype(jnp.int32)
    shot_indices = _select_shot_indices(acq.n_transducers, acq.n_shots)

    source_reference_gridpoints = None
    source_coefficients = None
    receiver_reference_gridpoints = None
    receiver_coefficients = None

    if acq.interpolation_type == "hicks":
        # Keep the reference implementation's assumptions:
        # - physical origin at (0, 0)
        # - same coordinates used for sources and receivers
        spacing = (float(grid.dx), float(grid.dy))
        origin = (0.0, 0.0)
        coordinates_np = _grid_to_physical_coordinates(
            coords,
            spacing=spacing,
            origin=origin,
            epsilon_scale=(
                acq.coordinate_epsilon_scale if acq.apply_coordinate_epsilon else 0.0
            ),
        )

        src_ref, src_coeff = _calculate_hicks(
            coordinates_np,
            spacing=spacing,
            origin=origin,
            smooth=True,
        )
        rec_ref, rec_coeff = _calculate_hicks(
            coordinates_np,
            spacing=spacing,
            origin=origin,
            smooth=False,
        )
        source_reference_gridpoints = jnp.asarray(src_ref)
        source_coefficients = jnp.asarray(src_coeff)
        receiver_reference_gridpoints = jnp.asarray(rec_ref)
        receiver_coefficients = jnp.asarray(rec_coeff)
    elif acq.interpolation_type != "linear":
        raise ValueError(
            "Unsupported interpolation_type "
            f"'{acq.interpolation_type}'. Use 'linear' or 'hicks'."
        )

    return AcquisitionGeometry(
        geometry_type="elliptical",
        n_transducers=acq.n_transducers,
        n_shots=int(shot_indices.shape[0]),
        n_time_samples=config.time.nt,
        interpolation_type=acq.interpolation_type,
        transducer_coordinates=coords,
        transducer_indices=indices,
        shot_indices=shot_indices,
        source_wavelet=_tone_burst_wavelet(
            config.time.nt,
            config.time.dt,
            acq.source_frequency_hz,
            acq.source_cycles,
        )
        * acq.source_amplitude,
        source_reference_gridpoints=source_reference_gridpoints,
        source_coefficients=source_coefficients,
        receiver_reference_gridpoints=receiver_reference_gridpoints,
        receiver_coefficients=receiver_coefficients,
        metadata=(
            ("ellipse_scale_x", acq.ellipse_scale_x),
            ("ellipse_scale_y", acq.ellipse_scale_y),
            ("interpolation_type", acq.interpolation_type),
            ("source_frequency_hz", acq.source_frequency_hz),
            ("source_cycles", acq.source_cycles),
            ("coordinate_epsilon_scale", acq.coordinate_epsilon_scale),
            ("apply_coordinate_epsilon", acq.apply_coordinate_epsilon),
            # Keeping the amplitude explicit in metadata makes it easier to
            # audit source-normalisation changes when we compare against the
            # bundled Stride scripts, which do not add a large extra factor.
            ("source_amplitude", acq.source_amplitude),
        ),
    )


def build_stride_acquisition(reference_settings: dict[str, Any]) -> AcquisitionGeometry:
    """Convert the benchmark script settings into the shared acquisition object."""

    return AcquisitionGeometry(
        geometry_type=str(reference_settings["geometry"]),
        n_transducers=int(reference_settings["num_locations"]),
        n_shots=int(reference_settings["num_locations"]),
        n_time_samples=int(reference_settings["time_num"]),
        interpolation_type=str(reference_settings.get("interpolation_type", "linear")),
        metadata=tuple(sorted(reference_settings.items())),
    )
