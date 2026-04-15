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
    transducer_indices: jnp.ndarray | None = None
    shot_indices: jnp.ndarray | None = None
    source_wavelet: jnp.ndarray | None = None
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

    return AcquisitionGeometry(
        geometry_type="elliptical",
        n_transducers=acq.n_transducers,
        n_shots=int(shot_indices.shape[0]),
        n_time_samples=config.time.nt,
        transducer_indices=indices,
        shot_indices=shot_indices,
        source_wavelet=_tone_burst_wavelet(
            config.time.nt,
            config.time.dt,
            acq.source_frequency_hz,
            acq.source_cycles,
        )
        * acq.source_amplitude,
        metadata=(
            ("ellipse_scale_x", acq.ellipse_scale_x),
            ("ellipse_scale_y", acq.ellipse_scale_y),
            ("source_frequency_hz", acq.source_frequency_hz),
            ("source_cycles", acq.source_cycles),
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
        metadata=tuple(sorted(reference_settings.items())),
    )
