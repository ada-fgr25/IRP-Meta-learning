"""Frequency-domain utilities shared by the JAX and analysis workflows.

This module keeps two closely related pieces of functionality together:

* NumPy spectrum helpers for analysis and benchmark comparison
* JAX-native trace filters used by the inversion continuation schedule

The NumPy helpers mirror the Stride-style FFT utilities closely so repository
code can inspect spectra with the same conventions discussed during review.
The JAX helpers stay separate from those analysis functions because the
optimisation path needs differentiable implementations that also expose the
adjoint action required by the explicit reverse-time gradient code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def magnitude_spectrum(signal, dt, db: bool = True):
    """Calculate the one-sided magnitude spectrum of a signal or signal batch."""

    num = signal.shape[-1]
    freqs = np.fft.rfftfreq(num, dt)
    signal_fft = np.fft.rfft(signal, axis=-1, norm="forward")
    signal_magnitude = np.abs(signal_fft)

    if db:
        signal_magnitude = 20 * np.log10(
            (signal_magnitude + 1e-31) / (np.max(signal_magnitude) + 1e-31)
        )

    return freqs, signal_magnitude


def phase_spectrum(signal, dt):
    """Calculate the one-sided phase spectrum of a signal or signal batch."""

    num = signal.shape[-1]
    freqs = np.fft.rfftfreq(num, dt)
    signal_fft = np.fft.rfft(signal, axis=-1)
    signal_phase = np.angle(signal_fft)
    return freqs, signal_phase


def bandwidth(signal, dt, cutoff: float = -10):
    """Calculate the bandwidth of a signal at a given decibel cutoff."""

    freqs, signal_fft = magnitude_spectrum(signal, dt, db=True)

    if len(signal_fft.shape) > 1:
        signal_fft = np.mean(signal_fft, axis=0)

    num_freqs = signal_fft.shape[-1]

    f_min = 0.0
    for f in range(num_freqs):
        if signal_fft[f] > cutoff:
            f_min = float(freqs[f])
            break

    f_centre = float(freqs[int(np.argmax(signal_fft))])

    f_max = float(freqs[-1])
    for f in reversed(range(num_freqs)):
        if signal_fft[f] > cutoff:
            f_max = float(freqs[f])
            break

    return f_min, f_centre, f_max


def _cosine_lowpass_kernel(period: int, dtype) -> jnp.ndarray:
    """Build Stride's cosine low-pass kernel for one continuation stage.

    Stride's default `f_max` continuation path uses `lowpass_filter_cos` rather
    than a hard spectral cutoff. Recreating the same kernel shape here makes the
    JAX loss schedule much closer to the reference workflow while still keeping
    the implementation JIT-friendly.
    """

    filter_length = 2 * period + 1
    taps = jnp.arange(1, filter_length + 1, dtype=dtype)
    kernel = 1.0 - jnp.cos((2.0 * jnp.pi * taps) / (filter_length + 1))
    return kernel / jnp.sum(kernel)


def _convolve_traces_same(traces: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Apply one zero-padded 1D convolution pass along the last axis."""

    batch, num = traces.shape
    lhs = traces.reshape(batch, num, 1)
    rhs = kernel[:, None, None]
    filtered = jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    return filtered[..., 0]


def _cosine_lowpass_filter(
    traces: jnp.ndarray,
    dt: float,
    f_max_hz: float,
    relaxation: float,
    order: int,
    zero_phase: bool,
    adjoint: bool,
) -> jnp.ndarray:
    """Approximate Stride's cosine low-pass filter in JAX.

    The Stride pipeline passes `f_max * dt / relaxation` into
    `lowpass_filter_cos`, which then normalises by Nyquist internally. We keep
    the same dimensionless conversion so JAX and Stride use comparable
    continuation bandwidths. The explicit `adjoint` flag matters because the
    non-zero-phase filter is not self-adjoint.
    """

    if not np.isfinite(f_max_hz):
        return traces

    safe_relaxation = max(float(relaxation), 1.0e-6)
    dimless_fmax = float(f_max_hz) * float(dt) / safe_relaxation
    normalized_fmax = dimless_fmax / 0.5
    if normalized_fmax <= 0.0:
        return jnp.zeros_like(traces)

    period = max(int(np.round(1.0 / normalized_fmax)), 1)
    kernel = _cosine_lowpass_kernel(period, traces.dtype)

    flattened = traces.reshape((-1, traces.shape[-1]))
    if adjoint:
        flattened = jnp.flip(flattened, axis=-1)

    if not zero_phase:
        flattened = jnp.pad(flattened, ((0, 0), (period, 0)))

    filtered = flattened
    for _ in range(max(int(order), 1)):
        filtered = _convolve_traces_same(filtered, kernel)

    if not zero_phase:
        filtered = filtered[:, : traces.shape[-1]]

    if adjoint:
        filtered = jnp.flip(filtered, axis=-1)

    return filtered.reshape(traces.shape)


def bandlimit_traces(
    traces: jnp.ndarray,
    dt: float,
    f_max_hz: float | None,
    axis: int = 1,
    *,
    filter_type: str = "cos",
    relaxation: float = 0.75,
    order: int = 1,
    zero_phase: bool = False,
    adjoint: bool = False,
) -> jnp.ndarray:
    """Apply the configured low-pass continuation filter along the time axis.

    Stride's inversion pipeline does not use a hard FFT mask for `f_max`
    continuation. Its default low-pass path uses a cosine filter with a
    relaxation factor, and the reverse-time gradient needs the adjoint of that
    filter when the pass is not zero phase. We therefore expose both the
    forward and adjoint actions here instead of assuming a symmetric operator.
    """

    if f_max_hz is None or not np.isfinite(f_max_hz):
        return traces

    # Move the time axis to the end so one FFT call handles arbitrary leading
    # batch dimensions such as `[shot, receiver]` cleanly.
    moved = jnp.moveaxis(traces, axis, -1)
    if filter_type == "cos":
        filtered = _cosine_lowpass_filter(
            moved,
            dt,
            float(f_max_hz),
            relaxation,
            order,
            zero_phase,
            adjoint,
        )
        return jnp.moveaxis(filtered, -1, axis)

    if filter_type == "fft":
        num = moved.shape[-1]
        freqs = jnp.fft.rfftfreq(num, dt)
        mask = (freqs <= float(f_max_hz)).astype(moved.dtype)
        spectrum = jnp.fft.rfft(moved, axis=-1)
        filtered = jnp.fft.irfft(spectrum * mask, n=num, axis=-1)
        return jnp.moveaxis(filtered, -1, axis)

    raise ValueError(
        f"Unsupported trace filter_type '{filter_type}'. Use 'cos' or 'fft'."
    )
