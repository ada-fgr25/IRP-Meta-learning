"""Frequency-domain utilities shared by the JAX and analysis workflows.

This module keeps two closely related pieces of functionality together:

* NumPy spectrum helpers for analysis and benchmark comparison
* a JAX-native trace band-limiter used by the inversion loss schedule

The NumPy helpers mirror the Stride-style FFT utilities closely so repository
code can inspect spectra with the same conventions discussed during review.
The JAX helper stays separate from those analysis functions because the
optimisation path needs a differentiable implementation.
"""

from __future__ import annotations

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


def bandlimit_traces(
    traces: jnp.ndarray,
    dt: float,
    f_max_hz: float | None,
    axis: int = 1,
) -> jnp.ndarray:
    """Apply a hard low-pass filter to trace data along the time axis.

    The tracked Stride benchmark uses an `f_max` schedule to grow the usable
    bandwidth over three inversion blocks. We mirror that idea by zeroing all
    FFT bins above the current stage cutoff and transforming back to the time
    domain. The operator is linear and zero-phase, which keeps the resulting
    loss easy to reason about.
    """

    if f_max_hz is None or not np.isfinite(f_max_hz):
        return traces

    num = traces.shape[axis]
    freqs = jnp.fft.rfftfreq(num, dt)
    mask = (freqs <= float(f_max_hz)).astype(traces.dtype)

    # Move the time axis to the end so one FFT call handles arbitrary leading
    # batch dimensions such as `[shot, receiver]` cleanly.
    moved = jnp.moveaxis(traces, axis, -1)
    spectrum = jnp.fft.rfft(moved, axis=-1)
    filtered = jnp.fft.irfft(spectrum * mask, n=num, axis=-1)
    return jnp.moveaxis(filtered, -1, axis)
