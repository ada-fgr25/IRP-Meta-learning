"""Regression tests for Stride-like trace filtering helpers."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from fwi.filtering import bandlimit_traces


def _reference_lowpass_filter_cos(
    data: np.ndarray,
    *,
    dt: float,
    f_max_hz: float,
    relaxation: float,
    order: int,
    zero_phase: bool,
    adjoint: bool,
) -> np.ndarray:
    """NumPy reference for Stride's cosine low-pass implementation."""

    f_max_dimless = f_max_hz * dt / relaxation
    normalized_fmax = f_max_dimless / 0.5
    period = max(int(np.round(1.0 / normalized_fmax)), 1)
    filter_length = 2 * period + 1

    table = np.zeros((filter_length,), dtype=np.float32)
    for i in range(1, filter_length + 1):
        table[i - 1] = 1.0 - np.cos(2.0 * np.pi * i / (filter_length + 1))
    table /= np.sum(table)

    traces = np.array(data, dtype=np.float32, copy=True)
    if adjoint:
        traces = np.flip(traces, axis=-1)

    if not zero_phase:
        traces = np.pad(traces, ((0, 0), (period, 0)), mode="constant")

    filtered = traces
    for _ in range(order):
        next_filtered = np.zeros_like(filtered)
        half_width = filter_length // 2
        for row in range(filtered.shape[0]):
            for col in range(filtered.shape[1]):
                value = 0.0
                for tap in range(filter_length):
                    src = col + tap - half_width
                    if 0 <= src < filtered.shape[1]:
                        value += float(table[tap]) * float(filtered[row, src])
                next_filtered[row, col] = value
        filtered = next_filtered

    if not zero_phase:
        filtered = filtered[:, : data.shape[-1]]

    if adjoint:
        filtered = np.flip(filtered, axis=-1)

    return filtered


class FilteringTests(unittest.TestCase):
    """Pin down the Stride-like continuation filter behaviour."""

    def test_cosine_lowpass_matches_reference_stride_shape(self):
        """The JAX cosine filter should match a small NumPy reference."""

        traces = np.array(
            [
                [0.0, 1.0, 0.0, -1.0, 0.5, 0.0],
                [1.0, -0.5, 0.25, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        kwargs = {
            "dt": 1.0e-6,
            "f_max_hz": 1.5e5,
            "relaxation": 0.75,
            "order": 1,
            "zero_phase": False,
            "adjoint": False,
        }

        expected = _reference_lowpass_filter_cos(traces, **kwargs)
        actual = np.asarray(
            bandlimit_traces(
                jnp.asarray(traces),
                kwargs["dt"],
                kwargs["f_max_hz"],
                axis=1,
                filter_type="cos",
                relaxation=kwargs["relaxation"],
                order=kwargs["order"],
                zero_phase=kwargs["zero_phase"],
                adjoint=kwargs["adjoint"],
            )
        )

        self.assertTrue(np.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6))

    def test_cosine_lowpass_adjoint_satisfies_inner_product_identity(self):
        """The explicit adjoint flag should implement the filter transpose."""

        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.standard_normal((3, 12), dtype=np.float32))
        y = jnp.asarray(rng.standard_normal((3, 12), dtype=np.float32))
        kwargs = {
            "axis": 1,
            "filter_type": "cos",
            "relaxation": 0.75,
            "order": 1,
            "zero_phase": False,
        }
        dt = 8.0e-8
        f_max_hz = 2.0e5

        fx = bandlimit_traces(x, dt, f_max_hz, **kwargs)
        fty = bandlimit_traces(
            y,
            dt,
            f_max_hz,
            **kwargs,
            adjoint=True,
        )

        lhs = float(jnp.vdot(fx, y))
        rhs = float(jnp.vdot(x, fty))
        self.assertAlmostEqual(lhs, rhs, places=5)


if __name__ == "__main__":
    unittest.main()
