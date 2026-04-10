"""Evaluation metrics for baseline FWI experiments."""

from __future__ import annotations

import jax.numpy as jnp


def compute_metrics(
    predicted_velocity: jnp.ndarray,
    true_velocity: jnp.ndarray,
    predicted_data: jnp.ndarray,
    observed_data: jnp.ndarray,
) -> dict[str, float]:
    """Compute a compact set of scalar metrics for experiment tracking."""

    model_residual = predicted_velocity - true_velocity
    data_residual = predicted_data - observed_data
    denom = jnp.linalg.norm(true_velocity) + 1.0e-8

    return {
        "model_rmse": float(jnp.sqrt(jnp.mean(model_residual**2))),
        "model_relative_l2": float(jnp.linalg.norm(model_residual) / denom),
        "data_rmse": float(jnp.sqrt(jnp.mean(data_residual**2))),
        "data_mae": float(jnp.mean(jnp.abs(data_residual))),
    }
