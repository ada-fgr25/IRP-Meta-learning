"""Phase 1 full-waveform inversion package.

This package contains the current baseline implementation for differentiable
brain-imaging FWI in JAX. It is kept directly under `src/fwi` so experiment
scripts and notebooks can use short, readable imports.
"""

from .backends import build_backend
from .metrics import compute_metrics
from .optimisers import run_adam, run_lbfgsb, run_sgd
from .problem import (
    build_brain_fwi_problem,
    dldx,
    init_params,
    loss,
    sample,
    sample_batch,
)

__all__ = [
    "build_backend",
    "build_brain_fwi_problem",
    "compute_metrics",
    "dldx",
    "init_params",
    "loss",
    "run_adam",
    "run_lbfgsb",
    "run_sgd",
    "sample",
    "sample_batch",
]
