"""Phase 1 full-waveform inversion package.

The package keeps a small convenience API at the top level, but we resolve
those exports lazily so importing a lightweight helper module such as
`fwi.stride_benchmark` does not immediately pull in heavier optional runtime
dependencies from unrelated parts of the codebase.
"""

from __future__ import annotations

from importlib import import_module

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

_EXPORTS = {
    "build_backend": (".backends", "build_backend"),
    "build_brain_fwi_problem": (".problem", "build_brain_fwi_problem"),
    "compute_metrics": (".metrics", "compute_metrics"),
    "dldx": (".problem", "dldx"),
    "init_params": (".problem", "init_params"),
    "loss": (".problem", "loss"),
    "run_adam": (".optimisers", "run_adam"),
    "run_lbfgsb": (".optimisers", "run_lbfgsb"),
    "run_sgd": (".optimisers", "run_sgd"),
    "sample": (".problem", "sample"),
    "sample_batch": (".problem", "sample_batch"),
}


def __getattr__(name: str):
    """Resolve package-level exports only when they are first requested."""

    if name not in _EXPORTS:
        raise AttributeError(f"module 'fwi' has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
