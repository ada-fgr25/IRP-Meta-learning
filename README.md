# Meta-Learning Optimisation for Full-Waveform Inversion (FWI)

This repository contains the implementation of my MSc Individual Research Project at Imperial College London.

## 📌 Overview

Full-waveform inversion (FWI) is a powerful technique for reconstructing high-resolution acoustic velocity models from wavefield data. However, it is computationally expensive and highly non-convex.

This project explores **meta-learning optimisation strategies** for FWI, where the goal is to learn improved update rules instead of hand-designing optimisation algorithms.

The approach combines:

* Differentiable physics (forward and adjoint wave solvers)
* Learned optimisation methods
* Gradient-based inverse problem frameworks

## 🎯 Objectives

* Develop a differentiable FWI pipeline using JAX
* Design a meta-learned optimiser for iterative reconstruction
* Compare learned optimisation against classical methods (e.g. SGD, Adam, L-BFGS)
* Evaluate convergence speed, reconstruction quality, and robustness

## 🛠️ Tech Stack

* Python
* JAX
* NumPy / SciPy
* Matplotlib

## 📂 Repository Structure (Planned)

```text
src/            # Core implementation (FWI, optimisers, models)
experiments/    # Experiment scripts, outputs, and configurations
notebooks/      # Exploratory analysis and visualisations
data/           # Synthetic datasets and small cached artefacts
resources/      # Local reference material (ignored by git)
tests/          # Smoke tests for the baseline implementation
```

## 🚧 Status

This project is currently in early development.

Phase 1 now includes a runnable baseline for a synthetic brain-imaging FWI problem:

* A differentiable 2D acoustic wave solver implemented directly in JAX
* An explicit adjoint-state gradient in JAX, exposed through the shared FWI problem API
* Stride-inspired elliptical acquisition geometry for brain ultrasound
* A procedural brain phantom with skull, ventricles, and lesion structure
* Classical optimisation baselines using SGD, Adam, and L-BFGS-B
* Basic evaluation metrics for model and data misfit
* A lightweight wrapper for the bundled Stride scripts so the reference brain
  example can be run as a benchmark outside the differentiable JAX path
* A shared acquisition/problem API so experiments can inspect JAX and Stride
  workflows through one common surface
* A Stride-oriented Phase 1 configuration that now uses the tracked benchmark's
  `500 x 370` grid, `256`-location acquisition ring, `2500` time samples,
  `0.25 MHz` `3`-cycle tone-burst source family, Stride-style `0.5 * sum(r^2)`
  loss scaling, and a band-limited `f_max` continuation schedule

The repository also contains local reference resources from Stride and Descend under `resources/`. Those files are currently used as design references rather than imported runtime dependencies.

## 🚀 Research Roadmap

The project will be developed progressively, starting from simple and interpretable baselines towards more advanced learned optimisation strategies.

### Phase 1 — Classical FWI Baseline

* Implement differentiable FWI with forward and adjoint solvers
* Optimisation using standard methods (SGD, Adam, L-BFGS)
* Establish baseline performance and evaluation metrics

Current implementation note:

* The active baseline keeps the forward solver in JAX and now uses an explicit JAX adjoint-state implementation for `dldx`.
* The JAX baseline is the research path for differentiable optimisation and future meta-learning experiments.
* The explicit adjoint is written purely in JAX, so higher-order meta-gradients remain available for learned-optimiser experiments.
* The local Stride scripts under `resources/stride_fwi_brain/` are treated separately as a benchmark path rather than part of the end-to-end autodiff stack.

### Phase 2 — Meta-Learned Scalar Optimisation

* Learn global optimisation hyperparameters (learning rate, momentum)
* Explore time-dependent or piecewise schedules
* Compare against classical optimisers

### Phase 3 — Spatially Adaptive Updates

* Learn spatially varying step sizes (diagonal preconditioning)
* Introduce simple learned mappings from gradients to updates

### Phase 4 — Learned Update Operators

* Parameterise update rules using neural networks (e.g. CNNs)
* Incorporate gradient history and optimisation state
* Study stability and generalisation

### Phase 5 — Geometry-Aware Optimisation

* Explore structured update rules inspired by non-Euclidean optimisation
* Investigate connections to Bregman distances and learned metrics

### Phase 6 — Uncertainty-Aware Optimisation (Exploratory)

* Model uncertainty in update steps (stochastic optimisation)
* Investigate its role in exploration vs refinement
* Explore uncertainty-based stopping criteria

### Phase 7 — Evaluation and Analysis

* Benchmark across synthetic datasets
* Analyse convergence behaviour and robustness
* Study generalisation across different problem instances

## 📖 References

* Adler & Öktem (2017) — Learned iterative reconstruction
* Andrychowicz et al. (2016) — Learning to learn by gradient descent
* Benning et al. (2021) — Bregman optimisation methods

A more explicit running list of papers, software packages, and local benchmark
sources used by the repository lives in [REFERENCES.md](/home/fgr25/IRP/IRP-Meta-learning/REFERENCES.md).

## 🔗 Related Work

- Descend (Moseley et al., 2024): https://gitlab.com/benmoseley/descend-pmlr-2024

## ▶️ Running The Phase 1 Baseline

The baseline experiment entrypoint is:

```bash
PYTHONPATH=src python experiments/phase1_brain_fwi.py
```

This experiment is intentionally JAX-only so the optimisation path remains
fully differentiable.

Implementation note:

* `fwi.problem.build_brain_fwi_problem(...)` now returns a structured `FWIProblem` object with a shared `acquisition` description.
* `fwi.problem.dldx(...)` uses the JAX backend's explicit adjoint-state routine rather than relying on generic reverse-mode differentiation through the full time loop.

Useful options include:

* `--optimizer {sgd,adam,lbfgsb}`
* `--max-freqs-hz` to set the Stride-like `f_max` continuation schedule
* `--shots-per-iter` and `--seed` to control the random shot subsets
* `--checkpoint-interval` to trade extra recomputation for lower adjoint memory
* `--nx`, `--ny`, `--nt` to override the benchmark-aligned spatial and temporal defaults
* `--n-transducers`, `--n-shots` to adjust acquisition cost or available shot pool

Outputs are written to `experiments/outputs/phase1_brain_fwi/` and include:

* reconstruction plots
* scalar metrics in JSON
* optimisation history in JSON

The core implementation lives directly under `src/fwi/` so imports stay short and explicit, for example `from fwi.problem import init_params`.

Parity note:

* The JAX baseline now intentionally tracks several Stride inverse-script choices more closely: the same benchmark-scale geometry/time defaults, SGD with step size `5` as the default optimiser, random `32`-shot subsets per iteration, a `3`-block `f_max` schedule `[0.1, 0.2, 0.3] MHz`, and Stride-style L2 loss scaling.
* The JAX path is still an approximation rather than a full Stride reimplementation. In particular, it uses the repository's JAX solver and trace-domain FFT masking to approximate Stride's `f_max` continuation rather than calling Stride's in-process Devito pipeline.
* Full-grid runs are still demanding. The solver now reduces memory by accumulating shots sequentially and replaying the adjoint in checkpointed time segments, but large benchmark-scale runs may still need smaller shot subsets, smaller checkpoint intervals, or more capable hardware.

## ▶️ Running The Stride Benchmark

The bundled Stride benchmark wrapper lives at:

```bash
PYTHONPATH=src python experiments/stride_brain_benchmark.py --mode both
```

Useful options include:

* `--mode {forward,inverse,both}` to choose which bundled Stride script to run
* `--resource-dir` to point at a different local Stride resource directory
* `--python` to select the interpreter used to launch the Stride scripts
* `--dry-run` to print the commands without executing them

This wrapper simply orchestrates the reference scripts already stored under
`resources/stride_fwi_brain/`. It is intended for benchmarking and qualitative
comparison, not for differentiable meta-learning.

Even though the Stride workflow is still benchmark-only, it now reports its
survey setup through the same shared acquisition API used by the JAX path. That
makes it easier to write experiment code that can swap between the research
solver and the benchmark reference without changing all of its bookkeeping.

Replication note:

* The wrapper is not its own Stride reimplementation. Reproducing the benchmark requires the tracked Stride scripts in `experiments/stride_brain_reference/`, especially [01_script_forward.py](/home/fgr25/IRP/IRP-Meta-learning/experiments/stride_brain_reference/01_script_forward.py) and [02_script_inverse.py](/home/fgr25/IRP/IRP-Meta-learning/experiments/stride_brain_reference/02_script_inverse.py).
* With the default resource directory, the benchmark settings come from those scripts directly: elliptical geometry with `256` locations, `0.25 MHz` tone-burst source, `3` inversion blocks, `8` iterations per block, `32` shots per iteration, `f_max` schedule `[0.1, 0.2, 0.3] MHz`, `OT4` kernel, Hicks interpolation, and `cpu` platform.
* You can inspect the exact commands and the encoded benchmark settings without running Stride via `PYTHONPATH=src python experiments/stride_brain_benchmark.py --dry-run --mode both`.

## 👤 Author

Francesco Giuseppe Remondi
MSc Applied Computational Science and Engineering
Imperial College London
