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
* Stride-inspired elliptical acquisition geometry for brain ultrasound
* A procedural brain phantom with skull, ventricles, and lesion structure
* Classical optimisation baselines using SGD, Adam, and L-BFGS-B
* Basic evaluation metrics for model and data misfit
* A lightweight wrapper for the bundled Stride scripts so the reference brain
  example can be run as a benchmark outside the differentiable JAX path

The repository also contains local reference resources from Stride and Descend under `resources/`. Those files are currently used as design references rather than imported runtime dependencies.

## 🚀 Research Roadmap

The project will be developed progressively, starting from simple and interpretable baselines towards more advanced learned optimisation strategies.

### Phase 1 — Classical FWI Baseline

* Implement differentiable FWI with forward and adjoint solvers
* Optimisation using standard methods (SGD, Adam, L-BFGS)
* Establish baseline performance and evaluation metrics

Current implementation note:

* The active baseline uses a JAX-native solver so the full forward and adjoint pipeline remains differentiable end to end.
* The JAX baseline is the research path for differentiable optimisation and future meta-learning experiments.
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

## 🔗 Related Work

- Descend (Moseley et al., 2024): https://gitlab.com/benmoseley/descend-pmlr-2024

## ▶️ Running The Phase 1 Baseline

The baseline experiment entrypoint is:

```bash
PYTHONPATH=src python experiments/phase1_brain_fwi.py --optimizer adam --steps 20
```

This experiment is intentionally JAX-only so the optimisation path remains
fully differentiable.

Useful options include:

* `--optimizer {sgd,adam,lbfgsb}`
* `--nx`, `--ny`, `--nt` to change grid and recording sizes
* `--n-transducers`, `--n-shots` to adjust acquisition cost

Outputs are written to `experiments/outputs/phase1_brain_fwi/` and include:

* reconstruction plots
* scalar metrics in JSON
* optimisation history in JSON

The core implementation lives directly under `src/fwi/` so imports stay short and explicit, for example `from fwi.problem import init_params`.

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

Replication note:

* The wrapper is not its own Stride reimplementation. Reproducing the benchmark requires the bundled Stride scripts in `resources/stride_fwi_brain/`, especially [01_script_forward.py](/home/fgr25/IRP/IRP-Meta-learning/resources/stride_fwi_brain/01_script_forward.py) and [02_script_inverse.py](/home/fgr25/IRP/IRP-Meta-learning/resources/stride_fwi_brain/02_script_inverse.py).
* With the default resource directory, the benchmark settings come from those scripts directly: elliptical geometry with `256` locations, `0.25 MHz` tone-burst source, `3` inversion blocks, `8` iterations per block, `32` shots per iteration, `f_max` schedule `[0.1, 0.2, 0.3] MHz`, `OT4` kernel, Hicks interpolation, and `cpu` platform.
* You can inspect the exact commands and the encoded benchmark settings without running Stride via `PYTHONPATH=src python experiments/stride_brain_benchmark.py --dry-run --mode both`.

## 👤 Author

Francesco Giuseppe Remondi
MSc Applied Computational Science and Engineering
Imperial College London
