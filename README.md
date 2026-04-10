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

The repository also contains local reference resources from Stride and Descend under `resources/`. Those files are currently used as design references rather than imported runtime dependencies.

## 🚀 Research Roadmap

The project will be developed progressively, starting from simple and interpretable baselines towards more advanced learned optimisation strategies.

### Phase 1 — Classical FWI Baseline

* Implement differentiable FWI with forward and adjoint solvers
* Optimisation using standard methods (SGD, Adam, L-BFGS)
* Establish baseline performance and evaluation metrics

Current implementation note:

* The active baseline uses a JAX-native solver so the full forward and adjoint pipeline remains differentiable end to end.
* A Devito-backed route is still relevant for future work, especially if we want a higher-fidelity solver that mirrors Stride more closely, but that will likely require a custom JAX wrapper rather than direct automatic differentiation through Stride itself.

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

Useful options include:

* `--optimizer {sgd,adam,lbfgsb}`
* `--nx`, `--ny`, `--nt` to change grid and recording sizes
* `--n-transducers`, `--n-shots` to adjust acquisition cost

Outputs are written to `experiments/outputs/phase1_brain_fwi/` and include:

* reconstruction plots
* scalar metrics in JSON
* optimisation history in JSON

## 👤 Author

Francesco Giuseppe Remondi
MSc Applied Computational Science and Engineering
Imperial College London
