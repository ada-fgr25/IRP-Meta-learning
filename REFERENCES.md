# References

This file collects the main papers, software packages, and local reference
materials used so far in the repository. The aim is not to be a fully polished
bibliography yet, but to keep one visible and tracked place where the project
dependencies and conceptual inspirations are recorded.

## Papers

* Adler, J. and Oktem, O. (2017). "Learned Primal-Dual Reconstruction." This is one of the core learned-iterative-reconstruction references behind the project's meta-learning motivation.
* Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., Shillingford, B., and de Freitas, N. (2016). "Learning to learn by gradient descent by gradient descent." This is a foundational reference for learned optimisers and unrolled meta-optimisation.
* Benning, M., Burger, M., Celledoni, E., Ehrhardt, M. J., Owren, B., and Schonlieb, C.-B. (2021). "A Bregman framework for inverse problems and deep learning." This informs the optimisation and inverse-problem perspective described in the roadmap.

## Software And Framework References

* JAX. The differentiable research path in this repository is built around JAX and its reverse-mode automatic differentiation machinery.
* Optax. The current classical optimisation baselines and future meta-optimisation outer loops use Optax-style update rules.
* Devito. Devito remains an important reference for explicit forward/adjoint PDE workflows, even though the current benchmark direction has shifted toward using Stride directly for high-fidelity comparison.
* Stride. The benchmark path in this repository is based on the Stride brain-ultrasound example and the Stride package API used in the tracked benchmark scripts.
* Descend. Descend is a reference point for the meta-learning training structure, especially the idea of keeping the outer optimiser-learning loop in JAX.

## Local Tracked Benchmark References

These files are part of the repository and define the visible benchmark path:

* [experiments/stride_brain_reference/01_script_forward.py](/home/fgr25/IRP/IRP-Meta-learning/experiments/stride_brain_reference/01_script_forward.py) — tracked Stride forward benchmark script.
* [experiments/stride_brain_reference/02_script_inverse.py](/home/fgr25/IRP/IRP-Meta-learning/experiments/stride_brain_reference/02_script_inverse.py) — tracked Stride inverse benchmark script.
* [experiments/stride_brain_reference/README.md](/home/fgr25/IRP/IRP-Meta-learning/experiments/stride_brain_reference/README.md) — notes on the tracked Stride benchmark copies.

## Local Untracked Design References

These are not part of a fresh clone, but they have still been used as design
references during development:

* `resources/stride_fwi_brain/` — original local Stride brain scripts and artefacts used to shape the benchmark wrapper and tracked copies.
* `resources/descend-pmlr-2024-main/` — local Descend checkout used to inspect the JAX-native meta-learning training structure.

## Tracked Data Artefacts

The currently tracked HDF5 models used by both the JAX baseline and the Stride
benchmark copies are:

* [data/alpha2D-TrueModel.h5](/home/fgr25/IRP/IRP-Meta-learning/data/alpha2D-TrueModel.h5)
* [data/alpha2D-StartingModel.h5](/home/fgr25/IRP/IRP-Meta-learning/data/alpha2D-StartingModel.h5)

## Notes

* The README keeps a concise project overview. This file is the more explicit running list of references used so far.
* As the project grows, this file can be expanded into a fuller bibliography with DOIs, URLs, and more complete citation metadata.
