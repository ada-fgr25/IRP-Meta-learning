# References

This file collects the main papers, software packages, and local reference
materials used so far in the repository. The aim is not to be a fully polished
bibliography yet, but to keep one visible and tracked place where the project
dependencies and conceptual inspirations are recorded.

## Papers

* Adler, J. and Oktem, O. (2017). "Learned Primal-Dual Reconstruction." This is one of the core learned-iterative-reconstruction references behind the project's meta-learning motivation.
* Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., Shillingford, B., and de Freitas, N. (2016). "Learning to learn by gradient descent by gradient descent." This is a foundational reference for learned optimisers and unrolled meta-optimisation.
* Benning, M., Burger, M., Celledoni, E., Ehrhardt, M. J., Owren, B., and Schonlieb, C.-B. (2021). "A Bregman framework for inverse problems and deep learning." This informs the optimisation and inverse-problem perspective described in the roadmap.
* Plessix, R.-E. (2006). "A review of the adjoint-state method for computing the gradient of a functional with geophysical applications." This is the clearest conceptual reference for the explicit forward/adjoint split now implemented in the JAX backend.

## Software And Framework References

* JAX. The differentiable research path in this repository is built around JAX and its reverse-mode automatic differentiation machinery.
* Optax. The current classical optimisation baselines and future meta-optimisation outer loops use Optax-style update rules.
* Devito. Devito remains an important reference for explicit forward/adjoint PDE workflows, even though the current benchmark direction has shifted toward using Stride directly for high-fidelity comparison.
* Stride. The benchmark path in this repository is based on the Stride brain-ultrasound example and the Stride package API used in the tracked benchmark scripts.
* Descend. Descend is a reference point for the meta-learning training structure, especially the idea of keeping the outer optimiser-learning loop in JAX.
* NumPy FFT documentation. This is the reference convention for the repository's tracked spectrum helpers and the new FFT-domain band-limiting used to approximate Stride's `f_max` continuation from within the JAX solver path.

## Implementation Notes

The current backend split is now:

* `jax` backend: explicit forward simulation in JAX, explicit adjoint-state gradient in JAX, and higher-order differentiation retained because the whole routine is still expressed with JAX primitives.
* `stride` backend: benchmark-only orchestration around the tracked reference scripts, but exposed through the same acquisition/problem-facing API so experiments can swap bookkeeping more easily.
* The JAX Phase 1 driver now also carries a Stride-oriented operating mode by default: benchmark-scale grid/time settings, a `3`-cycle `0.25 MHz` tone-burst source family, Stride-style `0.5 * sum(r^2)` loss scaling, random `32`-shot subsets per iteration, and a `0.1/0.2/0.3 MHz` `f_max` schedule implemented with a Stride-like cosine low-pass continuation filter.
* Random shot subset scheduling in the JAX driver now follows Stride's queue-based `select_shot_ids(..., randomly=True)` semantics more closely, including no-repeat consumption of one permutation before wraparound and occasional short batches at queue boundaries.
* The JAX solver now matches the tracked Stride `IsoAcousticDevito` source handling more closely: default `OT4` time stepping, the same `2 * dt**2 * vp / max(dx, dy)` source scaling, and optional first-derivative source injection through `diff_source`.
* The JAX acquisition/solver path now includes a Stride-style Hicks interpolation option with precomputed sinc/Kaiser coefficients for both source injection and receiver sampling, including the source-side smoothing tweak used in the tracked Stride code.
* Hicks acquisition setup in JAX now also mirrors Stride's `IsoAcousticDevito` sparse-function construction more closely by applying the same spacing-scaled coordinate epsilon (`1e-3 * spacing` by default) before computing precomputed interpolation references/coefficients.
* The JAX optimiser now mirrors Stride's default `ProcessGlobalGradient` pipeline more closely by applying a Stride-like cosine-ramped mask (`mask_rampoff=10`), Gaussian smoothing (`smooth_sigma=0.25` by default), and `NormField`-style model-dependent scaling (`norm_guess_change`) before SGD/Adam updates, with post-update model clipping retained as the analogue of Stride's `ProcessModelIteration`.
* The JAX misfit path now includes a differentiable analogue of Stride's `ProcessObserved` + `ProcessTraces` defaults (`shift/filter` preconditioning and `mute/filter/norm` pre-misfit conditioning, with optional `scale_per_shot`), and the explicit adjoint now obtains trace cotangents by differentiating through that processing stack.
* Optional `scale_per_shot` handling in the JAX trace pipeline now mirrors Stride's `scale_to` choice more closely by using raw observed traces as the scaling reference.
* Trace-conditioning controls in the JAX path are now split per Stride pipeline step (`filter_wavelets`, `filter_traces`, `mute_traces`, `norm_per_shot`, `scale_per_shot`) to support tighter step-by-step parity and ablation.
* The JAX trace-conditioning path now also mirrors Stride's separation between `filter_wavelets_relaxation` and `filter_traces_relaxation` rather than using one shared relaxation value.
* The JAX misfit path now includes an optional differentiable time-weighting stage (disabled by default) to support parity/ablation against Stride's optional `time_weighting` pipeline hook.
* The JAX boundary treatment now supports a Stride-inspired `sponge2` mode in addition to mask-based damping, and the solver runs on a padded domain so the absorber sits outside the physical model. This brings the update equation closer to Stride's second-order sponge formulation, but it is still an approximation rather than a full operator-level Devito boundary implementation.
* The JAX spatial operator now defaults to `space_order=10` to match Stride's Devito discretisation much more closely.
* The JAX solver now also supports fixed density/buoyancy and attenuation fields so the forward physics can include those terms even though the inversion still optimises velocity only.
* The JAX attenuation path now mirrors Stride's Devito handling more closely by applying Stride's `dB/cm` to Neper conversion and a centered-in-time attenuation update for the `power=0` law.
* The JAX `sponge2` boundary path now aligns more closely with Stride defaults by removing hard edge clamping and using Stride's absorbing-width reflection-coefficient heuristic when not overridden.
* The JAX `stride_like` damping-mask path now also removes extra hard edge clamping so boundary attenuation is governed directly by the Stride-style damping profile.
* The JAX `sponge2` update now also mirrors the Devito stencil's local `vp^2` scaling of the sponge boundary term in the second-order time update.
* The JAX `sponge2` step now also follows Devito's interior/boundary stencil split explicitly by deriving binary subdomain masks from the sponge damping field and combining both update forms.
* Stride-like damping-field velocity scaling in JAX now follows Stride's pointwise local-velocity scaling behavior rather than applying a global max-velocity factor.
* JAX source preprocessing now includes Stride-like Tukey windowing with configurable time bounds for both forward source injection and adjoint-source preparation, matching the corresponding Stride Devito setup patterns.
* The JAX solver now has a dedicated forward-only survey path that avoids adjoint checkpoint tensor allocation; forward-only calls can optionally batch shots in small vmapped chunks to tune throughput versus memory.
* The explicit JAX adjoint path now also supports optional shot mini-batching during gradient accumulation, while preserving equivalence to sequential accumulation by masking partial-tail batches.
* The remaining deliberate gaps are still important when interpreting results: the lack of a full complex-frequency-shift PML2 auxiliary-field system and the lack of direct buoyancy/attenuation parameter inversion still differ from the full Devito operator stack.

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
