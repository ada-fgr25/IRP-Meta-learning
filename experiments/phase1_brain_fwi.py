"""Run the Phase 1 differentiable brain FWI baseline.

This script is intentionally lightweight: it assembles a baseline synthetic
brain-imaging FWI problem, runs one classical optimiser, writes metrics/history
to disk, and optionally displays the reconstruction figure interactively.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax

# Point Matplotlib at a writable cache directory to avoid noisy warnings in
# constrained environments such as shared shells or remote workspaces.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from fwi.acoustics import simulate_survey
from fwi.config import (
    AcquisitionConfig,
    BrainFWIConfig,
    GridConfig,
    ModelConfig,
    SolverConfig,
    TimeConfig,
)
from fwi.metrics import compute_metrics
from fwi.optimisers import run_adam, run_lbfgsb, run_sgd
from fwi.problem import dldx, init_params


def parse_args():
    """Parse command-line arguments for the baseline experiment.

    The defaults target a modest CPU-sized problem so the experiment can be run
    from a laptop terminal. For larger runs we can increase the grid size, the
    recording duration, and the number of shots/transducers.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "lbfgsb"], default="adam")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5.0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=72)
    parser.add_argument("--nt", type=int, default=320)
    parser.add_argument("--n-transducers", type=int, default=48)
    parser.add_argument("--n-shots", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/phase1_brain_fwi"),
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display the reconstruction figure in an interactive window as well as saving it.",
    )
    return parser.parse_args()


def build_config(args) -> BrainFWIConfig:
    """Build an experiment configuration from CLI options.

    Keeping this translation in one place makes it easier to later swap the
    CLI parser for YAML configs or notebook-driven experiments without having to
    touch the inversion code itself.
    """

    return BrainFWIConfig(
        grid=GridConfig(nx=args.nx, ny=args.ny),
        time=TimeConfig(nt=args.nt),
        acquisition=AcquisitionConfig(
            n_transducers=args.n_transducers,
            n_shots=args.n_shots,
        ),
        model=ModelConfig(),
        solver=SolverConfig(),
    )


def main():
    """Run a full classical FWI experiment and persist summaries to disk."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    key = jax.random.PRNGKey(0)
    params = init_params(key, config=config, backend_name="jax")
    config = params["config"]
    x0, auxs, x_exact = params["x0"], (params["y_obs"],), params["x_exact"]
    bounds = (config.model.min_velocity, config.model.max_velocity)

    # JIT the objective once so repeated optimiser calls stay reasonably fast.
    loss_grad_fn = jax.jit(lambda model: dldx(params, model, auxs))

    if args.optimizer == "sgd":
        x_hat, history, final_loss, snapshots = run_sgd(
            x0,
            loss_grad_fn,
            learning_rate=args.learning_rate,
            n_steps=args.steps,
            bounds=bounds,
            true_model=x_exact,
        )
    elif args.optimizer == "adam":
        x_hat, history, final_loss, snapshots = run_adam(
            x0,
            loss_grad_fn,
            learning_rate=args.learning_rate,
            n_steps=args.steps,
            bounds=bounds,
            true_model=x_exact,
        )
    else:
        x_hat, history, final_loss, snapshots = run_lbfgsb(
            x0,
            loss_grad_fn,
            maxiter=args.steps,
            bounds=bounds,
            true_model=x_exact,
        )

    y_hat = simulate_survey(x_hat, params["geometry"], config)
    metrics = compute_metrics(x_hat, x_exact, y_hat, auxs[0])
    metrics["final_loss"] = final_loss
    metrics["optimizer"] = args.optimizer
    metrics["steps"] = args.steps
    metrics["n_transducers"] = config.acquisition.n_transducers
    metrics["n_shots"] = config.acquisition.n_shots
    metrics["n_receivers_per_shot"] = int(params["geometry"]["transducer_indices"].shape[0])
    metrics["model_source"] = config.model.source
    reconstruction_path = args.output_dir / f"{args.optimizer}_reconstruction.png"
    metrics_path = args.output_dir / f"{args.optimizer}_metrics.json"
    history_path = args.output_dir / f"{args.optimizer}_history.json"
    checkpoint_steps = [0, 10, 20]
    panels = [("True velocity", x_exact)]
    for step in checkpoint_steps:
        image = snapshots.get(step, x_hat if step >= args.steps else x0)
        panels.append((f"Step {step}", image))

    figure = plt.figure(figsize=(4 * len(panels), 4))
    axes = figure.subplots(1, len(panels))
    for ax, (title, image) in zip(axes, panels):
        im = ax.imshow(image.T, origin="lower", cmap="viridis")
        figure.colorbar(im, ax=ax)
        ax.set_title(title)

    figure.tight_layout()
    figure.savefig(reconstruction_path, dpi=150)

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    print(
        "Acquisition: "
        f"{config.acquisition.n_shots} shots, "
        f"{config.acquisition.n_transducers} transducers"
    )
    print(f"Saved reconstruction plot to: {reconstruction_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved optimisation history to: {history_path}")

    if args.show_plots:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
