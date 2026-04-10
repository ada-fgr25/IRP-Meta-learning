"""Run the Phase 1 differentiable brain FWI baseline."""

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

from irp_meta_learning.fwi.acoustics import simulate_survey
from irp_meta_learning.fwi.config import (
    AcquisitionConfig,
    BrainFWIConfig,
    GridConfig,
    ModelConfig,
    SolverConfig,
    TimeConfig,
)
from irp_meta_learning.fwi.metrics import compute_metrics
from irp_meta_learning.fwi.optimisers import run_adam, run_lbfgsb, run_sgd
from irp_meta_learning.fwi.problem import dldx, init_params


def parse_args():
    """Parse command-line arguments for the baseline experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "lbfgsb"], default="adam")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5.0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=72)
    parser.add_argument("--nt", type=int, default=320)
    parser.add_argument("--n-transducers", type=int, default=24)
    parser.add_argument("--n-shots", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/phase1_brain_fwi"))
    return parser.parse_args()


def build_config(args) -> BrainFWIConfig:
    """Build an experiment configuration from CLI options."""

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
    x0, auxs, x_exact = params["x0"], (params["y_obs"],), params["x_exact"]
    bounds = (config.model.min_velocity, config.model.max_velocity)

    # JIT the objective once so repeated optimiser calls stay reasonably fast.
    loss_grad_fn = jax.jit(lambda model: dldx(params, model, auxs))

    if args.optimizer == "sgd":
        x_hat, history, final_loss = run_sgd(
            x0,
            loss_grad_fn,
            learning_rate=args.learning_rate,
            n_steps=args.steps,
            bounds=bounds,
            true_model=x_exact,
        )
    elif args.optimizer == "adam":
        x_hat, history, final_loss = run_adam(
            x0,
            loss_grad_fn,
            learning_rate=args.learning_rate,
            n_steps=args.steps,
            bounds=bounds,
            true_model=x_exact,
        )
    else:
        x_hat, history, final_loss = run_lbfgsb(
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

    figure = plt.figure(figsize=(12, 4))
    axes = figure.subplots(1, 3)
    for ax, image, title in zip(
        axes,
        [x_exact, x0, x_hat],
        ["True velocity", "Initial model", "Recovered model"],
    ):
        im = ax.imshow(image.T, origin="lower", cmap="viridis")
        figure.colorbar(im, ax=ax)
        ax.set_title(title)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.optimizer}_reconstruction.png", dpi=150)
    plt.close(figure)

    with (args.output_dir / f"{args.optimizer}_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with (args.output_dir / f"{args.optimizer}_history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
