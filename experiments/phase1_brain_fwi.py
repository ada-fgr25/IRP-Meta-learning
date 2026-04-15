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
import jax.numpy as jnp
import numpy as np
import optax

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
from fwi.optimisers import run_lbfgsb, run_stagewise_optax
from fwi.problem import dldx, init_params


def parse_args():
    """Parse command-line arguments for the baseline experiment.

    The defaults target a modest CPU-sized problem so the experiment can be run
    from a laptop terminal. For larger runs we can increase the grid size, the
    recording duration, and the number of shots/transducers.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "lbfgsb"], default="sgd")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5.0)
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--ny", type=int, default=370)
    parser.add_argument("--nt", type=int, default=2500)
    parser.add_argument("--n-transducers", type=int, default=256)
    parser.add_argument("--n-shots", type=int, default=256)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=32,
        help="Number of time steps to replay per adjoint checkpoint segment.",
    )
    parser.add_argument(
        "--max-freqs-hz",
        type=str,
        default="100000,200000,300000",
        help="Comma-separated stage cutoffs for Stride-style frequency continuation.",
    )
    parser.add_argument("--shots-per-iter", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
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
        solver=SolverConfig(checkpoint_interval=args.checkpoint_interval),
    )


def _parse_frequency_schedule(text: str) -> tuple[float, ...]:
    """Parse a comma-separated Stride-style `f_max` schedule in Hertz."""

    if not text.strip():
        return (np.inf,)
    return tuple(float(part.strip()) for part in text.split(","))


def _split_steps(total_steps: int, n_stages: int) -> tuple[int, ...]:
    """Split a total iteration budget as evenly as possible across stages."""

    base = total_steps // n_stages
    remainder = total_steps % n_stages
    return tuple(base + (1 if i < remainder else 0) for i in range(n_stages))


def _build_random_shot_schedule(
    available_shots: jnp.ndarray,
    stage_steps: tuple[int, ...],
    shots_per_iter: int,
    seed: int,
) -> tuple[tuple[jnp.ndarray, ...], ...]:
    """Pre-sample one deterministic random shot subset for each iteration.

    Stride's inverse benchmark chooses `32` shots randomly on every iteration.
    We precompute that schedule once so the run is reproducible and so each
    optimiser step sees a stable subset if its loss is evaluated multiple times.
    """

    rng = np.random.default_rng(seed)
    shot_positions = np.arange(int(available_shots.shape[0]), dtype=np.int32)
    batch_size = min(int(shots_per_iter), int(available_shots.shape[0]))
    schedule = []

    for n_steps in stage_steps:
        stage_schedule = []
        for _ in range(n_steps):
            chosen_positions = rng.choice(
                shot_positions, size=batch_size, replace=False
            )
            chosen_positions = np.sort(chosen_positions)
            stage_schedule.append(jnp.asarray(chosen_positions, dtype=jnp.int32))
        schedule.append(tuple(stage_schedule))

    return tuple(schedule)


def _shared_limits(images):
    """Return a common color scale for a collection of absolute model images."""

    vmin = min(float(image.min()) for image in images)
    vmax = max(float(image.max()) for image in images)
    return vmin, vmax


def _symmetric_limits(images):
    """Return a symmetric color scale for signed difference images."""

    vmax = max(float(jax.numpy.max(jax.numpy.abs(image))) for image in images)
    return -vmax, vmax


def _has_meaningful_change(
    initial_model,
    final_model,
    atol: float = 1.0e-8,
) -> bool:
    """Return whether optimisation changed the model by more than roundoff.

    The experiment can legitimately make almost no progress over a tiny number
    of iterations or on a nearly stationary loss surface. In those cases we
    would rather omit a redundant difference panel than imply a meaningful
    update took place.
    """

    return not bool(jax.numpy.allclose(initial_model, final_model, atol=atol, rtol=0.0))


def _plot_history(history: list[dict[str, float]], path: Path) -> None:
    """Persist a compact within-stage optimisation-history figure.

    The continuation schedule deliberately changes the objective between
    stages, so raw losses are not directly comparable across the whole run.
    This figure therefore resets the x-axis within each stage and only plots
    within-stage changes, which makes the local optimisation behaviour much
    easier to interpret.
    """

    if not history:
        return

    stages: dict[int, list[dict[str, float]]] = {}
    for entry in history:
        stage = int(entry.get("stage", 0.0))
        stages.setdefault(stage, []).append(entry)

    ordered_stages = [stages[key] for key in sorted(stages)]
    has_rmse = all("model_rmse" in entry for entry in history)
    nrows = 2 if has_rmse else 1
    ncols = len(ordered_stages)

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        squeeze=False,
    )

    for stage_index, stage_entries in enumerate(ordered_stages):
        stage_steps = list(range(len(stage_entries)))
        stage_losses = [entry["loss"] for entry in stage_entries]

        loss_ax = axes[0][stage_index]
        loss_ax.plot(stage_steps, stage_losses, marker="o", linewidth=1.5)
        loss_ax.set_title(f"Stage {stage_index} loss")
        loss_ax.set_xlabel("Step within stage")
        loss_ax.set_ylabel("Loss")
        loss_ax.set_yscale("log")
        loss_ax.grid(True, alpha=0.3)

        if has_rmse:
            stage_rmses = [entry["model_rmse"] for entry in stage_entries]
            rmse_ax = axes[1][stage_index]
            rmse_ax.plot(stage_steps, stage_rmses, marker="o", linewidth=1.5)
            rmse_ax.set_title(f"Stage {stage_index} RMSE")
            rmse_ax.set_xlabel("Step within stage")
            rmse_ax.set_ylabel("RMSE")
            rmse_ax.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    """Run a full classical FWI experiment and persist summaries to disk."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    max_freqs_hz = _parse_frequency_schedule(args.max_freqs_hz)
    stage_steps = _split_steps(args.steps, len(max_freqs_hz))
    key = jax.random.PRNGKey(args.seed)
    params = init_params(key, config=config, backend_name="jax")
    config = params["config"]
    x0, auxs, x_exact = params["x0"], (params["y_obs"],), params["x_exact"]
    bounds = (config.model.min_velocity, config.model.max_velocity)
    all_shot_indices = params["acquisition"].require_solver_arrays()[1]
    shot_schedule = _build_random_shot_schedule(
        all_shot_indices,
        stage_steps,
        args.shots_per_iter,
        args.seed,
    )

    def make_loss_grad_fn(stage_index: int):
        stage_auxs = (auxs[0], max_freqs_hz[stage_index])
        return jax.jit(lambda model: dldx(params, model, stage_auxs))

    def make_step_loss_grad_fn(stage_index: int, step_index: int):
        shot_positions = shot_schedule[stage_index][step_index]
        stage_auxs = (
            auxs[0][shot_positions],
            max_freqs_hz[stage_index],
            all_shot_indices[shot_positions],
        )
        return jax.jit(lambda model: dldx(params, model, stage_auxs))

    if args.optimizer == "sgd":
        x_hat, history, final_loss, snapshots = run_stagewise_optax(
            x0,
            make_loss_grad_fn,
            lambda: optax.sgd(learning_rate=args.learning_rate),
            stage_steps,
            bounds,
            true_model=x_exact,
            make_step_loss_grad_fn=make_step_loss_grad_fn,
        )
    elif args.optimizer == "adam":
        x_hat, history, final_loss, snapshots = run_stagewise_optax(
            x0,
            make_loss_grad_fn,
            lambda: optax.adam(learning_rate=args.learning_rate),
            stage_steps,
            bounds,
            true_model=x_exact,
            make_step_loss_grad_fn=make_step_loss_grad_fn,
        )
    else:
        loss_grad_fn = make_loss_grad_fn(len(max_freqs_hz) - 1)
        x_hat, history, final_loss, snapshots = run_lbfgsb(
            x0,
            loss_grad_fn,
            maxiter=args.steps,
            bounds=bounds,
            true_model=x_exact,
        )

    y_hat = simulate_survey(x_hat, params["acquisition"], config)
    metrics = compute_metrics(x_hat, x_exact, y_hat, auxs[0])
    metrics["backend"] = "jax"
    metrics["final_loss"] = final_loss
    metrics["optimizer"] = args.optimizer
    metrics["steps"] = args.steps
    metrics["n_transducers"] = config.acquisition.n_transducers
    metrics["n_shots"] = config.acquisition.n_shots
    metrics["n_receivers_per_shot"] = int(params["acquisition"].n_receivers)
    metrics["model_source"] = config.model.source
    metrics["max_freqs_hz"] = list(max_freqs_hz)
    metrics["stage_steps"] = list(stage_steps)
    metrics["shots_per_iter"] = args.shots_per_iter
    metrics["seed"] = args.seed
    metrics["checkpoint_interval"] = config.solver.checkpoint_interval
    metrics["initial_model_rmse"] = float(
        jax.numpy.sqrt(jax.numpy.mean((x0 - x_exact) ** 2))
    )
    metrics["rmse_improvement"] = metrics["initial_model_rmse"] - metrics["model_rmse"]
    metrics["update_l2_norm"] = float(jax.numpy.linalg.norm(x_hat - x0))
    reconstruction_path = args.output_dir / f"{args.optimizer}_reconstruction.png"
    history_plot_path = args.output_dir / f"{args.optimizer}_history.png"
    metrics_path = args.output_dir / f"{args.optimizer}_metrics.json"
    history_path = args.output_dir / f"{args.optimizer}_history.json"
    panels = [
        ("True velocity", x_exact),
        ("Initial model", x0),
        (f"Final model (step {args.steps})", x_hat),
    ]
    diff_panels = [("True - Final", x_exact - x_hat)]
    if _has_meaningful_change(x0, x_hat):
        diff_panels.insert(0, ("Final - Initial", x_hat - x0))

    absolute_images = [image for _, image in panels]
    abs_vmin, abs_vmax = _shared_limits(absolute_images)
    diff_vmin, diff_vmax = _symmetric_limits([image for _, image in diff_panels])

    figure = plt.figure(figsize=(4 * (len(panels) + len(diff_panels)), 4))
    axes = figure.subplots(1, len(panels) + len(diff_panels))
    for ax, (title, image) in zip(axes[: len(panels)], panels):
        im = ax.imshow(
            image.T,
            origin="lower",
            cmap="viridis",
            vmin=abs_vmin,
            vmax=abs_vmax,
        )
        figure.colorbar(im, ax=ax)
        ax.set_title(title)
    for ax, (title, image) in zip(axes[len(panels) :], diff_panels):
        im = ax.imshow(
            image.T,
            origin="lower",
            cmap="coolwarm",
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        figure.colorbar(im, ax=ax)
        ax.set_title(title)

    figure.tight_layout()
    figure.savefig(reconstruction_path, dpi=150)
    _plot_history(history, history_plot_path)

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
    print(f"Max frequencies (Hz): {max_freqs_hz}")
    print(f"Random shots per iteration: {args.shots_per_iter}")
    print(f"Checkpoint interval: {config.solver.checkpoint_interval}")
    print(f"Saved reconstruction plot to: {reconstruction_path}")
    print(f"Saved optimisation history plot to: {history_plot_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved optimisation history to: {history_path}")

    if args.show_plots:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
