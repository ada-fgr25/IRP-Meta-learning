"""Run the Phase 1 differentiable brain FWI baseline.

This script is intentionally lightweight: it assembles a baseline synthetic
brain-imaging FWI problem, runs one classical optimiser, writes metrics/history
to disk, and optionally displays the reconstruction figure interactively.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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
from fwi.filtering import bandlimit_traces
from fwi.metrics import compute_metrics
from fwi.optimisers import (
    process_global_gradient_stride_like,
    run_lbfgsb,
    run_stagewise_optax,
)
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
        "--interpolation-type",
        choices=["linear", "hicks"],
        default="linear",
        help="Source/receiver interpolation mode for the JAX solver.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=32,
        help="Number of time steps to replay per adjoint checkpoint segment.",
    )
    parser.add_argument(
        "--damping-mode",
        choices=["legacy", "stride_like", "sponge2"],
        default="stride_like",
        help="Boundary damping model used by the JAX solver.",
    )
    parser.add_argument(
        "--damping-type",
        choices=["sine", "power"],
        default="sine",
        help="Profile type used by stride-like/sponge2 damping field construction.",
    )
    parser.add_argument(
        "--damping-cells",
        type=int,
        default=40,
        help="Absorbing-layer width in grid cells.",
    )
    parser.add_argument(
        "--damping-power-degree",
        type=int,
        default=2,
        help="Power exponent used when --damping-type=power.",
    )
    parser.add_argument(
        "--damping-reflection-coefficient",
        type=float,
        default=1.0e-3,
        help="Reflection target used to derive damping strength when not overridden.",
    )
    parser.add_argument(
        "--damping-max-coefficient",
        type=float,
        default=None,
        help="Optional explicit damping coefficient override.",
    )
    parser.add_argument(
        "--damping-velocity-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale damping by maximum velocity as in Stride's damping helper.",
    )
    parser.add_argument(
        "--stride-grad-processing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a Stride-like global gradient processing pipeline before updates.",
    )
    parser.add_argument(
        "--mask-grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero gradients in the absorbing frame before applying the update.",
    )
    parser.add_argument(
        "--smooth-grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply spatial smoothing to gradients before applying the update.",
    )
    parser.add_argument(
        "--norm-grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise gradient amplitudes before applying the update.",
    )
    parser.add_argument(
        "--grad-smooth-radius",
        type=int,
        default=2,
        help="Smoothing radius used when --smooth-grad is enabled.",
    )
    parser.add_argument(
        "--max-freqs-hz",
        type=str,
        default="100000,200000,300000",
        help="Comma-separated stage cutoffs for Stride-style frequency continuation.",
    )
    parser.add_argument("--shots-per-iter", type=int, default=32)
    parser.add_argument(
        "--final-shots",
        type=int,
        default=None,
        help=(
            "Optional number of shots to use for final data-domain metrics. "
            "If omitted, metrics use all shots."
        ),
    )
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
    parser.add_argument(
        "--print-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print live optimisation progress (stage/step/loss/shot batch).",
    )
    parser.add_argument(
        "--print-shot-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Print active source/shot IDs for each step before the batched "
            "forward+adjoint run."
        ),
    )
    parser.add_argument(
        "--first-iter-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save first/mid/final iteration diagnostics for each frequency block.",
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
            interpolation_type=args.interpolation_type,
        ),
        model=ModelConfig(),
        solver=SolverConfig(
            checkpoint_interval=args.checkpoint_interval,
            damping_mode=args.damping_mode,
            damping_type=args.damping_type,
            damping_cells=args.damping_cells,
            damping_power_degree=args.damping_power_degree,
            damping_reflection_coefficient=args.damping_reflection_coefficient,
            damping_max_coefficient=args.damping_max_coefficient,
            damping_velocity_scale=args.damping_velocity_scale,
            stride_grad_processing=args.stride_grad_processing,
            mask_grad=args.mask_grad,
            smooth_grad=args.smooth_grad,
            norm_grad=args.norm_grad,
            grad_smooth_radius=args.grad_smooth_radius,
        ),
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


def _select_final_metric_shot_positions(
    all_shot_indices: jnp.ndarray,
    final_shots: int | None,
    seed: int,
) -> jnp.ndarray:
    """Select deterministic shot positions used for final metric evaluation."""

    total = int(all_shot_indices.shape[0])
    if final_shots is None or final_shots <= 0 or final_shots >= total:
        return jnp.arange(total, dtype=jnp.int32)

    rng = np.random.default_rng(seed + 7_919)
    chosen = np.sort(
        rng.choice(np.arange(total, dtype=np.int32), size=final_shots, replace=False)
    )
    return jnp.asarray(chosen, dtype=jnp.int32)


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


def _save_iteration_diagnostics(
    *,
    args,
    acquisition: object,
    config: BrainFWIConfig,
    dt: float,
    model: jnp.ndarray,
    gradient: jnp.ndarray,
    update_direction: jnp.ndarray,
    true_model: jnp.ndarray,
    observed_batch: jnp.ndarray,
    active_shot_indices: jnp.ndarray,
    fmax_hz: float,
    stage_index: int,
    step_index: int,
    n_steps_in_stage: int,
    position_tag: str,
    loss_value: float,
    output_dir: Path,
) -> None:
    """Persist diagnostics for one selected optimiser iteration."""

    modelled_batch = simulate_survey(
        model,
        acquisition,
        config,
        shot_indices=active_shot_indices,
    )
    residual = bandlimit_traces(modelled_batch - observed_batch, dt, fmax_hz)

    grad_norm = float(jnp.linalg.norm(gradient))
    update_norm = float(jnp.linalg.norm(update_direction))
    residual_norm = float(jnp.linalg.norm(residual))
    grad_update_dot = float(jnp.vdot(gradient, update_direction))
    descent_cosine = grad_update_dot / max(grad_norm * update_norm, 1.0e-20)

    diagnostics = {
        "optimizer": args.optimizer,
        "stage_index": stage_index,
        "step_index": step_index,
        "position_tag": position_tag,
        "n_steps_in_stage": n_steps_in_stage,
        "shots_in_batch": int(active_shot_indices.shape[0]),
        "fmax_hz": float(fmax_hz),
        "loss": float(loss_value),
        "gradient_l2_norm": grad_norm,
        "update_l2_norm": update_norm,
        "misfit_l2_norm": residual_norm,
        "grad_update_dot": grad_update_dot,
        "descent_cosine": float(descent_cosine),
    }

    stage_h = stage_index + 1
    step_h = step_index + 1
    file_stem = (
        f"{args.optimizer}_stage{stage_h:02d}_"
        f"{position_tag}_step{step_h:03d}_diagnostics"
    )
    diagnostics_path = output_dir / f"{file_stem}.json"
    with diagnostics_path.open("w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    model_np = np.asarray(model)
    model_error_np = np.asarray(true_model - model)
    grad_np = np.asarray(gradient)
    update_np = np.asarray(update_direction)
    residual_first_shot = np.asarray(residual[0])
    grad_error_alignment_np = np.asarray(gradient * (true_model - model))

    model_vmin, model_vmax = _shared_limits([model_np])
    model_error_vmin, model_error_vmax = _symmetric_limits([model_error_np])
    grad_vmin, grad_vmax = _symmetric_limits([grad_np])
    update_vmin, update_vmax = _symmetric_limits([update_np])
    alignment_vmin, alignment_vmax = _symmetric_limits([grad_error_alignment_np])

    figure = plt.figure(figsize=(18, 9))
    axes = figure.subplots(2, 3)

    im0 = axes[0, 0].imshow(
        model_np.T,
        origin="lower",
        cmap="viridis",
        vmin=model_vmin,
        vmax=model_vmax,
    )
    figure.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].set_title("Initial model x0")

    im1 = axes[0, 1].imshow(
        grad_np.T,
        origin="lower",
        cmap="coolwarm",
        vmin=grad_vmin,
        vmax=grad_vmax,
    )
    figure.colorbar(im1, ax=axes[0, 1])
    axes[0, 1].set_title("First-step gradient")

    im2 = axes[1, 0].imshow(
        update_np.T,
        origin="lower",
        cmap="coolwarm",
        vmin=update_vmin,
        vmax=update_vmax,
    )
    figure.colorbar(im2, ax=axes[1, 0])
    axes[1, 0].set_title("First-step update direction")

    im3 = axes[1, 1].imshow(
        residual_first_shot.T,
        origin="lower",
        cmap="coolwarm",
        aspect="auto",
    )
    figure.colorbar(im3, ax=axes[1, 1])
    axes[1, 1].set_title("First-shot residual (time x receiver)")
    axes[1, 1].set_xlabel("Time sample")
    axes[1, 1].set_ylabel("Receiver index")

    im4 = axes[0, 2].imshow(
        model_error_np.T,
        origin="lower",
        cmap="coolwarm",
        vmin=model_error_vmin,
        vmax=model_error_vmax,
    )
    figure.colorbar(im4, ax=axes[0, 2])
    axes[0, 2].set_title("Model error (x_exact - x0)")

    im5 = axes[1, 2].imshow(
        grad_error_alignment_np.T,
        origin="lower",
        cmap="coolwarm",
        vmin=alignment_vmin,
        vmax=alignment_vmax,
    )
    figure.colorbar(im5, ax=axes[1, 2])
    axes[1, 2].set_title("Pointwise alignment g*(x_exact - x0)")

    figure.suptitle(
        "Iteration diagnostics | "
        f"stage={stage_h} ({position_tag}) step={step_h}/{n_steps_in_stage} | "
        f"loss={diagnostics['loss']:.4e} | "
        f"||g||={grad_norm:.4e} | "
        f"||du||={update_norm:.4e} | "
        f"cos(g,du)={descent_cosine:.4f}",
        fontsize=11,
    )
    figure.tight_layout()

    diagnostics_plot_path = output_dir / f"{file_stem}.png"
    figure.savefig(diagnostics_plot_path, dpi=160)
    plt.close(figure)

    print(
        f"Saved iteration diagnostics JSON to: {diagnostics_path}",
        flush=True,
    )
    print(
        f"Saved iteration diagnostics plot to: {diagnostics_plot_path}",
        flush=True,
    )


def _diagnostic_steps_for_stage(n_steps: int) -> dict[int, str]:
    """Return first/mid/final diagnostic positions for one stage."""

    if n_steps <= 0:
        return {}
    mid = n_steps // 2
    mapping = {
        0: "first",
        mid: "mid",
        n_steps - 1: "final",
    }
    return mapping


def _format_shot_ids_for_log(shot_ids: jnp.ndarray, max_items: int = 8) -> str:
    """Build a compact preview string for active source/shot IDs."""

    values = [int(v) for v in jnp.asarray(shot_ids).tolist()]
    if len(values) <= max_items:
        return str(values)
    head_count = max_items // 2
    tail_count = max_items - head_count
    return f"{values[:head_count]} ... {values[-tail_count:]} (total={len(values)})"


def _write_run_complete_marker(
    output_dir: Path,
    optimizer: str,
    *,
    steps: int,
    max_freqs_hz: tuple[float, ...],
    metrics_path: Path,
    history_path: Path,
    reconstruction_path: Path,
    history_plot_path: Path,
) -> Path:
    """Persist an explicit completion marker for long WSL runs."""

    marker_path = output_dir / f"{optimizer}_RUN_COMPLETE.json"
    payload = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": optimizer,
        "steps": int(steps),
        "max_freqs_hz": [float(v) for v in max_freqs_hz],
        "artifacts": {
            "metrics_json": str(metrics_path),
            "history_json": str(history_path),
            "reconstruction_png": str(reconstruction_path),
            "history_png": str(history_plot_path),
        },
    }
    with marker_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return marker_path


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

    diagnostic_steps_by_stage = tuple(
        _diagnostic_steps_for_stage(n_steps) for n_steps in stage_steps
    )

    def progress_callback(event: dict[str, float]) -> None:
        """Print compact progress lines so long runs are easier to monitor."""

        if not args.print_progress:
            return

        if event.get("event") == "stage_start":
            stage = int(event["stage"]) + 1
            n_stages = len(stage_steps)
            n_steps_in_stage = int(event["n_steps"])
            stage_fmax_hz = max_freqs_hz[stage - 1]
            print(
                f"[stage {stage}/{n_stages}] start | "
                f"f_max={stage_fmax_hz:.0f} Hz | "
                f"steps={n_steps_in_stage}",
                flush=True,
            )
            return

        stage_zero_based = int(event["stage"])
        stage = stage_zero_based + 1
        step_in_stage = int(event["step_in_stage"]) + 1
        n_steps_in_stage = int(event["n_steps_in_stage"])
        global_step = int(event["step"]) + 1
        shot_batch_size = int(
            shot_schedule[stage_zero_based][step_in_stage - 1].shape[0]
        )
        loss_value = float(event["loss"])
        print(
            f"[stage {stage}/{len(stage_steps)} step {step_in_stage}/{n_steps_in_stage}] "
            f"global_step={global_step} | "
            f"shots={shot_batch_size} | "
            f"loss={loss_value:.6e}",
            flush=True,
        )

    # Compile once and pass per-step data as dynamic inputs. This avoids creating
    # many distinct jitted closures that capture large constants and can trigger
    # high memory pressure on benchmark-scale runs.
    batched_loss_grad = jax.jit(
        lambda model, observed_batch, active_shot_indices, fmax_hz: dldx(
            params,
            model,
            (observed_batch, fmax_hz, active_shot_indices),
        ),
        static_argnames=("fmax_hz",),
    )
    full_loss_grad = jax.jit(
        lambda model, fmax_hz: dldx(
            params,
            model,
            (auxs[0], fmax_hz),
        ),
        static_argnames=("fmax_hz",),
    )

    def make_loss_grad_fn(stage_index: int):
        fmax_hz = max_freqs_hz[stage_index]
        return lambda model: full_loss_grad(model, fmax_hz=fmax_hz)

    def make_step_loss_grad_fn(stage_index: int, step_index: int):
        shot_positions = shot_schedule[stage_index][step_index]
        observed_batch = auxs[0][shot_positions]
        active_shot_indices = all_shot_indices[shot_positions]
        fmax_hz = max_freqs_hz[stage_index]

        if args.print_progress and args.print_shot_progress:
            stage_h = stage_index + 1
            step_h = step_index + 1
            n_steps_h = stage_steps[stage_index]
            shot_preview = _format_shot_ids_for_log(active_shot_indices)
            print(
                f"[stage {stage_h}/{len(stage_steps)} step {step_h}/{n_steps_h}] "
                f"active source ids={shot_preview}",
                flush=True,
            )

        return lambda model: batched_loss_grad(
            model,
            observed_batch,
            active_shot_indices,
            fmax_hz=fmax_hz,
        )

    def step_callback(event: dict[str, object]) -> None:
        """Save diagnostics at selected points for each continuation block."""

        if not args.first_iter_diagnostics:
            return

        stage_index = int(event["stage_index"])
        step_index = int(event["step_in_stage"])
        n_steps_in_stage = int(event["n_steps_in_stage"])
        step_targets = diagnostic_steps_by_stage[stage_index]
        if step_index not in step_targets:
            return

        shot_positions = shot_schedule[stage_index][step_index]
        active_shot_indices = all_shot_indices[shot_positions]
        observed_batch = auxs[0][shot_positions]
        position_tag = step_targets[step_index]

        model_before = jnp.asarray(event["model_before"])
        model_after = jnp.asarray(event["model_after"])
        gradient = jnp.asarray(event["gradient"])
        update_direction = model_after - model_before
        loss_value = float(jnp.asarray(event["loss"]).reshape(()))

        _save_iteration_diagnostics(
            args=args,
            acquisition=params["acquisition"],
            config=params["config"],
            dt=params["config"].time.dt,
            model=model_before,
            gradient=gradient,
            update_direction=update_direction,
            true_model=x_exact,
            observed_batch=observed_batch,
            active_shot_indices=active_shot_indices,
            fmax_hz=max_freqs_hz[stage_index],
            stage_index=stage_index,
            step_index=step_index,
            n_steps_in_stage=n_steps_in_stage,
            position_tag=position_tag,
            loss_value=loss_value,
            output_dir=args.output_dir,
        )

    def process_grad_fn(
        model: jnp.ndarray,
        grad: jnp.ndarray,
        stage_index: int,
        step_index: int,
    ) -> jnp.ndarray:
        """Apply the configured Stride-like gradient preprocessing stack."""

        del model, stage_index, step_index
        if not config.solver.stride_grad_processing:
            return grad

        return process_global_gradient_stride_like(
            grad,
            damping_cells=config.solver.damping_cells,
            mask_grad=config.solver.mask_grad,
            smooth_grad=config.solver.smooth_grad,
            smooth_radius=config.solver.grad_smooth_radius,
            norm_grad=config.solver.norm_grad,
        )

    if args.optimizer == "sgd":
        x_hat, history, final_loss, snapshots = run_stagewise_optax(
            x0,
            make_loss_grad_fn,
            lambda: optax.sgd(learning_rate=args.learning_rate),
            stage_steps,
            bounds,
            true_model=x_exact,
            make_step_loss_grad_fn=make_step_loss_grad_fn,
            process_grad_fn=process_grad_fn,
            progress_callback=progress_callback,
            step_callback=step_callback,
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
            process_grad_fn=process_grad_fn,
            progress_callback=progress_callback,
            step_callback=step_callback,
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

    model_residual = x_hat - x_exact
    model_denom = jax.numpy.linalg.norm(x_exact) + 1.0e-8
    metrics = {
        "model_rmse": float(jax.numpy.sqrt(jax.numpy.mean(model_residual**2))),
        "model_relative_l2": float(jax.numpy.linalg.norm(model_residual) / model_denom),
    }
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
    metrics["final_shots"] = args.final_shots
    metrics["seed"] = args.seed
    metrics["checkpoint_interval"] = config.solver.checkpoint_interval
    metrics["damping_mode"] = config.solver.damping_mode
    metrics["damping_type"] = config.solver.damping_type
    metrics["damping_cells"] = config.solver.damping_cells
    metrics["damping_power_degree"] = config.solver.damping_power_degree
    metrics["damping_reflection_coefficient"] = (
        config.solver.damping_reflection_coefficient
    )
    metrics["damping_max_coefficient"] = config.solver.damping_max_coefficient
    metrics["damping_velocity_scale"] = config.solver.damping_velocity_scale
    metrics["extra_cells_x"] = config.solver.extra_cells_x
    metrics["extra_cells_y"] = config.solver.extra_cells_y
    metrics["space_order"] = config.solver.space_order
    metrics["trace_filter_type"] = config.solver.trace_filter_type
    metrics["trace_filter_relaxation"] = config.solver.trace_filter_relaxation
    metrics["trace_filter_order"] = config.solver.trace_filter_order
    metrics["trace_filter_zero_phase"] = config.solver.trace_filter_zero_phase
    metrics["stride_grad_processing"] = config.solver.stride_grad_processing
    metrics["mask_grad"] = config.solver.mask_grad
    metrics["smooth_grad"] = config.solver.smooth_grad
    metrics["norm_grad"] = config.solver.norm_grad
    metrics["grad_smooth_radius"] = config.solver.grad_smooth_radius
    metrics["initial_model_rmse"] = float(
        jax.numpy.sqrt(jax.numpy.mean((x0 - x_exact) ** 2))
    )
    metrics["rmse_improvement"] = metrics["initial_model_rmse"] - metrics["model_rmse"]
    metrics["update_l2_norm"] = float(jax.numpy.linalg.norm(x_hat - x0))
    # Data-domain metrics are filled after the optional full-survey pass below.
    metrics["data_rmse"] = None
    metrics["data_mae"] = None
    metrics["data_metrics_status"] = "pending_full_survey"
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
        # Use a dedicated symmetric scale per difference panel so subtle model
        # updates are not visually flattened by much larger true-model errors.
        diff_vmin, diff_vmax = _symmetric_limits([image])
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
    # Persist non-expensive artifacts first so a late memory spike in the final
    # survey pass cannot prevent reconstruction/history outputs from being
    # updated for the current run.
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved reconstruction plot to: {reconstruction_path}")
    print(f"Saved optimisation history plot to: {history_plot_path}")
    print(f"Saved optimisation history to: {history_path}")
    final_metric_positions = _select_final_metric_shot_positions(
        all_shot_indices,
        args.final_shots,
        args.seed,
    )
    final_metric_shot_indices = all_shot_indices[final_metric_positions]
    observed_for_metrics = auxs[0][final_metric_positions]
    metrics["final_metric_shots_used"] = int(final_metric_positions.shape[0])
    metrics["final_metric_total_shots"] = int(all_shot_indices.shape[0])
    print(
        "Computing final data metrics on "
        f"{metrics['final_metric_shots_used']}/"
        f"{metrics['final_metric_total_shots']} shots "
        "(memory-intensive post-processing step)...",
        flush=True,
    )

    y_hat = simulate_survey(
        x_hat,
        params["acquisition"],
        config,
        shot_indices=final_metric_shot_indices,
    )
    full_metrics = compute_metrics(x_hat, x_exact, y_hat, observed_for_metrics)
    metrics["data_rmse"] = full_metrics["data_rmse"]
    metrics["data_mae"] = full_metrics["data_mae"]
    metrics["data_metrics_status"] = "complete"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    print(
        "Acquisition: "
        f"{config.acquisition.n_shots} shots, "
        f"{config.acquisition.n_transducers} transducers"
    )
    print(f"Max frequencies (Hz): {max_freqs_hz}")
    print(f"Random shots per iteration: {args.shots_per_iter}")
    print(f"Checkpoint interval: {config.solver.checkpoint_interval}")
    print(
        "Boundary damping mode/type/cells: "
        f"{config.solver.damping_mode}/"
        f"{config.solver.damping_type}/"
        f"{config.solver.damping_cells}"
    )
    print(f"Stride-like grad processing: {config.solver.stride_grad_processing}")
    print(
        "Grad pipeline (mask/smooth/norm, radius): "
        f"{config.solver.mask_grad}/"
        f"{config.solver.smooth_grad}/"
        f"{config.solver.norm_grad}, "
        f"{config.solver.grad_smooth_radius}"
    )
    print(f"Saved metrics to: {metrics_path}")
    run_complete_marker = _write_run_complete_marker(
        args.output_dir,
        args.optimizer,
        steps=args.steps,
        max_freqs_hz=max_freqs_hz,
        metrics_path=metrics_path,
        history_path=history_path,
        reconstruction_path=reconstruction_path,
        history_plot_path=history_plot_path,
    )
    print(f"Saved completion marker to: {run_complete_marker}")

    if args.show_plots:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
