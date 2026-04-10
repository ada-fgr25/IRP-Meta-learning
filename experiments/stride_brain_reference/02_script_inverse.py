"""Tracked Stride inverse benchmark script for the brain-ultrasound example.

This file mirrors the local reference inversion script while resolving tracked
model files from the repository root so benchmark replication does not depend on
an ignored `resources/` directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stride import *


REPO_ROOT = Path(__file__).resolve().parents[2]
STARTING_MODEL_PATH = REPO_ROOT / "data" / "alpha2D-StartingModel.h5"


async def main(runtime):
    """Run the tracked Stride inverse benchmark exactly as configured."""

    np.random.seed(12345)

    shape = (500, 370)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(
        shape=shape,
        extra=extra,
        absorbing=absorbing,
        spacing=spacing,
    )

    start = 0.0
    step = 0.08e-6
    num = 2500

    time = Time(start=start, step=step, num=num)
    problem = Problem(name="alpha2D", space=space, time=time)

    vp = ScalarField.parameter(name="vp", grid=problem.grid, needs_grad=True)
    vp.load(str(STARTING_MODEL_PATH))
    problem.medium.add(vp)

    problem.transducers.default()
    num_locations = 256
    problem.geometry.default("elliptical", num_locations)
    problem.acquisitions.load(
        path=problem.output_folder,
        project_name=problem.name,
        version=0,
    )

    problem.plot()

    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    step_size = 5
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1450.0, max=3000.0)
    optimiser = GradientDescent(
        vp,
        step_size=step_size,
        process_grad=process_grad,
        process_model=process_model,
    )

    optimisation_loop = OptimisationLoop()
    max_freqs = [0.1e6, 0.2e6, 0.3e6]
    num_blocks = len(max_freqs)
    num_iters = 8
    num_shots_per_iter = 32

    for block, freq in optimisation_loop.blocks(num_blocks, max_freqs):
        await adjoint(
            problem,
            pde,
            loss,
            optimisation_loop,
            optimiser,
            vp,
            num_iters=num_iters,
            select_shots=dict(num=num_shots_per_iter, randomly=True),
            f_max=freq,
            max_freqs=max_freqs,
            kernel="OT4",
            fw3d_mode=True,
            interpolation_type="hicks",
            platform="cpu",
        )
    vp.plot()


if __name__ == "__main__":
    mosaic.run(main)
