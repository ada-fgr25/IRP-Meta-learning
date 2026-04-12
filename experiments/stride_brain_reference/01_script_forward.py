"""Tracked Stride forward benchmark script for the brain-ultrasound example.

This file mirrors the local reference script that originally lived only under
`resources/stride_fwi_brain/`, but it resolves the model paths relative to the
repository root so a fresh clone can execute the benchmark with tracked files
alone.
"""

from __future__ import annotations

from pathlib import Path

import stride
from stride.utils import wavelets


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUE_MODEL_PATH = REPO_ROOT / "data" / "alpha2D-TrueModel.h5"


async def main(runtime):
    """Run the tracked Stride forward benchmark exactly as configured."""

    shape = (500, 370)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = stride.Space(
        shape=shape,
        extra=extra,
        absorbing=absorbing,
        spacing=spacing,
    )

    start = 0.0
    step = 0.08e-6
    num = 2500

    time = stride.Time(start=start, step=step, num=num)
    problem = stride.Problem(name="alpha2D", space=space, time=time)

    vp = stride.ScalarField(name="vp", grid=problem.grid)
    vp.load(str(TRUE_MODEL_PATH))
    problem.medium.add(vp)

    problem.transducers.default()
    num_locations = 256
    problem.geometry.default("elliptical", num_locations)
    problem.acquisitions.default()

    f_centre = 0.25e6
    n_cycles = 3
    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(
            f_centre,
            n_cycles,
            time.num,
            time.step,
        )

    problem.plot()
    pde = stride.IsoAcousticDevito.remote(
        grid=problem.grid,
        len=runtime.num_workers,
    )
    await stride.forward(
        problem,
        pde,
        vp,
        fw3d_mode=True,
        interpolation_type="hicks",
        kernel="OT4",
        platform="cpu",
    )


if __name__ == "__main__":
    stride.mosaic.run(main)
