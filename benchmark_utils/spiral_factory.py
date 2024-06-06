#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm

from mrinufft.trajectories.trajectory2D import (
    initialize_2D_spiral,
)

from mrinufft.trajectories.maths import R2D

def flip2center(mask_cols: list[int], center_value: int) -> np.ndarray:
    """
    Reorder a list by starting by a center_position and alternating left/right.

    Parameters
    ----------
    mask_cols: list or np.array
        List of columns to reorder.
    center_pos: int
        Position of the center column.

    Returns
    -------
    np.array: reordered columns.
    """
    center_pos = np.argmin(np.abs(np.array(mask_cols) - center_value))
    mask_cols = list(mask_cols)
    left = mask_cols[center_pos::-1]
    right = mask_cols[center_pos + 1 :]
    new_cols = []
    while left or right:
        if left:
            new_cols.append(left.pop(0))
        if right:
            new_cols.append(right.pop(0))
    return np.array(new_cols)

def get_kspace_slice_loc(
    dim_size: int,
    center_prop: int | float,
    accel: int = 4,
    pdf: str = "gaussian",
    rng = None,
    order: str = "center-out",
) -> np.ndarray:
    """Get slice index at a random position.

    Parameters
    ----------
    dim_size: int
        Dimension size
    center_prop: float or int
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state

    Returns
    -------
    np.ndarray: array of size dim_size/accel.
    """
    if accel == 0:
        return np.arange(dim_size)  # type: ignore

    indexes = list(range(dim_size))

    if not isinstance(center_prop, int):
        center_prop = int(center_prop * dim_size)

    center_start = (dim_size - center_prop) // 2
    center_stop = (dim_size + center_prop) // 2
    center_indexes = indexes[center_start:center_stop]
    borders = np.asarray([*indexes[:center_start], *indexes[center_stop:]])

    n_samples_borders = (dim_size - len(center_indexes)) // accel
    if n_samples_borders < 1:
        raise ValueError(
            "acceleration factor, center_prop and dimension not compatible."
            "Edges will not be sampled. "
        )

    rng = np.random.default_rng(rng)

    if pdf == "gaussian":
        p = norm.pdf(np.linspace(norm.ppf(0.001), norm.ppf(0.999), len(borders)))
    elif pdf == "uniform":
        p = np.ones(len(borders))
    else:
        raise ValueError("Unsupported value for pdf.")
        # TODO: allow custom pdf as argument (vector or function.)

    p /= np.sum(p)
    sampled_in_border = list(
        rng.choice(borders, size=n_samples_borders, replace=False, p=p)
    )

    line_locs = np.array(sorted(center_indexes + sampled_in_border))
    # apply order of lines
    if order == "center-out":
        line_locs = flip2center(sorted(line_locs), dim_size // 2)
    elif order == "random":
        line_locs = rng.permutation(line_locs)
    elif order == "top-down":
        line_locs = np.array(sorted(line_locs))
    else:
        raise ValueError(f"Unknown direction '{order}'.")
    return line_locs

def stack_spiral_factory(
    shape: tuple[int,int,int],
    accelz: int,
    acsz: int | float,
    n_samples: int,
    nb_revolutions: int,
    shot_time_ms: int | None = None,
    in_out: bool = True,
    spiral: str = "archimedes",
    orderz = "center-out",
    pdfz = "gaussian",
    rng  = None,
    rotate_angle: float = 0.0,
) -> np.ndarray:
    """Generate a trajectory of stack of spiral."""
    sizeZ = shape[-1]

    z_index = get_kspace_slice_loc(sizeZ, acsz, accelz, pdf=pdfz, rng=rng, order=orderz)

    if not isinstance(rotate_angle, float):
        rotate_angle = rotate_angle.value

    spiral2D = initialize_2D_spiral(
        Nc=1,
        Ns=n_samples,
        nb_revolutions=nb_revolutions,
        spiral=spiral,
        in_out=in_out,
    ).reshape(-1, 2)
    z_kspace = (z_index - sizeZ // 2) / sizeZ
    # create the equivalent 3d trajectory
    nsamples = len(spiral2D)
    nz = len(z_kspace)
    kspace_locs3d = np.zeros((nz, nsamples, 3), dtype=np.float32)
    # TODO use numpy api for this ?
    for i in range(nz):
        if rotate_angle != 0:
            rotated_spiral = spiral2D @ R2D(rotate_angle * i)
        else:
            rotated_spiral = spiral2D
        kspace_locs3d[i, :, :2] = rotated_spiral
        kspace_locs3d[i, :, 2] = z_kspace[i]

    return kspace_locs3d.astype(np.float32)
