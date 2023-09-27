"""Cartesian sampling simulation."""

import numpy as np
from scipy.stats import norm  # type: ignore


def validate_rng(rng = None):
    """Validate Random Number Generator.

    Parameters
    ----------
    rng: Generator or int or None (default)
        Random number generator or seed.

    Returns
    -------
    Generator: Random Number Generator.
    """
    if isinstance(rng, (int, list)):  # TODO use pattern matching
        return np.random.default_rng(rng)
    elif rng is None:
        return np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        raise ValueError("rng shoud be a numpy Generator, None or an integer seed.")



def get_kspace_slice_loc(
    dim_size,
    center_prop,
    accel= 4,
    pdf = "gaussian",
    rng = None,
    direction= "center-out",
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
    rng = validate_rng(rng)

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
    if direction == "center-out":
        line_locs = flip2center(sorted(line_locs), dim_size // 2)
    elif direction == "random":
        line_locs = rng.permutation(line_locs)
    elif direction is None:
        pass
    else:
        raise ValueError(f"Unknown direction '{direction}'.")
    return line_locs


def get_cartesian_mask(
    shape,
    rng = None,
    center_prop = 0.3,
    accel = 4,
    accel_axis = 0,
    pdf= "gaussian",
) -> np.ndarray:
    """
    Get a cartesian mask for fMRI kspace data.

    Parameters
    ----------
    shape: tuple
        shape of fMRI volume.
    n_frames: int
        number of frames.
    rng: Generator or int or None (default)
        Random number generator or seed.
    constant: bool
        If True, the mask is constant across time.
    center_prop: float
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state

    Returns
    -------
    np.ndarray: random mask for an acquisition.
    """
    rng = validate_rng(rng)

    mask = np.zeros(shape, dtype=np.int8)
    slicer = [slice(None, None, None)] * len(shape)
    if accel_axis < 0:
        accel_axis = len(shape) + accel_axis
    if not (0 <= accel_axis < len(shape)):
        raise ValueError(
            "accel_axis should be lower than the number of spatial dimension."
        )

    mask_loc = get_kspace_slice_loc(shape[accel_axis], center_prop, accel, pdf, rng)
    slicer[accel_axis + 1] = mask_loc
    mask[tuple(slicer)] = 1
    return mask


def flip2center(mask_cols, center_value):
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
