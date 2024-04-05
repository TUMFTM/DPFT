import os
import random

from typing import Any, List

import numpy as np
import torch


def as_list(item: Any) -> List[Any]:
    """Returns the given item as a list containing the item.

    Arguments:
        item: Any sequential or non sequential item.

    Returns:
        List containing the item or items.
    """
    if isinstance(item, list):
        return item
    if isinstance(item, (tuple, set)):
        return list(item)
    return [item]


def as_dtype(dtype: str):
    """Decorator to convert all inputs to a numpy array
    of a given data type.

    Arguments:
        dtype: Data type to convert the
            function inputs to.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            args = tuple((np.asarray(a, dtype=np.dtype(dtype)) for a in args))
            kwargs = {k: np.asarray(v, dtype=np.dtype(dtype)) for k, v in kwargs.items()}
            return func(*args, **kwargs)
        return wrapper
    return decorator


def interp(x: torch.Tensor,
           xp: torch.Tensor,
           fp: torch.Tensor,
           left: float = None,
           right: float = None) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (xp, fp), evaluated at x.

    Arguments:
        x: The x-coordinates at which to evaluate the
            interpolated values with shape (N, ).
        xp: The x-coordinates of the data points, must be
            monotonically increasing with shape (M, ).
        fp: The y-coordinates of the data points with shape (M, ).

    Returns:
        y: The interpolated values with shape (N, )
    """
    # Get anchor values of the linear interpolation function
    x0 = xp[0]
    x1 = xp[-1]

    y0 = fp[0]
    y1 = fp[-1]

    # Set correction values
    left = left if left is not None else y0
    right = right if right is not None else y1

    # Get linear interpolated values
    if torch.isclose((x1 - x0), torch.zeros_like(x0)):
        y = torch.zeros_like(x)
    else:
        y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Adjust values outside of the original value range
    y[x < x0] = left
    y[x > x1] = right

    return y


def round_perc(func):
    """Decorator to round all output values to their
    associated precision.
    """
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)

        # Determine minimal numerical percision of the result values
        perc = np.int_(np.min(
            [np.abs(np.log10(np.finfo(v.dtype).resolution)) for v in results]
        ))

        # Rund output values to avoid numerical error propagation
        return tuple(np.round(r, perc - 1) for r in results)
    return wrapper


def set_seed(seed: int) -> None:
    """ Sets a gloabl random seed.

    Sets a gloabl random seed and enforces deterministic algorithms.
    Reference: https://pytorch.org/docs/stable/notes/randomness.html

    Arguments:
        seed: Gloabl randon seed.
    """
    if seed is not None:
        # Python
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # Numpy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)

        # cuDNN
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
