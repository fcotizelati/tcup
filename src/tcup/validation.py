"""Input validation helpers for public fitting functions."""

from numbers import Integral
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


def _as_float_array(name: str, value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        raise ValueError(f"`{name}` must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain only finite values")
    return arr


def _as_observation_vector(name: str, value: ArrayLike, n: int) -> np.ndarray:
    arr = np.squeeze(_as_float_array(name, value))
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be a one-dimensional array")
    if arr.shape[0] != n:
        raise ValueError(
            f"`{name}` must have length {n}; got length {arr.shape[0]}"
        )
    return arr


def _validate_positive(name: str, values: np.ndarray) -> None:
    if np.any(values <= 0):
        raise ValueError(f"`{name}` must contain strictly positive values")


def _validate_covariances(cov_x: np.ndarray, n: int, d: int) -> np.ndarray:
    cov_x = _as_float_array("cov_x", cov_x)
    if cov_x.shape != (n, d, d):
        raise ValueError(
            f"`cov_x` must have shape ({n}, {d}, {d}); got {cov_x.shape}"
        )
    if not np.allclose(cov_x, np.swapaxes(cov_x, 1, 2)):
        raise ValueError("`cov_x` matrices must be symmetric")
    eigvals = np.linalg.eigvalsh(cov_x)
    if np.any(eigvals <= 0):
        raise ValueError("`cov_x` matrices must be positive definite")
    return cov_x


def _covariances_from_dx(dx: np.ndarray, n: int, d: int) -> np.ndarray:
    dx = _as_float_array("dx", dx)

    match dx.shape:
        case (n_dx, d1, d2):
            if n_dx != n:
                raise ValueError(
                    f"`dx` covariance array must contain {n} matrices; "
                    f"got {n_dx}"
                )
            if d1 != d2:
                raise ValueError("`dx` covariance matrices must be square")
            if d1 != d:
                raise ValueError(
                    f"`dx` covariance matrices must be {d}x{d}; got {d1}x{d2}"
                )
            return _validate_covariances(dx, n, d)
        case (n_dx, d_dx):
            if (n_dx, d_dx) != (n, d):
                raise ValueError(
                    f"`dx` must have shape ({n}, {d}); got {dx.shape}"
                )
            _validate_positive("dx", dx)
            return (
                np.broadcast_to(np.identity(d), (n, d, d))
                * dx[:, :, np.newaxis]
                * dx[:, np.newaxis, :]
            )
        case (n_dx,):
            if n_dx != n:
                raise ValueError(
                    f"`dx` must have length {n}; got length {n_dx}"
                )
            if d != 1:
                raise ValueError(
                    "one-dimensional `dx` can only be used with "
                    "one-dimensional `x`"
                )
            _validate_positive("dx", dx)
            return np.ones((n, 1, 1)) * dx.reshape(n, 1, 1) ** 2
        case _:
            raise ValueError(
                "`dx` must be an uncertainty vector, an (N, D) uncertainty "
                "array, or an (N, D, D) covariance array"
            )


def prepare_input_arrays(
    x: ArrayLike,
    y: ArrayLike,
    dy: ArrayLike,
    dx: Optional[ArrayLike] = None,
    cov_x: Optional[ArrayLike] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert and validate regression inputs.

    Returns x as an ``(N, D)`` array, y and dy as ``(N,)`` arrays, and cov_x as
    an ``(N, D, D)`` array.
    """

    if dx is not None and cov_x is not None:
        raise ValueError("Pass either `dx` or `cov_x`, not both")

    x = _as_float_array("x", x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim != 2:
        raise ValueError("`x` must be a one- or two-dimensional array")

    n, d = x.shape
    y = _as_observation_vector("y", y, n)
    dy = _as_observation_vector("dy", dy, n)
    _validate_positive("dy", dy)

    if cov_x is None:
        if dx is None:
            raise ValueError(
                "Couldn't identify x error data; "
                "please pass either `dx` or `cov_x`"
            )
        cov_x = _covariances_from_dx(np.asarray(dx), n, d)
    else:
        cov_x = _validate_covariances(np.asarray(cov_x), n, d)

    return x, y, dy, cov_x


def validate_component_count(n_components: Optional[int]) -> Optional[int]:
    if n_components is None:
        return None
    if not isinstance(n_components, Integral) or n_components < 1:
        raise ValueError("`x_prior_components` must be a positive integer")
    return int(n_components)
