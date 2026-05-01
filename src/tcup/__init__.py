from typing import Optional

from numpy.typing import ArrayLike

__all__ = ["tcup", "write_report"]


def __getattr__(name):
    if name == "write_report":
        from .report import write_report

        return write_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def tcup(
    x: ArrayLike,
    y: ArrayLike,
    dy: ArrayLike,
    dx: Optional[ArrayLike] = None,
    cov_x: Optional[ArrayLike] = None,
    seed: Optional[int] = None,
    backend: Optional[str] = "numpyro",
    **backend_kwargs,
):
    match backend:
        case "numpyro":
            from .numpyro import tcup as _tcup_numpyro

            return _tcup_numpyro(
                x, y, dy, dx, cov_x, seed=seed, **backend_kwargs
            )
        case "stan":
            from .stan import tcup as _tcup_stan

            return _tcup_stan(x, y, dy, dx, cov_x, seed=seed, **backend_kwargs)
        case _:
            raise NotImplementedError(f"Backend {backend} not recognised")
