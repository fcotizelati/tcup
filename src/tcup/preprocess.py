"""Routines for pre-processing the data before fitting."""

import contextlib

import numpy as np

# Bundled XDGMM still uses np.infty, which NumPy 2 removed.
if not hasattr(np, "infty"):
    np.infty = np.inf

from xdgmm import XDGMM  # noqa: E402


def deconvolve(x, cov_x, n_components=None, random_state=None):
    xdgmm = XDGMM(random_state=random_state)

    if n_components is None:
        with contextlib.redirect_stdout(None):
            _, optimal_n_comp, _ = xdgmm.bic_test(x, cov_x, range(1, 10))
        xdgmm.n_components = optimal_n_comp
    else:
        xdgmm.n_components = n_components

    xdgmm = xdgmm.fit(x, cov_x)

    return {
        "weights": xdgmm.weights,
        "means": xdgmm.mu,
        "vars": xdgmm.V,
    }
