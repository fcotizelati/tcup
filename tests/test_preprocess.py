import numpy as np

from tcup.preprocess import deconvolve


def test_deconvolve_smoke_with_numpy_2_compatibility():
    x = np.linspace(-1.0, 1.0, 12)[:, np.newaxis]
    cov_x = np.broadcast_to(np.eye(1) * 0.01, (x.shape[0], 1, 1)).copy()

    prior = deconvolve(x, cov_x, n_components=1, random_state=24601)

    assert prior["weights"].shape == (1,)
    assert prior["means"].shape == (1, 1)
    assert prior["vars"].shape == (1, 1, 1)
    assert np.isclose(prior["weights"].sum(), 1.0)
