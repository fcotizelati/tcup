import numpy as np
import pytest

from tcup.validation import prepare_input_arrays, validate_component_count


def test_prepare_input_arrays_accepts_array_like_inputs():
    x, y, dy, cov_x = prepare_input_arrays(
        x=[1, 2, 3],
        y=[2, 3, 4],
        dy=[0.1, 0.1, 0.1],
        dx=[0.2, 0.2, 0.2],
    )

    assert x.shape == (3, 1)
    assert y.shape == (3,)
    assert dy.shape == (3,)
    assert cov_x.shape == (3, 1, 1)
    assert np.isclose(cov_x[:, 0, 0], 0.04).all()


def test_prepare_input_arrays_rejects_ambiguous_x_errors():
    with pytest.raises(ValueError, match="either `dx` or `cov_x`"):
        prepare_input_arrays(
            x=[1, 2, 3],
            y=[2, 3, 4],
            dy=[0.1, 0.1, 0.1],
            dx=[0.2, 0.2, 0.2],
            cov_x=np.ones((3, 1, 1)),
        )


def test_prepare_input_arrays_rejects_bad_covariance_shape():
    with pytest.raises(ValueError, match="must have shape"):
        prepare_input_arrays(
            x=np.ones((3, 2)),
            y=np.ones(3),
            dy=np.ones(3),
            cov_x=np.ones((3, 1, 1)),
        )


def test_prepare_input_arrays_rejects_non_positive_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        prepare_input_arrays(
            x=[1, 2, 3],
            y=[2, 3, 4],
            dy=[0.1, 0.1, 0.1],
            cov_x=np.zeros((3, 1, 1)),
        )


def test_validate_component_count():
    assert validate_component_count(None) is None
    assert validate_component_count(3) == 3
    assert validate_component_count(np.int64(2)) == 2

    with pytest.raises(ValueError, match="positive integer"):
        validate_component_count(0)
