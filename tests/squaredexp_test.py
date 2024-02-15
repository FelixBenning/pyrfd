import torch
import pytest
from pyrfd.covariance import SquaredExponential

def test_learning_rate():
    scales = torch.rand(10) * 10
    for scale in scales:
        cov_model = SquaredExponential(scale=scale)

        # at the start
        assert cov_model.learning_rate(0, 1) == scale

        # at the end
        assert cov_model.learning_rate(-1,0) == scale**2 
        assert cov_model.learning_rate(-0.5, 1e-10) == pytest.approx(scale**2/0.5)
