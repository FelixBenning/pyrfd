import pytest
from pyrfd.covariance import SquaredExponential

@pytest.mark.parametrize("mean,var,scale", [(1,1,2)])
def test_squared_exponential(mean, var, scale):
    cov = SquaredExponential(
        mean=mean,
        variance=(var,0),
        gradient_var=(var/scale ** 2,0),
        dims=1
    )

    assert cov.scale == scale
    assert cov.variance == var

    assert cov.asymptotic_learning_rate() == (scale ** 2/mean)

    conservatism = [-0.01, 0, 0.01, 0.1, 0.5, 0.8, 0.9]
    rates = [cov.learning_rate(loss=mean, grad_norm=1, conservatism= c) for c in conservatism]
    
    if sorted(rates, reverse=True) != rates:
        raise AssertionError("Learning rates are not decreasing in conservatism")

