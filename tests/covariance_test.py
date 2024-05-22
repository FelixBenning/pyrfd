import pytest
import numpy as np
from pyrfd.covariance import SquaredExponential

rng = np.random.default_rng()

@pytest.mark.parametrize("mean,var,scale", [
    (1,1,2),
    (0.1,3,10),
    (10,3,0.5),
])
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
    
    
    for cost in [0, 0.1, mean, 3, -1]:
        # taking a zero step size should not change the cost
        assert cov.cond_expectation(0, cost, np.exp(rng.random())) == pytest.approx(cost)

        # taking a zero step size implies that there is no conditional variance
        # as the value is already known
        assert cov.cond_variance(0) == pytest.approx(0)

        # going really far away should imply that the conditional variance is the same as the variance
        assert cov.cond_variance(1e12) == pytest.approx(var)

