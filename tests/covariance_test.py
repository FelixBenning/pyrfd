import pytest
import numpy as np
from pyrfd.covariance import SquaredExponential

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "mean,var,scale,cost,grad_norm,dims",
    [
        # (1,1,2),
        # (0.1,3,10),
        # (10,3,0.5),
        (
            2.740690996317481,
            0.009362028077889665,
            6.240023213610232,
            1,
            25.3108,
            2291972,
        )
    ],
)
def test_squared_exponential(mean, var, scale, cost, grad_norm, dims):
    cov = SquaredExponential(
        mean=mean, variance=(var, 0), gradient_var=(dims * var / scale**2, 0), dims=dims
    )

    assert cov.scale == scale
    assert cov.variance == var

    assert cov.asymptotic_learning_rate() == pytest.approx(scale**2 / mean)

    # taking a zero step size should not change the cost
    assert cov.cond_expectation(0, cost, np.exp(rng.random())) == pytest.approx(cost)

    # taking a zero step size implies that there is no conditional variance
    # as the value is already known
    assert cov.cond_variance(0) == pytest.approx(0)

    # going really far away should imply that the conditional variance is the same as the variance
    assert cov.cond_variance(1e12) == pytest.approx(var)

    conservatism = [-0.1, -0.01, 0, 0.01, 0.1, 0.5, 0.8, 0.9]
    rates = [
        cov.learning_rate(loss=cost, grad_norm=grad_norm, conservatism=c)
        for c in conservatism
    ]

    if sorted(rates, reverse=True) != rates:
        raise AssertionError("Learning rates are not decreasing in conservatism")
