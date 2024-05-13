""" Weighted least squares regression for mean and covariance estimation """
import math
from logging import warning
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


class ScalarRegression(LinearRegression):
    """A 1-dimensional Linear Regression"""

    def __init__(self, intercept=None, slope=None):
        super().__init__()

        if intercept is not None:
            self.intercept_ = intercept
        if slope is not None:
            self.coef_ = np.array([slope])

    def __call__(self, x):
        res = super().predict(np.asarray(x).reshape(-1, 1)).reshape(-1)
        if len(res) == 1:
            return res[0]
        return res

    @property
    def is_plausible_variance_regression(self):
        """ variances are positive, bool if slope and intercept are postivie """
        return (self.slope > 0) and (self.intercept > 0)

    @property
    def slope(self):
        """slope' is the coefficient of the 1-dimensional regression"""
        return self.coef_[0]

    @property
    def intercept(self):
        """intercept is the intercept of the 1-dimensional regression"""
        return self.intercept_


def fit_mean_var(
    batch_sizes: np.array,
    batch_losses: np.array,
    *,
    max_bootstrap=100,
    var_reg=None,
    logging=False,
):
    """Bootstraps weighted least squares regression (WLS) to determine the mean,
    and variance of the loss at varying batchsizes. Returns the mean and variance
    regression.

    An existing regression can be passed to act as a starting point
    for the bootstrap.
    """
    batch_sizes = np.array(batch_sizes)
    batch_losses = np.array(batch_losses)
    b_inv = 1 / batch_sizes

    if var_reg is None:
        var_reg = ScalarRegression(intercept=0, slope=1)
        # we initialize with the assumption that the variance at batch_size=Inf is zero
        # this will give large batch sizes outsized weighting at the start which is better
        # than the other way around, which can easily lead to negative intercept estimations.

    # bootstrapping Weighted Least Squares (WLS)
    for idx in range(max_bootstrap):
        variances = var_reg(b_inv)  # variance at batchsizes 1/b

        mu = np.average(batch_losses, weights=1 / variances)
        centered_squares = (batch_losses - mu) ** 2

        old_intercept = var_reg.intercept
        # fourth moments i.e. 3*sigma^4 = 3 * var^2 are the variance of the centered
        # squares, the weights should be 1/these variances
        # (we leave out the 3 as it does not change the relative weights)
        var_reg.fit(
            b_inv.reshape(-1, 1),
            centered_squares,
            sample_weight=1 / variances**2,
        )

        if math.isclose(old_intercept, var_reg.intercept):
            if logging:
                tqdm.write(f"Bootstrapping WLS converged in {idx} iterations")
            return {"mean": mu, "var_regression": var_reg}

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    return {"mean": mu, "var_regression": var_reg}


def isotropic_derivative_var_estimation(
    batch_sizes: np.array,
    sq_grad_norms: np.array,
    *,
    max_bootstrap=100,
    g_var_reg=None,
    logging=False,
) -> ScalarRegression:
    """Bootstraps weighted least squares regression (WLS) to determine the
    expectation of gradient norms of the loss at varying batchsizes.
    Returns the regression of the mean against 1/b where b is the batchsize.

    An existing regression can be passed to act as a starting point
    for the bootstrap.
    """
    batch_sizes = np.array(batch_sizes)
    b_inv: np.array = 1 / batch_sizes

    if g_var_reg is None:
        g_var_reg = ScalarRegression(intercept=0, slope=1)
        # we initialize with the assumption that the variance at batch_size=Inf is zero
        # this will give large batch sizes outsized weighting at the start which is better
        # than the other way around, which can easily lead to negative intercept estimations.

    # bootstrapping WLS
    for idx in range(max_bootstrap):
        variances: np.array = g_var_reg(b_inv)  # variances at batchsize 1/b

        # squared grad norms are already (iid) sums of squared Gaussians
        # variance of squares is 3Var^2 but the 3 does not matter as it cancels
        # out in the weighting we also have a sum of squares (norm), but this
        # also only results in a constant which does not matter
        old_bias = g_var_reg.intercept
        g_var_reg.fit(
            b_inv.reshape(-1, 1), sq_grad_norms, sample_weight=1 / variances**2
        )

        if math.isclose(old_bias, g_var_reg.intercept):
            if logging:
                tqdm.write(f"Bootstrapping WLS converged in {idx} iterations")
            return g_var_reg

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    return g_var_reg
