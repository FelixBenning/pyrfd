"""
Module providing Covariance models to pass to RFD. I.e. they can be fitted
using loss samples and they provide a learning rate
"""

from abc import abstractmethod
from ctypes import ArgumentError
from logging import warning
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats
from tqdm import tqdm

from . import plots

from .batchsize import (
    batchsize_counts,
    empirical_intercept_variance,
    theoretical_intercept_variance,
)

from .sampling import CachedSamples, IsotropicSampler, budget_use


def selection(sorted_list, num_elts):
    """
    return a selection of num_elts from the sorted_list (evenly spaced in the index)
    always includes the first and last index
    """
    if len(sorted_list) < num_elts:
        return sorted_list
    idxs = np.round(np.linspace(0, len(sorted_list) - 1, num_elts)).astype(int)
    return sorted_list[idxs]


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
        var_reg = LinearRegression(fit_intercept=True)
        # we initialize with the assumption that the variance at batch_size=Inf is zero
        # this will give large batch sizes outsized weighting at the start which is better
        # than the other way around, which can easily lead to negative intercept estimations.
        var_reg.intercept_ = 0
        var_reg.coef_ = np.array([1])

    # bootstrapping Weighted Least Squares (WLS)
    for idx in range(max_bootstrap):
        variances = var_reg.predict(
            b_inv.reshape(-1, 1)
        )  # variance at batchsizes 1/b in X

        mu = np.average(batch_losses, weights=1 / variances)
        centered_squares = (batch_losses - mu) ** 2

        old_intercept = var_reg.intercept_
        # fourth moments i.e. 3*sigma^4 = 3 * var^2 are the variance of the centered
        # squares, the weights should be 1/these variances
        # (we leave out the 3 as it does not change the relative weights)
        var_reg.fit(
            b_inv.reshape(-1, 1),
            centered_squares,
            sample_weight=1 / variances**2,
        )

        if math.isclose(old_intercept, var_reg.intercept_):
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
) -> LinearRegression:
    """Bootstraps weighted least squares regression (WLS) to determine the
    expectation of gradient norms of the loss at varying batchsizes.
    Returns the regression of the mean against 1/b where b is the batchsize.

    An existing regression can be passed to act as a starting point
    for the bootstrap.
    """
    batch_sizes = np.array(batch_sizes)
    b_inv: np.array = 1 / batch_sizes

    if g_var_reg is None:
        g_var_reg = LinearRegression(fit_intercept=True)
        # we initialize with the assumption that the variance at batch_size=Inf is zero
        # this will give large batch sizes outsized weighting at the start which is better
        # than the other way around, which can easily lead to negative intercept estimations.
        g_var_reg.intercept_ = 0
        g_var_reg.coef_ = np.array([1])

    # bootstrapping WLS
    for idx in range(max_bootstrap):
        variances: np.array = g_var_reg.predict(
            b_inv.reshape(-1, 1)
        )  # variances at batchsize 1/b

        # squared grad norms are already (iid) sums of squared Gaussians
        # variance of squares is 3Var^2 but the 3 does not matter as it cancels
        # out in the weighting we also have a sum of squares (norm), but this
        # also only results in a constant which does not matter
        old_bias = g_var_reg.intercept_
        g_var_reg.fit(
            b_inv.reshape(-1, 1), sq_grad_norms, sample_weight=1 / variances**2
        )

        if math.isclose(old_bias, g_var_reg.intercept_):
            if logging:
                tqdm.write(f"Bootstrapping WLS converged in {idx} iterations")
            return g_var_reg

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    return g_var_reg


class IsotropicCovariance:
    """Abstract isotropic covariance class, providing some fallback methods.

    Can be subclassed for specific covariance models (see e.g. SquaredExponential)
    """

    __slots__ = "mean", "var_reg", "g_var_reg", "dims", "fitted"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        mean=None,
        var_reg=None,
        g_var_reg=None,
        dims=None,
        fitted=False,
    ) -> None:
        self.mean = mean
        self.var_reg = var_reg
        self.g_var_reg = g_var_reg
        self.dims = dims
        self.fitted = fitted

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    mean={self.mean},\n"
            f"    var_reg={self.var_reg},\n"
            f"    g_var_reg={self.g_var_reg},\n"
            f"    dims={self.dims},\n"
            f"    fitted={self.fitted}\n"
            ")"
        )

    @abstractmethod
    def learning_rate(self, loss, grad_norm):
        """learning rate of this covariance model from the RFD paper"""
        return NotImplemented

    def asymptotic_learning_rate(self, b_size_inverse=0, limiting_loss=0):
        """asymptotic learning rate of RFD

        b_size_inverse:
            The inverse 1/b of the batch size b for which the learning rate is used
            (default is 0)
        limiting_loss:
            The loss at the end of optimization (default is 0)
        """
        assert self.fitted, "The covariance has not been fitted yet."
        assert (
            b_size_inverse <= 1
        ), "Please pass the batch size inverse 1/b not the batch size b"
        enumerator = self.var_reg.predict(b_size_inverse)
        denominator = (
            self.g_var_reg.predict(b_size_inverse)
            / self.dims
            * (self.mean - limiting_loss)
        )
        return enumerator / denominator

    def fit(self, df: pd.DataFrame, dims):
        """Fit the covariance model with loss and gradient norm samples
        provided in a pandas dataframe, with columns containing:

        - batchsize
        - loss
        - grad_norm or sq_grad_norm
        """
        self.dims = dims
        if ("sq_grad_norm" not in df) and ("grad_norm" in df):
            df["sq_grad_norm"] = df["grad_norm"] ** 2

        tmp = fit_mean_var(df["batchsize"], df["loss"], var_reg=self.var_reg)
        self.mean = tmp["mean"]
        self.var_reg: LinearRegression = tmp["var_regression"]

        self.g_var_reg: LinearRegression = isotropic_derivative_var_estimation(
            df["batchsize"], df["sq_grad_norm"], g_var_reg=self.g_var_reg
        )
        self.fitted = True

        return self

    def plot_sanity_checks(self, df: pd.DataFrame):
        """Plot Sanity Check Plots"""
        fig, axs = plt.subplots(3, 2)

        plots.plot_loss(axs[0, 0], df, mean=self.mean, var_reg=self.var_reg)
        axs[0, 0].set_xscale("log")
        axs[0, 0].set_xlabel("")

        plots.plot_squared_losses(axs[1, 0], df, mean=self.mean, var_reg=self.var_reg)
        axs[1, 0].set_xlabel("")

        plots.plot_gradient_norms(
            axs[2, 0], df, g_var_reg=self.g_var_reg, dims=self.dims
        )

        return (fig, axs)

    # pylint: disable=too-many-arguments
    def auto_fit(
        self,
        model_factory,
        loss,
        data,
        *,
        cache=None,
        tol=0.4,
        initial_budget=6000,
        max_iter=10,
    ):
        """
        Automatically fit the covariance model
        ------

        Paremeters:
        1. A `model_factory` which returns the same randomly initialized [!]
        model every time it is called
        2. A `loss` function e.g. torch.nn.functional.nll_loss which accepts
        a prediction and a true value
        3. data, which can be passed to `torch.utils.DataLoader` with
        different batch size parameters such that it returns (x,y) tuples when
        iterated on
        """
        dims = sum(p.numel() for p in model_factory().parameters() if p.requires_grad)
        print(f"\n\nAutomatically fitting Covariance Model: {repr(self)}")

        cached_samples = CachedSamples(cache)
        sampler = IsotropicSampler(model_factory, loss, data)

        budget = initial_budget
        outer_pgb = None
        for idx in range(max_iter):
            # COPY of cached_samples, not ref
            samples = cached_samples.as_dataframe()

            bsize_counts = pd.Series()
            if len(cached_samples) > 0:
                bsize_counts = samples["batchsize"].value_counts()

            total_samples = budget_use(bsize_counts)
            if total_samples >= initial_budget:
                self.fit(samples, dims)

            var_mean = self.var_reg.intercept_
            if var_mean <= 0:
                # negative variance est -> reset
                self.fitted = False
                self.var_reg = None

            if self.fitted:
                var_var = empirical_intercept_variance(bsize_counts, self.var_reg)
                rel_error = np.sqrt(var_var) / var_mean
                tqdm.write(f"\nCheckpoint {idx}:")
                tqdm.write(
                    "-----------------------------------------------------------"
                )
                tqdm.write(f"Estimated relative error:               {rel_error}")

                if rel_error < tol:
                    tqdm.write(
                        f"\nSucessfully fitted the Covariance model to a relative error <{tol}"
                    )
                    break  # stop early

                dist = stats.rv_discrete(
                    values=(bsize_counts.index, bsize_counts / sum(bsize_counts))
                )
                lim_sdv = (
                    np.sqrt(theoretical_intercept_variance(dist, self.var_reg))
                    / var_mean
                )
                # need: lim_sdv/sqrt(budget) < tol
                pred_necessary_budget = (lim_sdv / tol) ** 2

                # allocate budget in 20% chunks to allow for early stopping
                budget = min(
                    pred_necessary_budget / 5,
                    (pred_necessary_budget - total_samples) * 1.1,
                )

                # PROGRESS Logging ================================================
                tqdm.write(
                    f"          samples needed (for tol={tol:.2}): {pred_necessary_budget:.0f}"
                )
                if not outer_pgb:
                    outer_pgb = tqdm(
                        total=int(np.ceil(pred_necessary_budget)),
                        desc="Progress in estimated samples needed",
                        unit="samples",
                        position=0,
                        leave=False,
                    )
                    outer_pgb.update(total_samples)
                else:
                    outer_pgb.total = int(np.ceil(pred_necessary_budget))
                outer_pgb.refresh()
                # PROGRESS ======================================================

            needed_bsize_counts = batchsize_counts(
                budget,
                self.var_reg,
                bsize_counts,
            )
            used_budget = sampler.sample(
                needed_bsize_counts,
                append_to=cached_samples,
            )
            if outer_pgb:
                outer_pgb.update(used_budget)
        if outer_pgb:
            outer_pgb.close()


class SquaredExponential(IsotropicCovariance):
    """The Squared exponential covariance model. I.e.

        C(x) = self.variance * exp(-x^2/(2*self.scale^2))

    needs to be fitted using .auto_fit or .fit.
    """

    @property
    def variance(self):
        """the estimated variance (should only be accessed after fitting)"""
        if self.fitted:
            return self.var_reg.intercept_
        raise ArgumentError(
            "The covariance is not fitted yet, use `auto_fit` or `fit` before use"
        )

    @property
    def scale(self):
        """the estimated scale (should only be accessed after fitting)"""
        if self.fitted:
            return np.sqrt(self.variance * self.dims / self.g_var_reg.intercept_)
        raise ArgumentError(
            "The covariance is not fitted yet, use `auto_fit` or `fit` before use"
        )

    def learning_rate(self, loss, grad_norm):
        """RFD learning rate from Random Function Descent paper"""
        tmp = (self.mean - loss) / 2
        return (self.scale**2) / (
            torch.sqrt(tmp**2 + (self.scale * grad_norm) ** 2) + tmp
        )


if __name__ == "__main__":
    pass
    # dims = 2_300_000
    # fig, axs = cov.fit(df, dims)
    # fig.show()
