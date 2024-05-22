"""
Module providing Covariance models to pass to RFD. I.e. they can be fitted
using loss samples and they provide a learning rate
"""

from __future__ import annotations
from abc import abstractmethod
from ctypes import ArgumentError
from logging import warning
from typing import Tuple
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from . import plots

from .batchsize import (
    batchsize_counts,
    empirical_intercept_variance,
    theoretical_intercept_variance,
)

from .sampling import CSVSampleCache, IsotropicSampler, SampleCache
from .regression import (
    ScalarRegression,
    fit_mean_var,
    isotropic_derivative_var_estimation,
)


class IsotropicCovariance:
    """Abstract isotropic covariance class, providing some fallback methods.

    Can be subclassed for specific covariance models (see e.g. SquaredExponential)
    """

    __slots__ = "mean", "var_reg", "g_var_reg", "dims", "_fitted"

    def __init__(
        self,
        *,
        mean=None,
        variance: Tuple[float, float] | None = None,
        gradient_var: Tuple[float, float] | None = None,
        dims=None,
    ) -> None:

        self.mean = mean
        self.dims = dims

        self.var_reg = None
        self.g_var_reg = None

        if variance is not None:
            self.var_reg = ScalarRegression(*variance)
            assert self.var_reg.is_plausible_variance_regression

        if gradient_var is not None:
            self.g_var_reg = ScalarRegression(*gradient_var)
            assert self.g_var_reg.is_plausible_variance_regression

        self._fitted = False
        if self._is_fitted():
            self._fitted = True

    def _is_fitted(self):
        return (self.mean is not None) and self.var_reg and self.g_var_reg and self.dims

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + self._repr_helper() + ")"

    def _repr_helper(self):
        var = repr(None)
        if self.var_reg:
            var = f"({self.var_reg.intercept}, {self.var_reg.slope})"

        g_var = repr(None)
        if self.g_var_reg:
            g_var = f"({self.g_var_reg.intercept}, {self.g_var_reg.slope})"

        return (
            f"mean={self.mean}, "
            f"variance={var}, "
            f"gradient_var={g_var}, "
            f"dims={self.dims}"
        )

    @abstractmethod
    def kernel(self, neg_sq_half):
        """the kernel of the covariance model as a function of
            `neg_sq_half = - dist^2/2`
        where dist is the distance between two points
        """
        return NotImplemented

    @abstractmethod
    def diff_kernel(self, neg_sq_half):
        """the derivative of the kernel of the covariance model as a function of
            `neg_sq_half = - dist^2/2`
        where dist is the distance between two points
        """
        return NotImplemented

    @abstractmethod
    def diff2_kernel(self, neg_sq_half):
        """the second derivative of the kernel of the covariance model as a function of
            `neg_sq_half = - dist^2/2`
        where dist is the distance between two points
        """
        return NotImplemented

    def cond_expectation(self, stepsize, loss, grad_norm, b_size_inv=0):
        """the conditional expectation of the cost given batch loss and gradient norm
        assuming a step in the direction of the negative gradient
        """
        neg_sq_half = -(stepsize**2) / 2

        corr = self.kernel(neg_sq_half) / self.var_reg(b_size_inv)

        d_cov = stepsize * self.diff_kernel(neg_sq_half)
        d_corr = d_cov * self.dims / self.g_var_reg(b_size_inv)

        return self.mean + corr * (loss - self.mean) - d_corr * grad_norm

    def diff_cond_expectation(self, stepsize, loss, grad_norm, b_size_inv=0):
        """derivative of the conditional expectation of the cost given batch loss and gradient norm
        assuming a step in the direction of the negative gradient
        """
        sq_step = stepsize**2
        neg_sq_half = -sq_step / 2

        corr = -stepsize * self.diff_kernel(neg_sq_half) / self.var_reg(b_size_inv)
        d_cov = self.diff_kernel(neg_sq_half) - sq_step * self.diff2_kernel(neg_sq_half)
        d_corr = d_cov * self.dims / self.g_var_reg(b_size_inv)

        return corr * (loss - self.mean) - d_corr * grad_norm

    def cond_variance(self, stepsize, b_size_inv=0):
        """the conditional variance of the cost given batch loss and gradient norm
        assuming a step in the direction of the negative gradient
        """
        sq_step = stepsize**2
        neg_sq_half = -sq_step / 2

        var = self.var_reg.intercept
        explained_var_1 = (self.kernel(neg_sq_half) ** 2) / self.var_reg(b_size_inv)

        d_cov_sq = sq_step * (self.diff_kernel(neg_sq_half) ** 2)
        explained_var_2 = d_cov_sq * self.dims / self.g_var_reg(b_size_inv)
        return var - explained_var_1 - explained_var_2

    def diff_cond_variance(self, stepsize, b_size_inv=0):
        """derivative of the conditional variance of the cost given batch loss and gradient norm
        assuming a step in the direction of the negative gradient
        """
        sq_step = stepsize**2
        neg_sq_half = -sq_step / 2
        t1 = stepsize * self.kernel(neg_sq_half) * self.diff_kernel(neg_sq_half)
        t1 /= self.var_reg(b_size_inv)

        t2 = self.diff_kernel(neg_sq_half) * self.diff2_kernel(neg_sq_half)
        t2 *= stepsize**3
        t3 = stepsize * self.diff_kernel(neg_sq_half) ** 2
        return 2 * (t1 + ((t2 - t3) * self.dims / self.g_var_reg(b_size_inv)))

    def learning_rate(self, loss, grad_norm, *, b_size_inv=0, conservatism=0):
        """learning rate of this covariance model from the RFD paper"""

        if conservatism != 0:
            regularization = stats.norm.ppf(conservatism / 2 + 0.5)

            def diff_cost(stepsize):
                cond_var = self.cond_variance(stepsize, b_size_inv=b_size_inv)
                d_cond_var = self.diff_cond_variance(stepsize, b_size_inv=b_size_inv)
                reg = regularization * 0.5 * d_cond_var / np.sqrt(cond_var)

                dce = self.diff_cond_expectation(stepsize, loss, grad_norm, b_size_inv)
                return dce + reg

        else:

            def diff_cost(stepsize):
                return self.diff_cond_expectation(stepsize, loss, grad_norm, b_size_inv)

        left = 0  # diff_cost(0) < 0
        right = next(s for s in [2**x for x in range(-20, 20)] if diff_cost(s) > 0)

        # find minimum
        stepsize = bisection_incr(diff_cost, left, right)

        return stepsize / grad_norm  # learning rate

    def asymptotic_learning_rate(self, b_size_inv=0, limiting_loss=0):
        """asymptotic learning rate of RFD

        b_size_inverse:
            The inverse 1/b of the batch size b for which the learning rate is used
            (default is 0)
        limiting_loss:
            The loss at the end of optimization (default is 0)
        """
        assert self._fitted, "The covariance has not been fitted yet."
        assert (
            b_size_inv <= 1
        ), "Please pass the batch size inverse 1/b not the batch size b"
        enumerator = self.var_reg(b_size_inv)
        denominator = (
            self.g_var_reg(b_size_inv) / self.dims * (self.mean - limiting_loss)
        )
        return enumerator / denominator

    def fit(self, df: pd.DataFrame):
        """Fit the covariance model with loss and gradient norm samples
        provided in a pandas dataframe, with columns containing:

        - batchsize
        - loss
        - grad_norm or sq_grad_norm
        """
        if ("sq_grad_norm" not in df) and ("grad_norm" in df):
            df["sq_grad_norm"] = df["grad_norm"] ** 2

        tmp = fit_mean_var(df["batchsize"], df["loss"], var_reg=self.var_reg)
        self.mean = tmp["mean"]
        self.var_reg: ScalarRegression = tmp["var_regression"]

        self.g_var_reg: ScalarRegression = isotropic_derivative_var_estimation(
            df["batchsize"], df["sq_grad_norm"], g_var_reg=self.g_var_reg
        )

        self._fitted = True

        if not self.var_reg.is_plausible_variance_regression:
            warning(
                "The variance regression has a negative intercept, since negative "
                "variances are not meaningful, the regression is reset to None."
            )
            self._fitted = False
            self.var_reg = None

        if not self.g_var_reg.is_plausible_variance_regression:
            warning(
                "The gradient variance regression has a negative intercept, since negative "
                "variances are not meaningful, the regression is reset to None."
            )
            self._fitted = False
            self.g_var_reg = None

        return self

    def plot_sanity_checks(self, df: pd.DataFrame, batch_sizes=None):
        """Plot Sanity Check Plots"""
        fig, axs = plt.subplots(3, 2, figsize=(9, 8))

        plots.plot_loss(axs[0, 0], df, mean=self.mean, var_reg=self.var_reg)
        axs[0, 0].set_xscale("log")
        axs[0, 0].set_xlabel("")

        plots.plot_squared_losses(axs[1, 0], df, mean=self.mean, var_reg=self.var_reg)
        axs[1, 0].set_xlabel("")

        plots.plot_gradient_norms(
            axs[2, 0],
            df,
            g_var_reg=self.g_var_reg,
            dims=self.dims,
        )

        plots.qq_plot_losses(axs[0, 1], df, batch_sizes=batch_sizes)
        plots.qq_plot_squared_losses(
            axs[1, 1], df, mean=self.mean, batch_sizes=batch_sizes
        )
        plots.qq_plot_sq_gradient_norms(
            axs[2, 1], df, dims=self.dims, batch_sizes=batch_sizes
        )

        return (fig, axs)

    # pylint: disable=too-many-arguments,too-many-locals
    def auto_fit(
        self,
        model_factory,
        loss,
        data,
        *,
        cache: SampleCache | str | None = None,
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
        print(f"\n\nAutomatically fitting Covariance Model: {repr(self)}")

        sampler = IsotropicSampler(
            model_factory,
            loss,
            data,
            cache=(cache if isinstance(cache, SampleCache) else CSVSampleCache(cache)),
        )
        self.dims = sampler.dims

        budget = initial_budget
        outer_pgb = None
        for idx in range(max_iter):
            if len(sampler) > 0 and sampler.sample_cost >= initial_budget:
                self.fit(sampler.snapshot_as_dataframe())

            bsize_counts = sampler.bsize_counts
            if self._fitted:
                var_var = empirical_intercept_variance(bsize_counts, self.var_reg)
                rel_error = np.sqrt(var_var) / self.var_reg.intercept_
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
                    / self.var_reg.intercept_
                )
                # need: lim_sdv/sqrt(budget) < tol
                pred_necessary_budget = (lim_sdv / tol) ** 2

                # allocate budget in 20% chunks to allow for early stopping
                budget = min(
                    pred_necessary_budget / 5,
                    (pred_necessary_budget - sampler.sample_cost) * 1.1,
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
                    outer_pgb.update(sampler.sample_cost)
                else:
                    outer_pgb.total = int(np.ceil(pred_necessary_budget))
                outer_pgb.refresh()
                # PROGRESS ======================================================

            needed_bsize_counts = batchsize_counts(
                budget,
                self.var_reg,
                bsize_counts,
            )
            used_budget = sampler.sample(needed_bsize_counts)
            if outer_pgb:
                outer_pgb.update(used_budget)
        if outer_pgb:
            outer_pgb.close()


class SquaredExponential(IsotropicCovariance):
    """The Squared exponential covariance model. I.e.

        C(|x-y|^2/2) = self.variance * exp(-|x-y|^2/(2*self.scale^2))

    needs to be fitted using .auto_fit or .fit.
    """

    @property
    def variance(self):
        """the estimated variance (should only be accessed after fitting)"""
        if self._fitted:
            return self.var_reg.intercept
        raise ArgumentError(
            "The covariance is not fitted yet, use `auto_fit` or `fit` before use"
        )

    @property
    def scale(self):
        """the estimated scale (should only be accessed after fitting)"""
        if self._fitted:
            return np.sqrt(self.variance * self.dims / self.g_var_reg.intercept)
        raise ArgumentError(
            "The covariance is not fitted yet, use `auto_fit` or `fit` before use"
        )

    def kernel(self, neg_sq_half):
        return self.variance * np.exp(neg_sq_half / (self.scale**2))

    def diff_kernel(self, neg_sq_half):
        return self.kernel(neg_sq_half) / (self.scale**2)

    def diff2_kernel(self, neg_sq_half):
        return self.kernel(neg_sq_half) / (self.scale**4)

    def learning_rate(self, loss, grad_norm, *, b_size_inv=0, conservatism=0):
        """RFD learning rate from Random Function Descent paper"""
        if conservatism != 0:  # Fallback
            return super().learning_rate(
                loss, grad_norm, b_size_inv=b_size_inv, conservatism=conservatism
            )

        var_reg = self.var_reg
        g_var_reg = self.g_var_reg

        # C(0)/(C(0) + 1/b * C_eps(0))
        var_adjust = var_reg.intercept / var_reg(b_size_inv)
        var_g_adjust = g_var_reg.intercept / g_var_reg(b_size_inv)

        t1 = var_adjust * (self.mean - loss) / 2
        t1 = t1 if t1 > 0 else 0  # stability
        t2 = torch.sqrt(
            torch.as_tensor(t1**2 + (self.scale * grad_norm * var_g_adjust) ** 2)
        )
        return var_g_adjust * (self.scale**2) / (t2 + t1)


class RationalQuadratic(IsotropicCovariance):
    """The Squared exponential covariance model. I.e.

        C(|x-y|^2/2) = self.variance * (1+ |x-y|^2/(2*self.scale^2))^(-self.beta/2)

    needs to be fitted using .auto_fit or .fit.
    """

    __slots__ = ("beta",)

    def __init__(self, *, beta, **kwargs):
        self.beta = beta
        super().__init__(**kwargs)  # calls _is_fitted, beta needs to be there

    def _is_fitted(self):
        return super()._is_fitted() and self.beta is not None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(beta={self.beta}, " + self._repr_helper() + ")"
        )

    @property
    def variance(self):
        """the estimated variance (should only be accessed after fitting)"""
        if self._fitted:
            return self.var_reg.intercept
        raise ArgumentError(
            "The covariance is not fitted yet, use `auto_fit` or `fit` before use"
        )

    @property
    def scale(self):
        """the estimated scale (should only be accessed after fitting)"""
        if self._fitted:
            return np.sqrt(self.variance * self.dims / self.g_var_reg.intercept)
        raise ArgumentError(
            "The covariance is not fitted yet, use `auto_fit` or `fit` before use"
        )

    def kernel(self, neg_sq_half):
        inner = 1 - 2 * neg_sq_half / (self.beta * self.scale**2)
        return self.variance * (inner ** (-self.beta / 2))

    def diff_kernel(self, neg_sq_half):
        inner = 1 - 2 * neg_sq_half / (self.beta * self.scale**2)
        return (self.variance / self.scale**2) * (inner ** (-self.beta / 2 - 1))

    def diff2_kernel(self, neg_sq_half):
        inner = 1 - 2 * neg_sq_half / (self.beta * self.scale**2)

        tmp = (self.variance / self.scale**4) * (1 + 2 / self.beta)
        return tmp * (inner ** (-self.beta / 2 - 2))

    def learning_rate(self, loss, grad_norm, *, b_size_inv=0, conservatism=0):
        """RFD learning rate from Random Function Descent paper"""
        if conservatism != 0:  # Fallback
            return super().learning_rate(
                loss, grad_norm, b_size_inv=b_size_inv, conservatism=conservatism
            )

        if grad_norm == 0:  # fast return
            return 0
        if loss >= self.mean:
            # stability: do not exceed this learning rate (not step size!)
            return self.scale * np.sqrt(self.beta / (1 + self.beta)) / grad_norm

        # --- Stochastic RFD ---
        var_reg = self.var_reg
        g_var_reg = self.g_var_reg

        # C(0)/(C(0) + 1/b * C_eps(0))
        var_adjust = var_reg.intercept / var_reg(b_size_inv)
        var_g_adjust = g_var_reg.intercept / g_var_reg(b_size_inv)

        grad_cost_quot = var_adjust / var_g_adjust * grad_norm / (self.mean - loss)
        # -------------------------

        tmp = np.sqrt(self.beta) / (self.scale * grad_cost_quot)
        polynomial = [-1, tmp, (1 + self.beta), tmp]
        # careful opposite order than np.root expects!

        minimum = bisection_incr(
            func=lambda x: evaluate_polynomial(polynomial, x),
            left=0,
            right=1 / np.sqrt(1 + self.beta),  # confer paper
        )

        # learning rate, not step size!
        return self.scale * np.sqrt(self.beta) * minimum / grad_norm


def evaluate_polynomial(polynomial, x):
    """evaluate a polynomial at x,

    polynomial is a list of coefficients
        where the index is the power of x
    """
    return sum(coeff * (x**idx) for idx, coeff in enumerate(polynomial))


def bisection_incr(func, left, right, tol=1e-10):
    """find the root of an increasing function on the interval [left, right]"""
    while right - left > tol:
        mid = (left + right) / 2
        if func(mid) == 0:
            return mid
        if func(mid) > 0:
            right = mid
        else:
            left = mid
    return (left + right) / 2


if __name__ == "__main__":
    pass
    # dims = 2_300_000
    # fig, axs = cov.fit(df, dims)
    # fig.show()
