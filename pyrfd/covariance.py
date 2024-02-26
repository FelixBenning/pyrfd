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

from .batchsize import (
    DEFAULT_VAR_REG,
    batchsize_counts,
    empirical_intercept_variance,
    limit_intercept_variance,
)

from .sampling import CachedSamples, IsotropicSampler, _budget

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
    var_reg=DEFAULT_VAR_REG,
    logging=False,
):
    batch_sizes = np.array(batch_sizes)
    batch_losses = np.array(batch_losses)
    b_inv = 1 / batch_sizes

    # we initialize with the assumption that the variance at batch_size=Inf is zero
    var_reg = LinearRegression(fit_intercept=True)
    var_reg.fit(
        np.array([[0], [1]]), np.array([0, 1])
    )  # initialize with beta0=1 and beta1=1

    # bootstrapping Weighted Least Squares (WLS)
    for idx in range(max_bootstrap):
        vars = var_reg.predict(b_inv.reshape(-1, 1))  # variance at batchsizes 1/b in X

        mu = np.average(batch_losses, weights=(1 / vars))
        centered_squares = (batch_losses - mu) ** 2

        old_intercept = var_reg.intercept_
        # fourth moments i.e. 3*sigma^4 = 3 * var^2 are the variance of the centered
        # squares, the weights should be 1/these variances
        # (we leave out the 3 as it does not change the relative weights)
        var_reg.fit(
            b_inv.reshape(-1, 1),
            centered_squares,
            sample_weight=1 / vars**2,
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
    g_var_reg=DEFAULT_VAR_REG,
    logging=False,
) -> LinearRegression:
    batch_sizes = np.array(batch_sizes)
    b_inv: np.array = 1 / batch_sizes

    # bootstrapping WLS
    for idx in range(max_bootstrap):
        vars: np.array = g_var_reg.predict(
            b_inv.reshape(-1, 1)
        )  # variances at batchsize 1/b

        # squared grad norms are already (iid) sums of squared Gaussians
        # variance of squares is 3Var^2 but the 3 does not matter as it cancels
        # out in the weighting we also have a sum of squares (norm), but this
        # also only results in a constant which does not matter
        old_bias = g_var_reg.intercept_
        g_var_reg.fit(b_inv.reshape(-1, 1), sq_grad_norms, sample_weight=(1 / vars**2))

        if math.isclose(old_bias, g_var_reg.intercept_):
            if logging:
                tqdm.write(f"Bootstrapping WLS converged in {idx} iterations")
            return g_var_reg

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    return g_var_reg


class IsotropicCovariance:
    __slots__ = "mean", "var_reg", "g_var_reg", "dims", "fitted"

    def __init__(self) -> None:
        self.fitted = False
        self.var_reg = DEFAULT_VAR_REG
        self.g_var_reg = DEFAULT_VAR_REG

    @abstractmethod
    def learning_rate(self, loss, grad_norm):
        return NotImplemented

    def fit(self, df: pd.DataFrame, dims):
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

        ## ============================
        ### === sanity check plots ===
        ## ============================
        grouped = df.groupby("batchsize", sort=True)
        b_size_grouped = grouped.agg(
            loss_mean=pd.NamedAgg(column="loss", aggfunc="mean"),
            loss_var=pd.NamedAgg(
                column="loss", aggfunc=lambda x: np.mean((x - self.mean) ** 2)
            ),
            sq_grad_norm_mean=pd.NamedAgg(column="sq_grad_norm", aggfunc="mean"),
        ).reset_index()
        b_size_g_inv: np.array = 1 / b_size_grouped["batchsize"]
        var_estimates = self.var_reg.predict(b_size_g_inv.to_numpy().reshape(-1, 1))

        # only select <100 examples per batch size
        reduced_df = df.groupby("batchsize").head(100).reset_index(drop=True)
        b_size_inv = 1 / reduced_df["batchsize"]

        fig, axs = plt.subplots(3, 2)

        ## ==== Plot Losses =====
        axs[0, 0].set_xscale("log")
        # axs[0,0].set_xlabel("1/b")

        # scatterplot
        axs[0, 0].scatter(
            b_size_inv,
            reduced_df["loss"],
            s=1,  # marker size
            label=r"$\mathcal{L}_b(w)$",
        )
        # batchwise means
        axs[0, 0].plot(
            b_size_g_inv,
            b_size_grouped["loss_mean"],
            marker="*",
            label="batchwise mean",
        )
        axs[0, 0].plot(
            b_size_g_inv,
            np.full_like(b_size_g_inv, fill_value=self.mean),
            label=rf"$\hat\mu={self.mean:.4}$",
        )
        axs[0, 0].fill_between(
            x=b_size_g_inv,
            y1=self.mean + stats.norm.ppf(0.025) * np.sqrt(var_estimates),
            y2=self.mean + stats.norm.ppf(0.975) * np.sqrt(var_estimates),
            alpha=0.3,
        )
        # legend
        axs[0, 0].legend(loc="upper left")

        ## QQplot
        # b_size_selection = np.array(selection(b_size_grouped["batchsize"], 5))[1:]
        # for bs in b_size_selection:
        bsize_counts = df["batchsize"].value_counts(sort=True, ascending=False)
        bs = bsize_counts.index[0]
        stats.probplot(grouped.get_group(bs)["loss"], dist="norm", plot=axs[0, 1])
        for idx, line in enumerate(axs[0, 1].get_lines()):
            # line.set_color(f"C{idx//2}")
            if idx % 2 == 0:  # scatter
                line.set_markersize(1)
                line.set_label(f"b={bs}")
            if idx % 2 == 1:
                line.set_linestyle("--")

        axs[0, 1].set_xlabel("")
        axs[0, 1].legend()

        ## === Plot squared errors =====
        # axs[1,0].set_xlabel("1/b")
        # axs[1,0].set_xscale("log")
        axs[1, 0].scatter(
            b_size_inv,
            (reduced_df["loss"] - self.mean) ** 2,
            s=1,  # marker size
            label=r"$(\mathcal{L}_b(w)-\hat{\mu})^2$",
        )
        axs[1, 0].plot(
            b_size_g_inv,
            b_size_grouped["loss_var"],
            marker="*",
            label="batchwise mean squares",
        )
        axs[1, 0].plot(
            b_size_g_inv,
            var_estimates,
            label=r"Var$(\mathcal{L}_b(w))$",
        )
        axs[1, 0].fill_between(
            x=b_size_g_inv,
            # χ^2 confidence bounds
            y1=var_estimates + (stats.chi2.ppf(0.025, df=1) - 1) * var_estimates,
            y2=var_estimates + (stats.chi2.ppf(0.975, df=1) - 1) * var_estimates,
            alpha=0.3,
        )
        axs[1, 0].legend(loc="upper left")

        # QQ-plot against chi^2
        # for bs in b_size_selection:
        stats.probplot(
            (grouped.get_group(bs)["loss"] - self.mean) ** 2,
            dist=stats.chi2(df=1),
            plot=axs[1, 1],
        )
        for idx, line in enumerate(axs[1, 1].get_lines()):
            # line.set_color(f"C{idx//2}")
            if idx % 2 == 0:  # scatter
                line.set_markersize(1)
                line.set_label(f"b={bs}")
            if idx % 2 == 1:
                line.set_linestyle("--")
        axs[1, 1].set_title("")
        axs[1, 1].set_xlabel("")
        axs[1, 1].legend()

        # ==== Plot Gradient Norms ====
        axs[2, 0].set_xlabel("1/b")
        axs[2, 0].scatter(
            b_size_inv,
            reduced_df["sq_grad_norm"],
            s=1,  # marker size
            label=r"$\|\nabla\mathcal{L}_b(w)\|^2$",
        )
        axs[2, 0].plot(
            b_size_g_inv,
            b_size_grouped["sq_grad_norm_mean"],
            marker="*",
            label="batchwise mean",
        )
        sq_norm_means = self.g_var_reg.predict(b_size_g_inv.to_numpy().reshape(-1, 1))
        axs[2, 0].plot(
            b_size_g_inv,
            sq_norm_means,
            label=r"$\mathbb{E}[\|\nabla\mathcal{L}_b(w)\|^2]$",
        )
        axs[2, 0].fill_between(
            x=b_size_g_inv,
            y1=sq_norm_means
            + (stats.chi2.ppf(0.025, dims) - dims) * sq_norm_means / dims,
            y2=sq_norm_means
            + (stats.chi2.ppf(0.975, dims) - dims) * sq_norm_means / dims,
            alpha=0.3,
        )

        # TODO: figure out wtf is wrong with the confidence interval of the squared norm

        axs[2, 0].legend(loc="upper left")

        # for bs in b_size_selection:
        stats.probplot(
            grouped.get_group(bs)["sq_grad_norm"],
            dist=stats.chi2(df=dims),
            plot=axs[2, 1],
        )
        for idx, line in enumerate(axs[2, 1].get_lines()):
            # line.set_color(f"C{idx//2}")
            if idx % 2 == 0:  # scatter
                line.set_markersize(1)
                line.set_label(f"b={bs}")
            if idx % 2 == 1:
                line.set_linestyle("--")
        axs[2, 1].set_title("")
        axs[2, 1].legend()

        return (
            (fig, axs),
            {
                "mean": self.mean,
                "var_reg": self.var_reg,
                "g_var_reg": self.g_var_reg,
            },
        )

    def auto_fit(
        self,
        model_factory,
        loss,
        data,
        cache=None,
        tol=0.4,
        initial_budget=6000,
        max_iter=10,
    ):
        """
        Automatically fit the covariance model
        ------

        Paremeters:
        1. A `model_factory` which returns the same randomly initialized [!] model every time it is called
        2. A `loss` function e.g. torch.nn.functional.nll_loss which accepts a prediction and a true value
        3. data, which can be passed to `torch.utils.DataLoader` with different batch size parameters such
            that it returns (x,y) tuples when iterated on
        """
        dims = sum(p.numel() for p in model_factory().parameters() if p.requires_grad)
        print(f"\n\nAutomatically fitting Covariance Model: {repr(self)}")

        sampler = IsotropicSampler(model_factory, loss, data)

        if cache:
            print("Tip: You can cancel sampling at any time, samples will be saved in the cache.")
        else:
            warning("Without a cache it is necessary to re-fit the covariance model every time. Please pass a filepath to the cache parameter")

        cached_samples = CachedSamples(cache)
        budget = initial_budget
        outer_pgb = None
        for idx in range(max_iter):
            # COPY of cached_samples, not ref
            samples = cached_samples.as_dataframe()

            bsize_counts = pd.Series()
            if len(cached_samples) > 0:
                bsize_counts = samples["batchsize"].value_counts()

            total_samples = _budget(bsize_counts)
            if total_samples >= initial_budget:
                self.fit(samples, dims)

            var_mean = self.var_reg.intercept_
            if var_mean <= 0: 
                # negative variance est -> reset
                self.fitted = False
                self.var_reg = DEFAULT_VAR_REG

            if self.fitted:
                var_var = empirical_intercept_variance(bsize_counts, self.var_reg)
                rel_error = np.sqrt(var_var) / var_mean
                tqdm.write(f"\nCheckpoint {idx}:")
                tqdm.write("-----------------------------------------------------------")
                tqdm.write(f"Estimated relative error:               {rel_error}")

                if rel_error < tol:
                    tqdm.write(f"\nSucessfully fitted the Covariance model to a relative error <{tol}")
                    break  # stop early

                dist = stats.rv_discrete(
                    values=(bsize_counts.index, bsize_counts / sum(bsize_counts))
                )
                lim_sdv = (
                    np.sqrt(limit_intercept_variance(dist, self.var_reg)) / var_mean
                )
                # need: lim_sdv/sqrt(budget) < tol
                pred_necessary_budget = (lim_sdv / tol) ** 2

                # allocate budget in 20% chunks to allow for early stopping
                budget = min(
                    pred_necessary_budget / 5,
                    (pred_necessary_budget - total_samples) * 1.1,
                )

                ## PROGRESS Logging ================================================
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
                ## PROGRESS ======================================================


            needed_bsize_counts = batchsize_counts(
                budget,
                self.var_reg,
                bsize_counts,
            )
            budget_use = sampler.sample(
                needed_bsize_counts,
                append_to=cached_samples,
            )
            if outer_pgb:
                outer_pgb.update(budget_use)
        if outer_pgb:
            outer_pgb.close()


class SquaredExponential(IsotropicCovariance):
    @property
    def variance(self):
        if self.fitted:
            return self.var_reg.intercept_
        raise ArgumentError("The covariance is not fitted yet, use `auto_fit` or `fit` before use")

    @property
    def scale(self):
        if self.fitted:
            return np.sqrt(self.variance * self.dims / self.g_var_reg.intercept_)
        raise ArgumentError("The covariance is not fitted yet, use `auto_fit` or `fit` before use")

    def learning_rate(self, loss, grad_norm):
        """RFD learning rate from Random Function Descent paper"""
        # numerically stable version
        # (cf. https://en.wikipedia.org/wiki/Numerical_stability#Example)
        # to avoid catastrophic cancellation in the difference
        tmp = (self.mean - loss) / 2
        return (self.scale**2) / (
            torch.sqrt(tmp**2 + (self.scale * grad_norm) ** 2) + tmp
        )


if __name__ == "__main__":
    pass
    # dims = 2_300_000
    # fig, axs = cov.fit(df, dims)
    # fig.show()
