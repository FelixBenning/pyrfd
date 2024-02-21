from abc import abstractmethod
from logging import warning
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats

from .sampling import CachedSamples, IsotropicSampler


def selection(sorted_list, num_elts):
    """
    return a selection of num_elts from the sorted_list (evenly spaced in the index)
    always includes the first and last index
    """
    if len(sorted_list) < num_elts:
        return sorted_list
    idxs = np.round(np.linspace(0, len(sorted_list) - 1, num_elts)).astype(int)
    return sorted_list[idxs]


def fit_mean_var(batch_sizes: np.array, batch_losses: np.array, max_bootstrap=100):
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
            print(f"Bootstrapping WLS converged in {idx} iterations")
            return {"mean": mu, "var_regression": var_reg}

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    return {"mean": mu, "var_regression": var_reg}


def isotropic_derivative_var_estimation(
    batch_sizes: np.array, sq_grad_norms: np.array, max_bootstrap=100
) -> LinearRegression:
    batch_sizes = np.array(batch_sizes)
    b_inv: np.array = 1 / batch_sizes

    g_var_reg = LinearRegression(fit_intercept=True)
    g_var_reg.fit(
        np.array([[0], [1]]), np.array([0, 1])
    )  # initialize with beta0=1 and beta1=1

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
            print(f"Bootstrapping WLS converged in {idx} iterations")
            return g_var_reg

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    return g_var_reg



class CovarianceModel:
    __slots__ = "mean", "var_reg", "g_var_reg"

    @abstractmethod
    def learning_rate(self, loss, grad_norm):
        return NotImplemented

    def isotropic_fit(self, df: pd.DataFrame, dims):
        if ("sq_grad_norm" not in df) and ("grad_norm" in df):
            df["sq_grad_norm"] = df["grad_norm"] ** 2

        tmp = fit_mean_var(df["batchsize"], df["loss"])
        self.mean = tmp["mean"]
        self.var_reg: LinearRegression = tmp["var_regression"]

        self.g_var_reg: LinearRegression = isotropic_derivative_var_estimation(
            df["batchsize"], df["sq_grad_norm"]
        )

        # # non-parametric estimates of C(0) and -C'(0) for the Loss and sample error ϵ
        C0_L = self.var_reg.intercept_ * dims
        C0_eps = self.var_reg.coef_[0] * dims
        Cprime0_L = self.g_var_reg.intercept_
        Cprime0_eps = self.g_var_reg.coef_[0]

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
        b_size_selection = np.array(selection(b_size_grouped["batchsize"], 5))[1:]
        for bs in b_size_selection:
            stats.probplot(grouped.get_group(bs)["loss"], dist="norm", plot=axs[0, 1])
        for idx, line in enumerate(axs[0, 1].get_lines()):
            line.set_color(f"C{idx//2}")
            if idx % 2 == 0:  # scatter
                line.set_markersize(1)
                line.set_label(f"b={b_size_selection[idx//2]}")
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
        for bs in b_size_selection:
            stats.probplot(
                (grouped.get_group(bs)["loss"] - self.mean) ** 2,
                dist=stats.chi2(df=1),
                plot=axs[1, 1],
            )
        for idx, line in enumerate(axs[1, 1].get_lines()):
            line.set_color(f"C{idx//2}")
            if idx % 2 == 0:  # scatter
                line.set_markersize(1)
                line.set_label(f"b={b_size_selection[idx//2]}")
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

        for bs in b_size_selection:
            stats.probplot(
                grouped.get_group(bs)["sq_grad_norm"],
                dist=stats.chi2(df=dims),
                plot=axs[2, 1],
            )
        for idx, line in enumerate(axs[2, 1].get_lines()):
            line.set_color(f"C{idx//2}")
            if idx % 2 == 0:  # scatter
                line.set_markersize(1)
                line.set_label(f"b={b_size_selection[idx//2]}")
            if idx % 2 == 1:
                line.set_linestyle("--")
        axs[2, 1].set_title("")
        axs[2, 1].legend()

        return {
            "mean": self.mean,
            "var": {"theo_loss": C0_L, "sample_error": C0_eps},
            "var_derivative": {"theo_loss": Cprime0_L, "sample_error": Cprime0_eps},
            "sanity_check_plots": (fig, axs),
        }


class SquaredExponential(CovarianceModel):
    __slots__ = "scale", "variance"

    def __init__(self, scale, mean=0, variance=1):
        self.mean = mean
        self.variance = variance
        self.scale = scale

    def auto_fit(self, model_factory, loss, data, cache=None, tol=1e-3):
        dims = sum(p.numel() for p in model_factory().parameters() if p.requires_grad)
        df = pd.DataFrame()
        sampler = IsotropicSampler(model_factory, loss, data)

        cached_samples = CachedSamples(cache)
        rel_error = 1
        while rel_error > tol:
            b_size_counts = {} # TODO: batchsize sampling

            sampler.sample(b_size_counts, cached_samples)
            df = pd.DataFrame(cached_samples.records)

            self.fit(df, dims)

    def fit(self, df: pd.DataFrame, dims):
        res = super().isotropic_fit(df, dims)
        self.mean = res["mean"]
        self.variance = res["var"]["theo_loss"]
        # σ²_ϵ = result.var.sample_error
        self.scale = np.sqrt(self.variance / res["var_derivative"]["theo_loss"])
        # s_ϵ = sqrt(σ²_ϵ / result.var_derivative.sample_error)

        return res["sanity_check_plots"]

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
    df = pd.read_csv("mnistSimpleCNN/data/loss_samples.csv")
    df = df[df["trainmode"] == False]
    cov = SquaredExponential(1)
    dims = 2_300_000
    fig, axs = cov.fit(df, dims)
    fig.show()