""" Provide plotting functions for sanity check plots """

import pandas as pd
import numpy as np
from scipy import stats

from .regression import ScalarRegression

def selection(sorted_list, num_elts):
    """
    return a selection of num_elts from the sorted_list (evenly spaced in the index)
    always includes the first and last index
    """
    if len(sorted_list) < num_elts:
        return sorted_list
    idxs = np.round(np.linspace(0, len(sorted_list) - 1, num_elts)).astype(int)
    return sorted_list[idxs]


def plot_loss(ax, df: pd.DataFrame, *, mean, var_reg: ScalarRegression):
    """
    Plot Losses with mean, batch wise mean and variance based 95%-confidence interval
    under the Gaussian assumption.
    """
    ### SCATTER PLOT ###

    # limit to 100 samples per batchsize for performance
    reduced_df = df.groupby("batchsize").head(100).reset_index(drop=True)

    ax.scatter(
        1 / reduced_df["batchsize"],
        reduced_df["loss"],
        s=1,  # marker size
        label=r"$\mathcal{L}_b(w)$",
    )

    ### STATISTICAL PLOTS ###

    grouped_df = (
        df.groupby("batchsize", sort=True)
        .agg(
            loss_mean=pd.NamedAgg(column="loss", aggfunc="mean"),
        )
        .reset_index()
    )
    b_size_inv: np.array = 1 / grouped_df["batchsize"]

    # batchwise mean
    ax.plot(
        b_size_inv,
        grouped_df["loss_mean"],
        marker="*",
        label="batchwise mean",
    )

    # constant mean estimate
    ax.plot(
        b_size_inv,
        np.full_like(b_size_inv, fill_value=mean),
        label=rf"$\hat\mu={mean:.4}$",
    )

    # 95% - confidence interval (with Gaussian assumption)
    var_estimates = var_reg(b_size_inv)
    ax.fill_between(
        x=b_size_inv,
        y1=mean + stats.norm.ppf(0.025) * np.sqrt(var_estimates),
        y2=mean + stats.norm.ppf(0.975) * np.sqrt(var_estimates),
        alpha=0.3,
    )

    ### META INFORMATION ###

    ax.legend(loc="upper left")
    ax.set_xlabel("1/b")


def qq_plot_losses(ax, df: pd.DataFrame, *, batch_sizes=None):
    """QQ-Plot against the normal distribution of the Losses"""
    if batch_sizes is None:  # select batch size with most data
        bsize_counts = df["batchsize"].value_counts(sort=True, ascending=False)
        batch_sizes = [bsize_counts.index[0]]

    for batch_size in batch_sizes:
        filtered_df = df.groupby("batchsize", sort=True).get_group(batch_size)
        stats.probplot(filtered_df["loss"], dist="norm", plot=ax)
        ax.get_lines()[-2].set_label(f"b={batch_size}")

    for idx, line in enumerate(ax.get_lines()):
        line.set_color(f"C{idx//2}")
        if idx % 2 == 0:  # scatter
            line.set_markersize(1)
        if idx % 2 == 1:  # regression line
            line.set_linestyle("--")

    ax.set_title("")
    ax.set_xlabel("")
    ax.legend()


def plot_squared_losses(ax, df: pd.DataFrame, *, mean, var_reg):
    """Plot squared centered losses with variance and batch wise variance estimates and
    95% - chi2 based confidence interval estimate
    """
    ### SCATTER PLOT ###

    # limit to 100 samples per batchsize for performance
    reduced_df = df.groupby("batchsize").head(100).reset_index(drop=True)

    ax.scatter(
        1 / reduced_df["batchsize"],
        (reduced_df["loss"] - mean) ** 2,
        s=1,  # marker size
        label=r"$(\mathcal{L}_b(w)-\hat{\mu})^2$",
    )

    ### STATISTICAL PLOTS ###

    grouped_df = (
        df.groupby("batchsize", sort=True)
        .agg(
            loss_var=pd.NamedAgg(
                column="loss", aggfunc=lambda x: np.mean((x - mean) ** 2)
            ),
        )
        .reset_index()
    )
    b_size_inv: np.array = 1 / grouped_df["batchsize"]

    # batchwise mean squares
    ax.plot(
        b_size_inv,
        grouped_df["loss_var"],
        marker="*",
        label="batchwise mean squares",
    )

    # variance regression
    var_estimates = var_reg(b_size_inv)
    ax.plot(
        b_size_inv,
        var_estimates,
        label=r"Var$(\mathcal{L}_b(w))$",
    )

    # 95% - confidence intervals based on squared Gaussian (i.e. Chi2 distribution)
    ax.fill_between(
        x=b_size_inv,
        # χ^2 confidence bounds
        y1=var_estimates + (stats.chi2.ppf(0.025, df=1) - 1) * var_estimates,
        y2=var_estimates + (stats.chi2.ppf(0.975, df=1) - 1) * var_estimates,
        alpha=0.3,
    )

    ### META INFORMATION ###
    ax.legend(loc="upper left")
    ax.set_xlabel("1/b")
    ax.set_title("")


def qq_plot_squared_losses(ax, df: pd.DataFrame, *, mean, batch_sizes=None):
    """QQ-Plot of squared losses against chi2 distribution"""
    if batch_sizes is None:  # select batch size with most data
        bsize_counts = df["batchsize"].value_counts(sort=True, ascending=False)
        batch_sizes = [bsize_counts.index[0]]

    # QQ-plot against chi^2
    for batch_size in batch_sizes:
        filtered_df = df.groupby("batchsize", sort=True).get_group(batch_size)
        sq_losses = (filtered_df["loss"] - mean) ** 2
        stats.probplot(sq_losses, dist=stats.chi2(df=1), plot=ax)
        ax.get_lines()[-2].set_label(f"b={batch_size}")

    for idx, line in enumerate(ax.get_lines()):
        line.set_color(f"C{idx//2}")
        if idx % 2 == 0:  # scatter
            line.set_markersize(1)
        if idx % 2 == 1:  # regression line
            line.set_linestyle("--")

    ax.set_title("")
    ax.set_xlabel("")
    ax.legend()


def plot_gradient_norms(ax, df: pd.DataFrame, *, g_var_reg: ScalarRegression, dims):
    """Plot gradient norms, mean and batch wise means and 95% confidence interval
    based on sums of squared Gaussian i.e. chi2(dims) assumption
    """
    ### SCATTER PLOT ###

    # limit to 100 samples per batchsize for performance
    reduced_df = df.groupby("batchsize").head(100).reset_index(drop=True)

    ax.scatter(
        1 / reduced_df["batchsize"],
        reduced_df["grad_norm"] ** 2,
        s=1,  # marker size
        label=r"$\|\nabla\mathcal{L}_b(w)\|^2$",
    )

    ### STATISTICAL PLOTS ###

    df["sq_grad_norm"] = df["grad_norm"] ** 2
    grouped_df = (
        df.groupby("batchsize", sort=True)
        .agg(
            sq_grad_norm_mean=pd.NamedAgg(column="sq_grad_norm", aggfunc="mean"),
        )
        .reset_index()
    )
    b_size_inv: np.array = 1 / grouped_df["batchsize"]

    # batch wise mean of squared gradient norms
    ax.plot(
        b_size_inv,
        grouped_df["sq_grad_norm_mean"],
        marker="*",
        label="batchwise mean",
    )

    # holistic squared gradient norms
    sq_norm_means = g_var_reg(b_size_inv)
    ax.plot(
        b_size_inv,
        sq_norm_means,
        label=r"$\mathbb{E}[\|\nabla\mathcal{L}_b(w)\|^2]$",
    )

    # confidence intervals based on sums of squared Gaussians (i.e. chi2(dims))
    ax.fill_between(
        x=b_size_inv,
        y1=sq_norm_means + (stats.chi2.ppf(0.025, dims) - dims) * sq_norm_means / dims,
        y2=sq_norm_means + (stats.chi2.ppf(0.975, dims) - dims) * sq_norm_means / dims,
        alpha=0.3,
    )

    ### META INFORMATION ###
    ax.legend(loc="upper left")
    ax.set_xlabel("1/b")


def qq_plot_sq_gradient_norms(ax, df: pd.DataFrame, *, dims, batch_sizes=None):
    """QQ-Plot of squared gradient norms against chi2(dims) distribution"""

    if batch_sizes is None:  # select batch size with most data
        bsize_counts = df["batchsize"].value_counts(sort=True, ascending=False)
        batch_sizes = [bsize_counts.index[0]]

    df["sq_grad_norm"] = df["grad_norm"] ** 2
    # QQ-plot against chi^2
    for batch_size in batch_sizes:
        filtered_df = df.groupby("batchsize", sort=True).get_group(batch_size)
        stats.probplot(
            filtered_df["sq_grad_norm"],
            dist=stats.chi2(df=dims),
            plot=ax,
        )
        ax.get_lines()[-2].set_label(f"b={batch_size}")

    for idx, line in enumerate(ax.get_lines()):
        line.set_color(f"C{idx//2}")
        if idx % 2 == 0:  # scatter
            line.set_markersize(1)
        if idx % 2 == 1:  # regression line
            line.set_linestyle("--")

    ax.set_title("")
    ax.legend()
