"""
Module for optimally sampling batch sizes for better variance estimates.
"""

from time import time
from ctypes import ArgumentError
import numpy as np
from scipy import optimize, stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .sampling import _budget

# "arbitrary" design
DEFAULT_VAR_REG = LinearRegression()
DEFAULT_VAR_REG.intercept_ = 0.05  # should be greater zero - cf. sampling
DEFAULT_VAR_REG.coef_ = np.array([1])

CUTOFF = 20  # no batch-sizes below


def sq_error_var(var_reg, b):
    """calculate the 4th moment (for Gaussian rvs) i.e. variance of centered
    squares, where the variance regression determines their variance for
    different batchsizes"""
    return 3 * var_reg.predict((1 / np.asarray(b)).reshape(-1, 1)) ** 2


def empirical_intercept_variance(counts, var_reg):
    """TODO: change or compare to a bootstrap version, this assumes theory applicable"""
    n = sum(counts)
    dist = stats.rv_discrete(
        name="epirical batchsize distribution",
        values=(counts.index.to_numpy(), (counts / n).to_numpy()),
    )
    theta = dist.expect(func=lambda x: 1 / sq_error_var(var_reg, x))
    w_1st_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * x))
    w_2nd_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * (x**2)))
    return w_2nd_mom / (n * (theta * w_2nd_mom - w_1st_mom**2))


def limit_intercept_variance(dist: stats.rv_discrete, var_reg):
    """limiting variance as sample budget grows to infinity, this assumes the
    number of batch size samples n is related to the buget in the following way:
        n E[B] < budget
    where B is the random batch size sampled from dist. Thus n = budget/E[B],
    and we can replace 1/n by E[B]/budget in the formula. Since we let budget to
    infinity, we remove the budget to get an asymptotic/convergent formulation.
    """
    theta = dist.expect(func=lambda x: 1 / sq_error_var(var_reg, x))
    w_1st_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * x))
    w_2nd_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * (x**2)))
    return (dist.mean() * w_2nd_mom) / (theta * w_2nd_mom - w_1st_mom**2)


def batchsize_dist(var_reg=DEFAULT_VAR_REG, logging=False):
    """Find the optimal batch size distribution (in a Gibbs distribution class)
    under the assumption that the variance regression var_reg is true. The Gibbs
    distribution class is of the form

        p(b) = exp(w[0]/(3*var_reg.predict(1/b)**2) - w[1]b)

    and optimized over `w`
    """
    beta_0 = var_reg.intercept_
    beta_1 = var_reg.coef_[0]
    if beta_0 <= 0:
        raise ArgumentError("Theoretical Variance estimate is not positive")

    max_loc_guess = np.ceil(beta_1 / beta_0)

    def gibbs_dist(w):
        # should really stop at infinite but cost...
        ks = np.arange(start=CUTOFF, stop=max_loc_guess + 1000, step=1)

        logits = w[0] / sq_error_var(var_reg, ks) - w[1] * ks
        max_logit = np.max(logits)
        good_logits = logits - max_logit  # avoid under-/overflow of softmax
        probits = np.exp(good_logits)  # not quite probabilities

        probabilities = probits / np.sum(probits)
        return stats.rv_discrete(name="batchsizeDist", values=(ks, probabilities))

    # solution for DEFAULT_VAR_REG
    weights = np.array([1.78032054e-16, 1.53346666e-02])

    # early return
    if var_reg == DEFAULT_VAR_REG:
        return gibbs_dist(weights)

    if logging:
        tqdm.write("Optimizing over batchsize distribution using Nelder-Mead")

        def callback(x):
            tqdm.write(
                f"> current parameters: {np.exp(x)})                        ", end="\r"
            )

    else:

        def callback(_):
            pass

    start = time()
    res = optimize.minimize(
        lambda log_w: limit_intercept_variance(gibbs_dist(np.exp(log_w)), var_reg),
        np.log(weights),
        method="Nelder-Mead",
        callback=callback,
    )
    end = time()
    weights = np.exp(res.x)
    if logging:
        tqdm.write(
            f"> Final batchsize distribution parameters: {weights}                       "
        )
        tqdm.write(f"> {res.message}")
        tqdm.write(f"> Time Elapsed: {end-start:.0f} (seconds)")

    return gibbs_dist(weights)


def batchsize_counts(
    budget, var_reg=DEFAULT_VAR_REG, existing_b_size_samples: pd.Series = pd.Series()
):
    """Determines the optimal batchsize distribution (in a Gibbs distribution class),
    then adjusts the distribution for existing batchsize samples. (i.e. sample fewer
    from the existing sizes), such that after `budget` is spent, the distribution
    should be as close as possible to the optimal distribution. Then samples from
    this adjusted distribution.

    return a series with counts of batchsizes (where the batchsizes are the index).
    """
    spent_budget = _budget(existing_b_size_samples)
    total = spent_budget + budget
    optimal_dist: stats.rv_discrete = batchsize_dist(var_reg)

    a, b = optimal_dist.support()
    support = range(int(a), int(b) + 1)
    df = pd.DataFrame(index=support, data={"desired_dist": optimal_dist.pmf(support)})
    df["desired_counts"] = df["desired_dist"] * total / optimal_dist.mean()

    pd.set_option("future.no_silent_downcasting", True)  # remove warning
    df = df.join(existing_b_size_samples.to_frame("existing_counts")).fillna(0)

    df["required_counts"] = df["desired_counts"] - df["existing_counts"]
    req_cts = df[df["required_counts"] > 0]["required_counts"]
    df["required_distribution"] = req_cts / req_cts.sum()
    df["required_distribution"] = df["required_distribution"].fillna(0)

    required_dist = stats.rv_discrete(
        values=(df.index, df["required_distribution"].to_numpy())
    )

    b_size_samples = required_dist.rvs(
        # estimate required samples and add padding
        size=np.ceil(budget / required_dist.mean()).astype(int) + 500,
        random_state=int(time()),
    )
    last_idx = np.searchsorted(np.cumsum(b_size_samples), budget) + 1

    required_counts = (
        pd.Series(b_size_samples[:last_idx]).value_counts(sort=False).sort_index()
    )
    return required_counts


if __name__ == "__main__":
    tqdm.write(batchsize_counts(10_000))
    # x= range(20,100)
    # plt.plot(x, dist.pmf(x), "ro", ms=12,mec="r")
    # X = batchsize_counts(1000)
    # plt.imshow(X, cmap="hot")
    # plt.show()
    # pass
