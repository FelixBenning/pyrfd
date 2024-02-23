from ctypes import ArgumentError
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import optimize, stats
import pandas as pd
from time import time


def sq_error_var(var_reg, b):
    return 3 * var_reg.predict((1/np.asarray(b)).reshape(-1, 1)) ** 2


DEFAULT_VAR_REG = LinearRegression()
DEFAULT_VAR_REG.intercept_ = 0.05
DEFAULT_VAR_REG.coef_ = np.array([1])

CUTOFF = 20 # no batch-sizes below

def intercept_variance(dist: stats.rv_discrete, var_reg):
    theta = dist.expect(func=lambda x: 1 / sq_error_var(var_reg, x))
    w_1st_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * x))
    w_2nd_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * (x**2)))
    return (dist.mean() * w_2nd_mom) / (theta * w_2nd_mom - w_1st_mom ** 2)


def batchsize_dist(var_reg=DEFAULT_VAR_REG):
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
    
    print("Optimizing over batchsize distribution") 
    start = time()
    res= optimize.minimize(
        lambda log_w: intercept_variance(gibbs_dist(np.exp(log_w)), var_reg),
        np.log(weights), 
        method="Nelder-Mead",
        callback=lambda x: print(f"> current parameters: {np.exp(x)})", end="\r"),
    )
    end = time()
    weights = np.exp(res.x)
    print(f"> Final batchsize distribution parameters: {weights}")
    print(f"> {res.message}")
    print(f"> Time Elapsed: {end-start:.0f} (seconds)")

    
    return gibbs_dist(weights)

def batchsize_counts(budget, var_reg=DEFAULT_VAR_REG, existing_b_size_samples:pd.Series=pd.Series()):
    spent_budget = sum([b * count for b, count in existing_b_size_samples.items()])
    total = spent_budget + budget
    optimal_dist:stats.rv_discrete = batchsize_dist(var_reg)

    a, b = optimal_dist.support()
    support = range(int(a), int(b) + 1)
    df = pd.DataFrame(
        index=support,
        data = {"desired_dist": optimal_dist.pmf(support)}
    )
    df["desired_counts"] = df["desired_dist"] * total
    df = df.join(existing_b_size_samples.to_frame("existing_counts")).fillna(0)
    df["required_counts"] = df["desired_counts"] - df["existing_counts"]
    df["required_distribution"] = df["required_counts"] / budget

    required_dist = stats.rv_discrete(values=(df.index, df["required_distribution"].to_numpy()))

    
    est_sample_num = np.ceil(budget/required_dist.mean()).astype(int)
    b_size_samples = required_dist.rvs(size= est_sample_num + 500, random_state=1)
    last_idx = np.searchsorted(np.cumsum(b_size_samples), budget)

    required_counts = pd.Series(b_size_samples[:last_idx]).value_counts(sort=False).sort_index()
    return required_counts



if __name__ == "__main__":
    print(batchsize_counts(10_000))
    # x= range(20,100)
    # plt.plot(x, dist.pmf(x), "ro", ms=12,mec="r")
    # X = batchsize_counts(1000)
    # plt.imshow(X, cmap="hot")
    # plt.show()
    # pass
