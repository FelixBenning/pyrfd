from ctypes import ArgumentError
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import optimize, stats


def sq_error_var(var_reg, b):
    return 3 * var_reg.predict((1/np.asarray(b)).reshape(-1, 1)) ** 2


DEFAULT_VAR_REG = LinearRegression()
DEFAULT_VAR_REG.intercept_ = 0.05
DEFAULT_VAR_REG.coef_ = np.array([1])


def intercept_variance(dist: stats.rv_discrete, sample_limit, var_reg):
    sample_count = sample_limit / dist.mean()
    theta = dist.expect(func=lambda x: 1 / sq_error_var(var_reg, x))
    w_1st_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * x))
    w_2nd_mom = dist.expect(func=lambda x: 1 / (sq_error_var(var_reg, x) * (x**2)))
    w_var = dist.expect(
        func=lambda x: 1 / (sq_error_var(var_reg, x)) * (1 / x - w_1st_mom / theta) ** 2
    )
    return w_2nd_mom / (sample_count * theta * w_var)


def batchsize_counts(sample_limit, var_reg=DEFAULT_VAR_REG):
    beta_0 = var_reg.intercept_
    beta_1 = var_reg.coef_[0]
    if beta_0 <= 0:
        raise ArgumentError("Theoretical Variance estimate is not positive")

    max_loc_guess = np.ceil(beta_1 / beta_0)


    cutoff = 20
    def gibbs_dist(w):
        # should really stop at infinite but cost...
        ks = np.arange(start=cutoff, stop=max_loc_guess + 500/w[1], step=1)

        logits = w[0] / sq_error_var(var_reg, ks) - w[1] * ks
        max_logit = np.max(logits)
        good_logits = logits - max_logit  # avoid under-/overflow of softmax
        probits = np.exp(good_logits)  # not quite probabilities

        probabilities = probits / np.sum(probits)
        return stats.rv_discrete(name="batchsizeDist", values=(ks, probabilities))

    
    res= optimize.minimize(
        lambda log_w: intercept_variance(gibbs_dist(np.exp(log_w)), sample_limit, var_reg),
        np.zeros(2),
        method="Nelder-Mead",
    )
    # w1s = np.exp2(np.linspace(-1, 1, 30))
    # w0s = np.exp2(np.linspace(-3, 1, 30))
    # X = np.empty((len(w0s), len(w1s)))
    # for idx0, w0 in enumerate(tqdm(w0s, desc="w0")):
    #     for idx1, w1 in enumerate(tqdm(w1s, desc="w1", leave=False)):
    #         X[idx0, idx1] = intercept_variance(gibbs_dist([w0, w1]), sample_limit, var_reg)
    # return X
    return res, gibbs_dist(np.exp(res.x))



if __name__ == "__main__":
    from matplotlib import pyplot as plt

    res, dist = batchsize_counts(10_000)
    x= range(20,100)
    plt.plot(x, dist.pmf(x), "ro", ms=12,mec="r")
    # X = batchsize_counts(1000)
    # plt.imshow(X, cmap="hot")
    # plt.show()
    pass
