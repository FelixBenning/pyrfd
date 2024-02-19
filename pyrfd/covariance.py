from abc import abstractmethod
from logging import warning
import math
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
import numpy as np
import torch

def fit_mean_var(self, batch_sizes, batch_losses, max_bootstrap=100):
    b_inv = [1 / b for b in batch_sizes]

    # we initialize with the assumption that the variance at batch_size=Inf is zero
    var_reg = LinearRegression(fit_intercept=True)
    var_reg.fit([0, 1], [0, 1])  # initialize with beta0=1 and beta1=1

    # bootstrapping Weighted Least Squares (WLS)
    for idx in range(max_bootstrap):
        vars = [var_reg.predict(x) for x in b_inv]  # variance at batchsizes 1/b in X

        mu = np.average(batch_losses, [1 / v for v in vars])
        centered_squares = [(Lb - mu) ** 2 for Lb in batch_losses]

        old_intercept = var_reg.intercept_
        # fourth moments i.e. 3*sigma^4 = 3 * var^2 are the variance of the centered
        # squares, the weights should be 1/these variances
        # (we leave out the 3 as it does not change the relative weights)
        var_reg.fit(b_inv, centered_squares, sample_weight=[1 / v**2 for v in vars])

        if math.isclose(old_intercept, var_reg.intercept_):
            print(f"Bootstrapping WLS converged in {idx} iterations")
            self.mean = mu
            self.var_reg = var_reg
            return {"mean": mu, "var_regression": var_reg}

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    self.mean = mu
    self.var_reg = var_reg
    return {"mean": mu, "var_regression": var_reg}

def isotropic_derivative_var_estimation(self, batch_sizes, sq_grad_norms, max_bootstrap=100) -> LinearRegression:
    b_inv = [1 / b for b in batch_sizes]

    g_var_reg = LinearRegression(fit_intercept=True)
    g_var_reg.fit([0, 1], [0, 1])  # initialize with beta0=1 and beta1=1

    # bootstrapping WLS
    for idx in range(max_bootstrap):
        vars = [g_var_reg.predict(x) for x in b_inv] # variances at batchsize 1/b

        # squared grad norms are already (iid) sums of squared Gaussians
        # variance of squares is 3Var^2 but the 3 does not matter as it cancels
        # out in the weighting we also have a sum of squares (norm), but this
        # also only results in a constant which does not matter
        old_bias = g_var_reg.intercept_
        g_var_reg.fit(b_inv, sq_grad_norms, [1/v**2 for v in vars])

        if math.isclose(old_bias, g_var_reg.intercept_):
            print(f"Bootstrapping WLS converged in {idx} iterations")
            self.g_var_reg = g_var_reg
            return g_var_reg

    warning(f"Bootstrapping WLS did not converge in max_bootstrap={max_bootstrap}")
    return g_var_reg

class CovarianceModel:
    __slots__ = "mean", "var_reg", "g_var_reg"

    @abstractmethod
    def learning_rate(self, loss, grad_norm):
        return NotImplemented
    
    def isotropic_cov_at_zero(self, df:DataFrame, dims):
        if ("sq_grad_norm" not in df) and ("grad_norm" in df):
            df["sq_grad_norm"] = df["grad_norm"] ** 2

        tmp =  fit_mean_var(df["batchsize"], df["loss"])
        self.mean = tmp["mean"]
        self.var_reg:LinearRegression = tmp["var_regression"]

        self.g_var_reg:LinearRegression = isotropic_derivative_var_estimation(df["batchsize"], df["sq_grad_norm"])

        # # non-parametric estimates of C(0) and -C'(0) for the Loss and sample error ϵ
        C0_L = self.var_reg.intercept_ * dims
        C0_eps = self.var_reg.coef_[0] * dims
        Cprime0_L = self.g_var_reg.intercept_
        Cprime0_eps = self.g_var_reg.coef_[0]


        ## ============================
        ### === sanity check plots ===
        ## ============================
        # b_size_grouped = DF.combine(
        #     DF.groupby(df, :batchsize, sort=true),
        #     :loss => Stat.mean,
        #     :loss => (x -> Stat.varm(x, μ)) => :loss_var,
        #     :sq_grad_norm => Stat.mean,
        # )
        # b_size_inv = 1 ./ b_size_grouped[:, :batchsize]
        # var_estimates = [var_reg([x]) for x in b_size_inv]

        # ## ==== Plot Losses =====
        # reduced_df = DF.combine(
        #     DF.groupby(df, :batchsize),
        #     # only select <100 examples per batch size
        #     DF.All() .=> (x->first(x,100)) .=> DF.All()
        # )

        # plt_losses = plot(
        #     1 ./ reduced_df[:, :batchsize], reduced_df[:, :loss],
        #     seriestype=:scatter,
        #     xlabel="1/b",
        #     label=L"\mathcal{L}_b(w)",
        #     markersize=1,
        #     legend=:topleft,
        #     xaxis=:log,
        #     # xticks=([0.01, 0.1, 0.25, 1/2, 1], ["1/100", "1/10", "1/4", "1/2", 1]),
        # )
        # plot!(
        #     plt_losses,
        #     b_size_inv, b_size_grouped[:, :loss_mean],
        #     markershape=:cross,
        #     label="batchwise mean"
        # )
        # plot!(
        #     plt_losses,
        #     b_size_inv, μ .* ones(length(b_size_inv)),
        #     ribbon= (
        #         -Dist.quantile.(Dist.Normal.(0, sqrt.(var_estimates)), 0.025),
        #         Dist.quantile.(Dist.Normal.(0, sqrt.(var_estimates)), 0.975)
        #     ),
        #     fillalpha=0.3,
        #     label=L"\hat{μ}=%$(round(μ, sigdigits=4))"
        # )

        # ## === Plot squared errors =====
        # plt_squares = plot(
        #     1 ./ reduced_df[:, :batchsize], map(Lb->(Lb - μ)^2, reduced_df[:, :loss]),
        #     seriestype=:scatter,
        #     # xaxis=:log,
        #     xlabel="1/b",
        #     label=L"(\mathcal{L}_b(w)-\hat{\mu})^2",
        #     markersize=1,
        #     # xticks=([0.1, 0.25, 1/2, 1], ["1/10", "1/4", "1/2", 1]),
        #     # fontfamily="Computer Modern",
        # );
        # plot!(
        #     plt_squares,
        #     b_size_inv,
        #     b_size_grouped[:,:loss_var],
        #     markershape=:cross,
        #     label="batchwise mean squares"
        # )

        # # χ^2 confidence bounds
        # lower_chisq(var) = (1-Dist.quantile(Dist.Chisq(1), 0.025)) * var
        # upper_chisq(var) = (Dist.quantile(Dist.Chisq(1), 0.975)-1) * var

        # plot!(
        #     plt_squares,
        #     b_size_inv, var_estimates,
        #     ribbon= (lower_chisq.(var_estimates), upper_chisq.(var_estimates)),
        #     fillalpha=0.3,
        #     label=L"Var($\mathcal{L}_b(w)$)",
        # )

        # # ==== Plot Gradient Norms ====
        # plt_grad_norms = plot(
        #     1 ./ reduced_df[:,:batchsize], reduced_df[:, :sq_grad_norm],
        #     seriestype=:scatter,
        #     markersize=1,
        #     xlabel="1/b",
        #     label=L"\|\nabla\mathcal{L}_b(w)\|^2",
        #     legend=:topleft,
        # )
        # plot!(
        #     plt_grad_norms,
        #     b_size_inv,
        #     b_size_grouped[:,:sq_grad_norm_mean],
        #     markershape=:cross,
        #     label="batchwise mean",
        # )

        # sq_norm_means = [sqg_norm_reg([x]) for x in b_size_inv]
        # lower_sq_norms(sqn_mean) = (dims - Dist.quantile(Dist.Chisq(dims),0.025)) * sqn_mean / dims
        # upper_sq_norms(sqn_mean) = (Dist.quantile(Dist.Chisq(dims), 0.975) - dims) * sqn_mean / dims

        # plot!(
        #     plt_grad_norms,
        #     b_size_inv, sq_norm_means,
        #     ribbon= (lower_sq_norms.(sq_norm_means), upper_sq_norms.(sq_norm_means)),
        #     fillalpha=0.3,
        #     label=L"$\mathbb{E}[\|\nabla\mathcal{L}_b(w)\|^2]$)",
        # )
        # # TODO: figure out wtf is wrong with the confidence interval of the squared norm

        # return (
        #     mean = μ,
        #     var = (theo_loss=C0_L, sample_error=C0_ϵ),
        #     var_derivative = (theo_loss=Cprime0_L, sample_error=Cprime0_ϵ),
        #     sanity_check_plots= (plt_losses, plt_squares, plt_grad_norms)
        # )




class SquaredExponential(CovarianceModel):
    __slots__ = "scale"

    def __init__(self, scale, mean=0, variance=1):
        self.mean = mean
        self.variance = variance
        self.scale = scale

    

    def learning_rate(self, loss, grad_norm):
        """RFD learning rate from Random Function Descent paper"""
        # numerically stable version
        # (cf. https://en.wikipedia.org/wiki/Numerical_stability#Example)
        # to avoid catastrophic cancellation in the difference
        tmp = (self.mean - loss) / 2
        return (self.scale**2) / (
            torch.sqrt(tmp**2 + (self.scale * grad_norm) ** 2) + tmp
        )
