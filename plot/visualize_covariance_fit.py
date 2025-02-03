from numpy import repeat
import torch.nn.functional as F
import pandas as pd

from matplotlib import rc
import matplotlib.pyplot as plt

from benchmarking.classification.mnist.models import CNN7
from benchmarking.classification.mnist.data import MNIST
from pyrfd import sampling, covariance, plots

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def visualize_fit():
    mnist = MNIST(batch_size=100)
    mnist.prepare_data()
    mnist.setup("fit")

    cov_model = covariance.IsotropicCovariance()

    sampler = sampling.IsotropicSampler(
        model_factory = CNN7,
        loss= F.nll_loss,
        data= mnist.data_train,
        cache= "MNIST_CNN7_oversampled.csv"
    )

    if len(sampler) == 0:
        b_sizes = range(10, 100, 10)
        sampler.sample(bsize_counts= pd.Series(
            data=repeat(600, len(b_sizes)),
            index= b_sizes
        ))
        b_sizes = range(100, 1001, 100)
        sampler.sample(bsize_counts= pd.Series(
            data=[60_000 // b_size for b_size in b_sizes],
            index= b_sizes
        ))

    cov_model.dims = sampler.dims
    cov_model.fit(sampler.snapshot_as_dataframe())

    print(f"asymptotic lr={cov_model.asymptotic_learning_rate()}")

    (fig, axs) = cov_model.plot_sanity_checks(
        sampler.snapshot_as_dataframe(),
        batch_sizes=[10,100,1000]
    )
    fig.tight_layout()
    fig.savefig("plot/MNIST_CNN7_covariance_fit.pdf")

    fig_sq_regression, ax_sq_regression = plt.subplots(figsize=(6, 4))
    plots.plot_squared_losses(
        ax_sq_regression,
        sampler.snapshot_as_dataframe(),
        mean= cov_model.mean,
        var_reg= cov_model.var_reg
    )
    fig_sq_regression.savefig("plot/MNIST_CNN7_squared_regression.pdf")




def get_run_covariance_and_cost(run):
    mnist = MNIST(batch_size=100)
    mnist.prepare_data()
    mnist.setup("fit")

    cov_model = covariance.IsotropicCovariance()
    sampler = sampling.IsotropicSampler(
        model_factory = CNN7,
        loss= F.nll_loss,
        data= mnist.data_train,
        cache=f"cache/MNIST/CNN7_run={run}/covariance_cache.csv"
    )
    cov_model.dims = sampler.dims
    cov_model.fit(sampler.snapshot_as_dataframe())
    return cov_model, sampler.sample_cost

def visualize_lr_variance():
    asympt_lr = []
    sample_costs = []
    for run in range(20):
        cov_model, sample_cost = get_run_covariance_and_cost(run)
        if sample_cost > 100000:
            print(f"run: {run}, sample_cost: {sample_cost}, asymptotic_lr: {cov_model.asymptotic_learning_rate()}")
        asympt_lr.append(cov_model.asymptotic_learning_rate())
        sample_costs.append(sample_cost)

    # print(asympt_lr)
    plt.figure(figsize=(4, 3))
    plt.hist(asympt_lr)
    plt.xlabel("Asymptotic learning rate")
    plt.tight_layout()
    plt.savefig("plot/MNIST_CNN7_asymptotic_lr.pdf")

    print(f"less_one_epoch: {len([cost for cost in sample_costs if cost < 60_000])}")
    plt.figure(figsize=(4, 3))
    plt.hist(sample_costs, label="Sample cost")
    plt.xlabel("Sample cost")
    plt.tight_layout()
    plt.savefig("plot/MNIST_CNN7_sample_cost.pdf")


if __name__ == "__main__":
    visualize_lr_variance()
    visualize_fit()
