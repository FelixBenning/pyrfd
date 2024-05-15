from numpy import repeat
import torch.nn.functional as F
import pandas as pd

from matplotlib import rc
import matplotlib.pyplot as plt

from benchmarking.classification.mnist.models import CNN7
from benchmarking.classification.mnist.data import MNIST
from pyrfd import sampling, covariance

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

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
plt.savefig("plot/covariance_fit.pdf")
plt.show()
