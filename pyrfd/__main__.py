"""
Example usage of the RFD module
"""

import torch
import torchvision as tv
from mnistSimpleCNN.models.modelM3 import ModelM3

from pyrfd import RFD, SquaredExponential, sampling

cov_model = SquaredExponential()
cov_model.auto_fit(
    model_factory=ModelM3,
    loss=torch.nn.functional.nll_loss,
    data=tv.datasets.MNIST(
        root="mnistSimpleCNN/data",
        train=True,
        transform=tv.transforms.ToTensor(),
    ),
    cache="cache/CNN3_mnist.csv",
)
cached_samples = sampling.CSVSampleCache("cache/CNN3_mnist.csv")
cov_model.plot_sanity_checks(cached_samples.as_dataframe())

rfd = RFD(ModelM3().parameters(), covariance_model=cov_model)


print(cov_model.scale)
