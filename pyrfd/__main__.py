import torch
from mnistSimpleCNN.datasets import MnistDataset
from mnistSimpleCNN.models.modelM3 import ModelM3

from .covariance import SquaredExponential

cov = SquaredExponential().auto_fit(
    model_factory=ModelM3,
    loss=torch.nn.functional.nll_loss,
    data=MnistDataset(training=True, transform=None),
    cache="cache/CNN3_mnist.csv",
)
