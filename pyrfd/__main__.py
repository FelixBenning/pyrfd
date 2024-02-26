import torch
import torchvision as tv
from mnistSimpleCNN.models.modelM3 import ModelM3

from pyrfd import RFD, SquaredExponential

cov_model = SquaredExponential()
cov_model.auto_fit(
    model_factory=ModelM3,
    loss=torch.nn.functional.nll_loss,
    data= tv.datasets.MNIST(
        root="mnistSimpleCNN/data",
        train=True,
        transform=tv.transforms.ToTensor()
    ),
    cache="cache/CNN3_mnist.csv",
)
rfd = RFD(
    ModelM3().parameters(),
    covariance_model=cov_model
)
print(cov_model.scale)