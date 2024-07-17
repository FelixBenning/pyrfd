""" AlgoPerf model for MNIST classification task. https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/mnist/mnist_pytorch/workload.py"""
from torch import nn
from torch.nn import functional as F

class AlgoPerf(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        input_size = 28 * 28
        num_hidden = 128
        num_classes = 10
        self.net = nn.Sequential(
            nn.Linear(input_size, num_hidden, bias=True),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_classes, bias=True),
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return F.log_softmax(self.net(x))
