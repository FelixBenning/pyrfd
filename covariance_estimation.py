# Mean and Covariance estimation

import torch.nn.functional as F

from classification.mnist.data import MNIST, FashionMNIST
from classification.mnist.models import CNN3, CNN5, CNN7, AlgoPerf



def main():
    loss = F.nll_loss
    dataset = MNIST
    model = CNN3

if __name__ == "__main__":
    main()