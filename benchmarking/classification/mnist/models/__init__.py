""" Available models for MNIST dataset. """

from .cnn3 import CNN3
from .cnn5 import CNN5
from .cnn7 import CNN7
from .algo_perf import AlgoPerf

__all__ = ["CNN3", "CNN5", "CNN7", "AlgoPerf"]
