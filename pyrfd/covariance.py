from abc import abstractmethod
import torch


class CovarianceModel:
    __slots__ = "mean", "variance"

    def __init__(self):
        pass

    @abstractmethod
    def learning_rate(self, loss, grad_norm):
        return NotImplemented


class SquaredExponential(CovarianceModel):
    __slots__ = "scale"

    def __init__(self, scale, mean=0, variance=1):
        self.mean = mean
        self.variance = variance
        self.scale = scale

    def learning_rate(self, loss, grad_norm):
        """RFD learning rate from Random Function Descent paper"""
        tmp = (self.mean - loss) / 2
        return (self.scale**2) / (
            torch.sqrt(tmp**2 + (self.scale * grad_norm) ** 2) + tmp
        )
