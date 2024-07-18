from torch.nn import (
    Conv2d,
    Linear,
    MaxPool2d,
    ReLU,
    Flatten,
    Sequential,
    BatchNorm1d,
    BatchNorm2d,
)


class SimplestCNNBNEverywhere(Sequential):
    def __init__(self):
        super().__init__(
            Conv2d(3, 32, kernel_size=3, padding="same"),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # [32, 16, 16]
            Conv2d(32, 64, kernel_size=3, padding="same"),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # [64, 8, 8]
            Conv2d(64, 64, kernel_size=3, padding="same"),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # [64, 4, 4]
            Flatten(),
            Linear(64 * 4 * 4, 256),
            ReLU(),
            Linear(256, 100),
            BatchNorm1d(100),
        )
