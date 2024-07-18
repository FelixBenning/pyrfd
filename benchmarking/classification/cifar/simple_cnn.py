from torch.nn import Conv2d, ReLU, Module, Linear, Flatten, Sequential


class ConvBlock(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding="same")
        self.relu = ReLU()
        self.conv2 = Conv2d(channels, channels, kernel_size=3, padding="same")

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class SimpleCNN(Sequential):
    def __init__(
        self, in_channels: int = 3, hidden_channels: int = 32, num_classes: int = 100
    ):
        super().__init__(
            Conv2d(in_channels, hidden_channels, kernel_size=3, padding="same"),
            ConvBlock(hidden_channels),
            Flatten(),
            Linear(hidden_channels * 32 * 32, 256),
            Linear(256, num_classes),
        )
