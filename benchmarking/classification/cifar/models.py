from torch import nn
from torchvision.models import resnet18


class Resnet18(nn.Module):
    """ Resnet18 model for CIFAR-100 (modifactions based on FOB benchmark) """
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=100, pretrained=False)
        # 7x7 conv is too large for 32x32 images
        self.model.conv1 = nn.Conv2d(
            in_channels=3, # rgb color
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=4,
            padding_mode="reflect",
        )
        # pooling small images is bad
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


