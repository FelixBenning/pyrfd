import torch

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100 as CIFAR100Dataset
from torchvision.transforms import v2
import lightning as L


class CIFAR100(L.LightningDataModule):
    """Represents the CIFAR10 dataset."""

    def __init__(self, data_dir: str = " ./data", batch_size: int = 100) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # cifar 100 has 60000 32x32 color images (600 images per class)
        cifar100_mean = (0.4914, 0.4822, 0.4465)
        cifar100_stddev = (0.2023, 0.1994, 0.2010)
        random_crop = v2.RandomCrop(
            size=32,
            padding=4,
            padding_mode="reflect",
        )
        horizontal_flip = v2.RandomHorizontalFlip(0.5)
        trivial_augment = v2.TrivialAugmentWide(
            interpolation=v2.InterpolationMode.BILINEAR
        )

        self.train_transforms = v2.Compose(
            [
                v2.ToImage(),
                random_crop,
                horizontal_flip,
                trivial_augment,
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(cifar100_mean, cifar100_stddev),
                v2.ToPureTensor(),
            ]
        )
        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(cifar100_mean, cifar100_stddev),
                v2.ToPureTensor(),
            ]
        )

    def prepare_data(self) -> None:
        # download
        CIFAR100Dataset(self.data_dir, train=True, download=True)
        CIFAR100Dataset(self.data_dir, train=False, download=True)

    # pylint: disable=attribute-defined-outside-init
    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended."""
        if stage == "fit":
            self.data_train = self._get_dataset(train=True)
            self.data_val = self._get_dataset(train=False)

        if stage == "validate":
            self.data_val = self._get_dataset(train=False)

        if stage == "test":
            self.data_test = self._get_dataset(train=False)

        if stage == "predict":
            self.data_predict = self._get_dataset(train=False)

    def _get_dataset(self, train: bool):
        if train:
            return CIFAR100Dataset(
                str(self.data_dir), train=True, transform=self.train_transforms
            )
        else:
            return CIFAR100Dataset(
                str(self.data_dir), train=False, transform=self.val_transforms
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)
