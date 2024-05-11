""" CIFAR Data Modules """

import lightning as L
import torch
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import CIFAR10 as CIFAR10Dataset, CIFAR100 as CIFAR100Dataset
from torchvision import transforms

class CIFAR10(L.LightningDataModule):
    """Represents the CIFAR10 dataset.
    """

    def __init__(self, data_dir: str=" ./data", batch_size: int = 100) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.transform = transforms.Compose()

    def prepare_data(self) -> None:
        # download
        CIFAR10Dataset(self.data_dir, train=True, download=True)
        CIFAR10Dataset(self.data_dir, train=False, download=True)

    # pylint: disable=attribute-defined-outside-init
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar10_full = CIFAR10Dataset(self.data_dir, train=True)
            self.train_data, self.validation_data = random_split(
                cifar10_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = CIFAR10Dataset(
                self.data_dir, train=False
            )

        if stage == "predict":
            self.prediction_data = CIFAR10Dataset(
                self.data_dir, train=False
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.prediction_data, batch_size=self.batch_size)


class CIFAR100(L.LightningDataModule):
    """Represents the CIFAR10 dataset.
    """

    def __init__(self, data_dir: str=" ./data", batch_size: int = 100) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.transform = transforms.Compose()

    def prepare_data(self) -> None:
        # download
        CIFAR100Dataset(self.data_dir, train=True, download=True)
        CIFAR100Dataset(self.data_dir, train=False, download=True)

    # pylint: disable=attribute-defined-outside-init
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar100_full = CIFAR100Dataset(self.data_dir, train=True)
            self.train_data, self.validation_data = random_split(
                cifar100_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = CIFAR100Dataset(
                self.data_dir, train=False
            )

        if stage == "predict":
            self.prediction_data = CIFAR100Dataset(
                self.data_dir, train=False
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.prediction_data, batch_size=self.batch_size)
