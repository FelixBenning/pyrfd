import lightning as L
import torch
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import (
    MNIST as MNISTDataset,
    FashionMNIST as FashionMNISTDataset,
)
from torchvision import transforms


class MNIST(L.LightningDataModule):
    """Represents the MNIST dataset.

    source: [Example LightningDataModule](
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule)
    """

    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download
        MNISTDataset(self.data_dir, train=True, download=True)
        MNISTDataset(self.data_dir, train=False, download=True)

    # pylint: disable=attribute-defined-outside-init
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = MNISTDataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.validation_data = MNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = MNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.prediction_data = MNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.prediction_data, batch_size=self.batch_size)


class FashionMNIST(L.LightningDataModule):
    """Represents the MNIST dataset.

    source: [Example LightningDataModule](
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule)
    """

    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download
        FashionMNISTDataset(self.data_dir, train=True, download=True)
        FashionMNISTDataset(self.data_dir, train=False, download=True)

    # pylint: disable=attribute-defined-outside-init
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = FashionMNISTDataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.validation_data = FashionMNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = FashionMNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.prediction_data = FashionMNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.prediction_data, batch_size=self.batch_size)
