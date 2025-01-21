import lightning as L
from torch.utils.data import DataLoader

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

        # # Possible data augmentation
        # self.train_transform = transforms.Compose(
        #     [
        #         transforms.RandomRotation(20),
        #         transforms.RandomAffine(0, translate=(0.2, 0.2)),
        #         self.transform,
        #     ]
        # )


    def prepare_data(self):
        # download
        MNISTDataset(self.data_dir, train=True, download=True)
        MNISTDataset(self.data_dir, train=False, download=True)

    # pylint: disable=attribute-defined-outside-init
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = MNISTDataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.data_val = MNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = MNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.data_predict = MNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)


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
            self.data_train = FashionMNISTDataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.data_val = FashionMNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = FashionMNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.data_predict = FashionMNISTDataset(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)
