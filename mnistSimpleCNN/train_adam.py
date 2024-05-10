from torch import optim, nn
from torch.utils.data import DataLoader
import torchvision as tv
import torch.nn.functional as F
import lightning as L

from mnistSimpleCNN.models.modelM3 import ModelM3


class ClassicOptimizerTraining(L.LightningModule):
    def __init__(self, model, optimizer=optim.Adam, loss=F.nll_loss):
        super().__init__()
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        input, target = batch
        prediction = self.model(input)
        loss_value = self.loss(prediction, target)
        self.log("train_loss", loss_value)
        return loss_value

    def configure_optimizers(self):
        optim = self.optimizer(self.parameters())
        return optim


def run():
    trainer = L.Trainer(max_epochs=2)

    train_loader = DataLoader(
        tv.datasets.MNIST(
            root="mnistSimpleCNN/data",
            train=True,
            transform=tv.transforms.ToTensor(),
        ),
        batch_size=120,
        shuffle=True,
    )
    model = ClassicOptimizerTraining(ModelM3())
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    run()
