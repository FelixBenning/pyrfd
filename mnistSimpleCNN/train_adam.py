from torch import optim, nn
from torch.utils.data import DataLoader
import torchvision as tv
import torch.nn.functional as F
import lightning as L
import torchmetrics.functional.classification as metrics

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
        # acc = metrics.multilabel_accuracy(prediction, target)
        # self.log("train_accuracy", acc, on_epoch=True)
        return loss_value
    
    def test_step(self, batch):
        input, target = batch
        prediction = self.model(input)
        loss_value = self.loss(prediction, target)
        self.log("test_loss", loss_value, on_step=True, on_epoch=True)

        # acc = metrics.multilabel_accuracy(prediction, target)
        # self.log("test_accuracy", acc, on_epoch=True)

        # rec = metrics.multilabel_recall(prediction, target)
        # self.log("test_recall", rec, on_epoch=True)
        
        # prec = metrics.multilabel_precision(prediction, target)
        # self.log("test_precision", prec, on_epoch=True)

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
    test_loader = DataLoader(
        tv.datasets.MNIST(
            root="mnistSimpleCNN/data",
            train=False,
            transform=tv.transforms.ToTensor(),
        ),
        batch_size=120,
        shuffle=False,
    )

    model = ClassicOptimizerTraining(ModelM3())
    trainer.fit(model=model, train_dataloaders=train_loader)
    trainer.test(model=model, dataloaders=test_loader)

if __name__ == "__main__":
    run()
