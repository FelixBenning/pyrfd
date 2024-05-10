from __future__ import annotations
from typing import Callable, Any
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision as tv
import torch.nn.functional as F
import lightning as L
import torchmetrics.functional.classification as metrics

from mnistSimpleCNN.models.modelM3 import ModelM3


class Classifier(L.LightningModule):
    def __init__(self, model, optimizer=optim.Adam):
        super().__init__()
        self.optimizer = optimizer
        self.model = model

    def training_step(self, batch, batch_idx):
        x_in, y_out = batch
        prediction: torch.Tensor = self.model(x_in)
        loss_value = F.nll_loss(prediction, y_out)
        self.log("train_loss", loss_value, prog_bar=True)
        acc = metrics.multiclass_accuracy(prediction, y_out, prediction.size(dim=1))
        self.log("train_accuracy", acc, on_epoch=True)
        return loss_value

    def test_step(self, batch):
        x_in, y_out = batch
        prediction: torch.Tensor = self.model(x_in)
        loss_value = F.nll_loss(prediction, y_out)
        self.log("test_loss", loss_value, on_step=True, on_epoch=True)

        acc = metrics.multiclass_accuracy(prediction, y_out, prediction.size(dim=1))
        self.log("test_accuracy", acc, on_epoch=True)

        rec = metrics.multiclass_recall(prediction, y_out, prediction.size(dim=1))
        self.log("test_recall", rec, on_epoch=True)

        prec = metrics.multiclass_precision(prediction, y_out, prediction.size(dim=1))
        self.log("test_precision", prec, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        # do the default optimizer step
        optimizer.step(closure=optimizer_closure)

        # log the learning rate
        if len(optimizer.param_groups) <= 1:
            pg = optimizer.param_groups[0]
            learning_rate = pg.get("learning_rate", pg["lr"])
            self.log("learning_rate", learning_rate)
        else:
            for (idx, pg) in enumerate(optimizer.param_groups):
                learning_rate = pg.get("learning_rate", pg["lr"])
                self.log(f"learning_rate_{idx}", learning_rate, on_step=True)


def run():
    trainer = L.Trainer(
        max_epochs=2,
        log_every_n_steps=1,
    )

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

    model = Classifier(ModelM3())
    trainer.fit(model=model, train_dataloaders=train_loader)
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    run()
