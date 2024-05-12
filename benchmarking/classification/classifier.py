""" Abstract Classifier Training """

from __future__ import annotations
from typing import Callable, Any

import torch
from torch import optim
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
import torchmetrics.functional.classification as metrics


class Classifier(L.LightningModule):
    """Abstract Classifier for training and testing a model on a classification dataset"""

    def __init__(self, model, optimizer, **hyperameters):
        super().__init__()
        self.optimizer = optimizer
        self.model = model
        self.hyperparemeters = hyperameters
        self.save_hyperparameters(ignore=["model", "optimizer", "covariance_model"])


    # pylint: disable=arguments-differ
    def training_step(self, batch, *args, **kwargs):
        """Apply Negative Log-Likelihood loss to the model output and logging"""
        x_in, y_out = batch
        prediction: torch.Tensor = self.model(x_in)
        loss_value = F.nll_loss(prediction, y_out)
        self.log("train/loss", loss_value, prog_bar=True)
        acc = metrics.multiclass_accuracy(prediction, y_out, prediction.size(dim=1))
        self.log("train/accuracy", acc, on_epoch=True)
        return loss_value

    def validation_step(self, batch):
        """Apply Negative Log-Likelihood loss to the model output and logging"""
        x_in, y_out = batch
        prediction: torch.Tensor = self.model(x_in)
        loss_value = F.nll_loss(prediction, y_out)
        self.log("val/loss", loss_value, on_epoch=True)

        acc = metrics.multiclass_accuracy(prediction, y_out, prediction.size(dim=1))
        self.log("val/accuracy", acc, on_epoch=True)

        rec = metrics.multiclass_recall(prediction, y_out, prediction.size(dim=1))
        self.log("val/recall", rec, on_epoch=True)

        prec = metrics.multiclass_precision(prediction, y_out, prediction.size(dim=1))
        self.log("val/precision", prec, on_epoch=True)

    # pylint: disable=arguments-differ
    def test_step(self, batch):
        """Apply Negative Log-Likelihood loss to the model output and logging"""
        x_in, y_out = batch
        prediction: torch.Tensor = self.model(x_in)
        loss_value = F.nll_loss(prediction, y_out)
        self.log("test/loss", loss_value, on_epoch=True)

        acc = metrics.multiclass_accuracy(prediction, y_out, prediction.size(dim=1))
        self.log("test/accuracy", acc, on_epoch=True)

        rec = metrics.multiclass_recall(prediction, y_out, prediction.size(dim=1))
        self.log("test/recall", rec, on_epoch=True)

        prec = metrics.multiclass_precision(prediction, y_out, prediction.size(dim=1))
        self.log("test/precision", prec, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.hyperparemeters)

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
            for idx, pg in enumerate(optimizer.param_groups):
                learning_rate = pg.get("learning_rate", pg["lr"])
                self.log(f"learning_rate_{idx}", learning_rate, on_step=True)
