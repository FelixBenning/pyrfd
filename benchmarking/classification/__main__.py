""" Benchmarking RFD """

from typing import Dict, Any

from torch import optim
import torch.nn.functional as F
import lightning as L

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

# pylint: disable=import-error,wrong-import-order
from mnist.data import MNIST
from mnist.models import CNN3, CNN5, CNN7  # pylint: disable=unused-import
from classifier import Classifier

from pyrfd import RFD, covariance


def train_classic(
    problem,
    opt: optim.Optimizer,
    hyperparameters: Dict[str, Any],
):
    """Train a Classifier with RFD"""

    data: L.LightningDataModule = problem["dataset"](batch_size=problem["batch_size"])
    classifier = Classifier(problem["model"](), optimizer=opt, **hyperparameters)

    trainer = trainer_from_problem(problem, opt_name=opt.__name__, hyperparameters=hyperparameters)
    trainer.fit(classifier, data)
    trainer.test(classifier, data)


def train_rfd(problem, hyperparameters):
    """Train a Classifier with RFD"""
    data: L.LightningDataModule = problem["dataset"](batch_size=problem["batch_size"])
    data.prepare_data()
    data.setup("fit")


    classifier = Classifier(
        problem["model"](),
        optimizer=RFD,
        **hyperparameters,
    )

    trainer = trainer_from_problem(problem, opt_name="RFD", hyperparameters=hyperparameters)
    trainer.fit(classifier, data)
    trainer.test(classifier, data)


def trainer_from_problem(problem, opt_name, hyperparameters):
    problem_id = f"{problem['dataset'].__name__}_{problem['model'].__name__}_b={problem['batch_size']}"

    hparams = "_".join([f"{key}={value}" for key, value in hyperparameters.items()])
    name = problem_id + "/" + opt_name + "(" + hparams + ")"
    tlogger = TensorBoardLogger("logs/TensorBoard", name=name)
    csvlogger = CSVLogger("logs", name=name)

    return L.Trainer(
        max_epochs=30,
        log_every_n_steps=1,
        logger=[tlogger, csvlogger],
    )


def main():
    problem = {
        "dataset": MNIST,
        "model": CNN7,
        "batch_size": 1000,
    }

    # fit covariance model
    data: L.LightningDataModule = problem["dataset"](batch_size=problem["batch_size"])
    data.prepare_data()
    data.setup("fit")
    covariance_model = covariance.SquaredExponential()
    covariance_model.auto_fit(
        model_factory=problem["model"],
        loss=F.nll_loss,
        data=data.data_train,
        cache=f"""cache/{problem["dataset"].__name__}/{problem["model"].__name__}/covariance_cache.csv""",
    )
    # ------

    train_rfd(
        problem,
        hyperparameters={
            "covariance_model": covariance_model,
        }
    )

    train_rfd(
        problem,
        hyperparameters={
            "covariance_model": covariance_model,
            "b_size_inv": 1/problem["batch_size"],
        }
    )

    train_classic(
        problem,
        opt=optim.SGD,
        hyperparameters={
            "lr": 1e-3,
            "momentum": 0,
        },
    )

    train_classic(
        problem,
        opt=optim.Adam,
        hyperparameters={
            "lr": 1e-3,
            "betas": (0.9, 0.999),
        },
    )


if __name__ == "__main__":
    main()
