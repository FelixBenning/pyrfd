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

    trainer = trainer_from_problem(problem, opt_name=opt.__name__)
    trainer.fit(classifier, data)
    trainer.test(classifier, data)


def train_rfd(problem, cov_string, covariance_model):
    """Train a Classifier with RFD"""
    data: L.LightningDataModule = problem["dataset"](batch_size=problem["batch_size"])
    data.prepare_data()
    data.setup("fit")

    covariance_model.auto_fit(
        model_factory=problem["model"],
        loss=F.nll_loss,
        data=data.train_data,
        cache=f"""cache/{problem["dataset"].__name__}/{problem["model"].__name__}/covariance_cache.csv""",
    )

    classifier = Classifier(
        problem["model"](), optimizer=RFD, covariance_model=covariance_model
    )

    trainer = trainer_from_problem(problem, opt_name=f"RFD({cov_string})")
    trainer.fit(classifier, data)
    trainer.test(classifier, data)


def trainer_from_problem(problem, opt_name):
    problem_id = f"{problem['dataset'].__name__}_{problem['model'].__name__}_{problem['batch_size']}"

    tlogger = TensorBoardLogger("logs/TensorBoard", name=problem_id + opt_name)
    csvlogger = CSVLogger("logs", name=problem_id + opt_name)

    tlogger.log_hyperparams(params={"batch_size": problem["batch_size"]})
    csvlogger.log_hyperparams(params={"batch_size": problem["batch_size"]})
    return L.Trainer(
        max_epochs=2,
        log_every_n_steps=1,
        logger=[tlogger, csvlogger],
    )


def main():
    problem = {
        "dataset": MNIST,
        "model": CNN3,
        "batch_size": 100,
    }

    train_rfd(
        problem,
        cov_string="SquaredExponential",
        covariance_model=covariance.SquaredExponential(),
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
