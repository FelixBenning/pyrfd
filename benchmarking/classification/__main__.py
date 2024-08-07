""" Benchmarking RFD """

import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning as L
import sys

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

# pylint: disable=import-error,wrong-import-order
from benchmarking.classification.cifar.data import CIFAR100
from benchmarking.classification.cifar.resnet18 import Resnet18
from benchmarking.classification.mnist.data import MNIST, FashionMNIST
from benchmarking.classification.mnist.models import CNN3, CNN5, CNN7, AlgoPerf  # pylint: disable=unused-import
from benchmarking.classification.classifier import Classifier

from pyrfd import RFD, covariance


def train(problem, opt: optim.Optimizer, hyperparameters):
    """Train a Classifier """

    data: L.LightningDataModule = problem["dataset"](batch_size=problem["batch_size"])
    classifier = Classifier(
        problem["model"](),
        optimizer=opt,
        loss=problem["loss"],
        **hyperparameters,
    )

    trainer = trainer_from_problem(problem, opt_name=opt.__name__, hyperparameters=hyperparameters)
    trainer.fit(classifier, data)
    trainer.test(classifier, data)


def trainer_from_problem(problem, opt_name, hyperparameters):
    problem_id = f"{problem['dataset'].__name__}_{problem['model'].__name__}_b={problem['batch_size']}"
    hparams = "_".join([f"{key}={value}" for key, value in hyperparameters.items()])
    name = problem_id + "/" + opt_name + "(" + hparams + ")" + f"/seed={problem['seed']}"

    tlogger = TensorBoardLogger("logs/TensorBoard", name=name)
    csvlogger = CSVLogger("logs", name=name)

    L.seed_everything(problem["seed"], workers=True)

    return L.Trainer(
        **problem["trainer_params"],
        logger=[tlogger, csvlogger],
    )


PROBLEMS = {
    "FashionMNIST_CNN5" : {
        "dataset": FashionMNIST,
        "model": CNN5,
        "loss": F.nll_loss,
        "batch_size": 128,
        "seed": 42,
        "tol": 0.3,
        "trainer_params": {
            "max_epochs": 30,
            "log_every_n_steps": 1,
        }
    },
    "MNIST_AlgoPerf":{
        "dataset": MNIST,
        "model": AlgoPerf,
        "loss": F.nll_loss,
        "batch_size": 128,
        "seed": 42,
        "tol": 0.3,
        "trainer_params": {
            "max_epochs": 30,
            "log_every_n_steps": 1,
        }
    },
    "MNIST_CNN3" : {
        "dataset": MNIST,
        "model": CNN3,
        "loss": F.nll_loss,
        "batch_size": 128,
        "seed": 42,
        "tol": 0.3,
        "trainer_params": {
            "max_epochs": 30,
            "log_every_n_steps": 1,
        }
    },
    "MNIST_CNN7" : {
        "dataset": MNIST,
        "model": CNN7,
        "loss": F.nll_loss,
        "batch_size": 1024,
        "seed": 42,
        "tol": 0.3,
        "trainer_params": {
            "max_epochs": 30,
            "log_every_n_steps": 1,
        }
    },
    "CIFAR100_resnet18": {
        "dataset": CIFAR100,
        "model": Resnet18,
        "loss": nn.CrossEntropyLoss(label_smoothing=0),
        "batch_size": 1024,
        "seed": 42,
        "tol": 0.3,
        "trainer_params": {
            "max_epochs": 50,
            "log_every_n_steps": 5,
        }
    },
}

def main(problem_name, opt):
    problem = PROBLEMS[problem_name]

    # fit covariance model
    data: L.LightningDataModule = problem["dataset"](batch_size=problem["batch_size"])
    data.prepare_data()
    data.setup("fit")

    if opt == "cov":
        for run in range(20):
            sq_exp_cov_model = covariance.SquaredExponential()
            sq_exp_cov_model.auto_fit(
                model_factory=problem["model"],
                loss=problem["loss"],
                data=data.data_train,
                tol=problem['tol'],
                cache=f"""cache/{problem["dataset"].__name__}/{problem["model"].__name__}_run={run}/covariance_cache.csv""",
            )

    torch.set_float32_matmul_precision("high")

    if opt == "RFD-SE":
        sq_exp_cov_model = covariance.SquaredExponential()
        sq_exp_cov_model.auto_fit(
            model_factory=problem["model"],
            loss=problem["loss"],
            data=data.data_train,
            tol=problem['tol'],
            cache=f"""cache/{problem["dataset"].__name__}/{problem["model"].__name__}/covariance_cache.csv""",
        )

        for seed in range(20):
            problem["seed"] = seed
            train(
                problem,
                opt=RFD,
                hyperparameters={
                    "covariance_model": sq_exp_cov_model,
                    "conservatism": 0.1,
                },
            )
            train(
                problem,
                opt=RFD,
                hyperparameters={
                    "covariance_model": sq_exp_cov_model,
                    "b_size_inv": 1/problem["batch_size"],
                    "conservatism": 0.1,
                },
            )
    
    if opt == "RFD-RQ":
        rat_quad_cov_model = covariance.RationalQuadratic(beta=1)
        rat_quad_cov_model.auto_fit(
            model_factory=problem["model"],
            loss=problem["loss"],
            data=data.data_train,
            tol=problem['tol'],
            cache=f"""cache/{problem["dataset"].__name__}/{problem["model"].__name__}/covariance_cache.csv""",
        )
        for seed in range(20):
            problem["seed"] = seed
            train(
                problem,
                opt=RFD,
                hyperparameters={
                    "covariance_model": rat_quad_cov_model,
                },
            )

    if opt == "Adam":
        for seed in range(20):
            problem["seed"] = seed
            for lr in [
                1e-1,
                1e-2,
                1e-3,
                1e-4
            ]:
                train(
                    problem,
                    opt=optim.Adam,
                    hyperparameters={
                        "lr": lr,
                        "betas": (0.9, 0.999),
                    },
                )

    if opt == "SGD":
        for seed in range(20):
            problem["seed"] = seed
            for lr in [1e1, 1e0, 1e-1, 1e-2]:
                train(
                    problem,
                    opt=optim.SGD,
                    hyperparameters={
                        "lr": lr
                    },
                )





if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
