import torch
from torch.optim import Adam
from benchmarking.classification.cifar.batch_norm_cnn import BatchNormCNN
from benchmarking.classification.cifar.simple_cnn import SimpleCNN
from benchmarking.classification.cifar.simplest_cnn import SimplestCNN
from benchmarking.classification.cifar.simplest_cnn_bn_last import SimplestCNNBNLast
from benchmarking.classification.cifar.vgg import VGG, vgg16_bn
from pyrfd.covariance import SquaredExponential
from pyrfd.optimizer import RFD
from torch.nn import CrossEntropyLoss
from .resnet18 import Resnet18
from .data import CIFAR100
from ..classifier import Classifier
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.trial import Trial
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks import EarlyStopping
import argparse


def objective(
    trial: Trial,
    Optimizer: RFD | Adam,
    Model: Resnet18 | VGG | SimpleCNN | BatchNormCNN | SimplestCNN | SimplestCNNBNLast,
    device: int,
    tolerance: float,
) -> float:
    # Global HPs
    MAX_EPOCHS = 300
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])

    # Initialize loss and datamodule
    loss = CrossEntropyLoss()
    dm = CIFAR100(batch_size=batch_size, num_workers=6)

    if Optimizer == RFD:
        # Get training dataset ready
        dm.prepare_data()
        dm.setup("fit")

        # Fit covariance model
        sq_exp_cov_model = SquaredExponential()
        sq_exp_cov_model.auto_fit(
            model_factory=Model,
            loss=loss,
            data=dm.data_train,
            tol=tolerance,
            cache=f"""cache/{CIFAR100.__name__}/{Model.__name__}/covariance_cache_2.csv""",
            max_iter=20,
        )

        # Setup optimizer HPs
        additional_classifier_kwargs = dict(
            covariance_model=sq_exp_cov_model,
            conservatism=trial.suggest_float("conservatism", 0.0, 1.0),
        )
    elif Optimizer == Adam:
        # Setup optimizer HPs
        additional_classifier_kwargs = dict(
            lr=trial.suggest_float("lr", 1e-6, 1e-1, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        )

    # Initialize model
    classifier = Classifier(
        model=Model(), optimizer=Optimizer, loss=loss, **additional_classifier_kwargs
    )

    # Setup trainer
    trainer = Trainer(
        devices=[device],
        max_epochs=MAX_EPOCHS,
        logger=TensorBoardLogger(
            save_dir="lightning_logs",
            version=trial.number,
            name=f"{CIFAR100.__name__}_{Model.__name__}_{Optimizer.__name__}",
        ),
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val/accuracy"),
            EarlyStopping("val/accuracy", patience=5, mode="max", verbose=True),
        ],
    )
    trainer.logger.log_hyperparams(
        {
            "batch_size": batch_size,
            "max_epochs": MAX_EPOCHS,
            **additional_classifier_kwargs,
        }
    )

    # Fit model and keep validation accuracy
    trainer.fit(model=classifier, datamodule=dm)
    val_accuracy = trainer.callback_metrics["val/accuracy"].item()

    # Test model and store metrics
    trainer.test(model=classifier, datamodule=dm)
    trial.set_user_attr(
        "test_accuracy", trainer.callback_metrics["test/accuracy"].item()
    )
    trial.set_user_attr("test_loss", trainer.callback_metrics["test/loss"].item())
    trial.set_user_attr("test_recall", trainer.callback_metrics["test/recall"].item())
    trial.set_user_attr(
        "test_precision", trainer.callback_metrics["test/precision"].item()
    )

    # Return validation accuracy for HP optimization
    return val_accuracy


def main():
    SEED = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, choices=["RFD", "Adam"], required=True)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "Resnet18",
            "VGG",
            "SimpleCNN",
            "BatchNormCNN",
            "SimplestCNN",
            "SimplestCNNBNLast",
        ],
        required=True,
    )
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    if args.optimizer == "RFD":
        Optimizer = RFD
    elif args.optimizer == "Adam":
        Optimizer = Adam
    else:
        raise ValueError("Invalid optimizer specified")

    if args.model == "Resnet18":
        Model = Resnet18
    elif args.model == "VGG":
        Model = vgg16_bn
    elif args.model == "SimpleCNN":
        Model = SimpleCNN
    elif args.model == "BatchNormCNN":
        Model = BatchNormCNN
    elif args.model == "SimplestCNN":
        Model = SimplestCNN
    elif args.model == "SimplestCNNBNLast":
        Model = SimplestCNNBNLast
    else:
        raise ValueError("Invalid model specified")

    torch.set_float32_matmul_precision("high")
    seed_everything(SEED)

    pruner = MedianPruner()
    sampler = TPESampler(seed=SEED)

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name=f"{CIFAR100.__name__}_{Model.__name__}_{Optimizer.__name__}",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(trial, Optimizer, Model, args.device, args.tolerance),
        timeout=args.timeout,
    )


main()
