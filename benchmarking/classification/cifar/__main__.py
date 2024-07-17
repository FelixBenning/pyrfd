import torch
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


def objective(trial: Trial) -> float:
    hyperparameters = dict()

    BATCH_SIZE = 64
    TOL = 0.3
    MAX_EPOCHS = 10

    dm = CIFAR100(batch_size=BATCH_SIZE)
    dm.prepare_data()
    dm.setup("fit")

    loss = CrossEntropyLoss()

    sq_exp_cov_model = SquaredExponential()
    sq_exp_cov_model.auto_fit(
        model_factory=Resnet18,
        loss=loss,
        data=dm.data_train,
        tol=TOL,
        cache=f"""cache/{CIFAR100.__name__}/{Resnet18.__name__}/covariance_cache.csv""",
    )

    classifier = Classifier(
        model=Resnet18(),
        optimizer=RFD,
        loss=loss,
        covariance_model=sq_exp_cov_model,
        conservatism=0.1,
    )

    # TODO
    trainer = Trainer(
        devices=[0],
        max_epochs=MAX_EPOCHS,
        logger=TensorBoardLogger(
            save_dir="lightning_logs",
            version=trial.number,
            name=f"{CIFAR100.__name__}_{Resnet18.__name__}",
        ),
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/accuracy")],
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model=classifier, datamodule=dm)
    trainer.test(model=classifier, datamodule=dm)

    trial.set_user_attr("test_accuracy", trainer.callback_metrics["test/accuracy"])
    trial.set_user_attr("test_loss", trainer.callback_metrics["test/loss"])
    trial.set_user_attr("test_recall", trainer.callback_metrics["test/recall"])
    trial.set_user_attr("test_precision", trainer.callback_metrics["test/precision"])

    return trainer.callback_metrics["val/accuracy"].item()


def main():
    torch.set_float32_matmul_precision("high")
    SEED = 0
    seed_everything(SEED)

    pruner = MedianPruner()
    sampler = TPESampler(seed=SEED)

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        pruner=pruner,
        sampler=sampler,
        direction="maximize",
    )

    study.optimize(objective, timeout=10)


main()
