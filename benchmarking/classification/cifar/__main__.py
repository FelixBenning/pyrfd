import torch
from pyrfd.covariance import SquaredExponential
from pyrfd.optimizer import RFD
from torch.nn import CrossEntropyLoss
from .resnet18 import Resnet18
from .data import CIFAR100
from ..classifier import Classifier
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks

torch.set_float32_matmul_precision("high")
seed_everything(0)

BATCH_SIZE = 1024
TOL = 0.3
MAX_EPOCHS = 100

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
    cache=f"""cache/{CIFAR100.__name__}/{Resnet18.__name__}/covariance_cache_2.csv""",
)

classifier = Classifier(
    model=Resnet18(),
    optimizer=RFD,
    loss=loss,
    covariance_model=sq_exp_cov_model,
    conservatism=0.1,
)

trainer = Trainer(
    devices=[0],
    max_epochs=MAX_EPOCHS,
)
trainer.fit(model=classifier, datamodule=dm)
trainer.test(model=classifier, datamodule=dm)
