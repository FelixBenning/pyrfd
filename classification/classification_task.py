""" ClassificationTask serialization """
from pydoc import locate
from typing import Any
from attr import dataclass
import yaml

from torch import nn
import lightning as L


@dataclass
class Params(dict):
    """Parameter dictionaries for ClasisficationTask"""

    dataset: dict[str, Any] = {}
    model: dict[str, Any] = {}
    loss: dict[str, Any] = {}
    training: dict[str, Any] = {}


def full_name(cls):
    """Return the full name of a class"""
    module = cls.__module__
    if module is None or module == "builtins" or module == "__main__":
        module = ""
    else:
        module += "."
    return f"{module}{cls.__qualname__}"


@dataclass
class ClassificationTask:
    model_class: type[nn.Module]
    dataset_class: type[L.LightningDataModule]
    params: Params[str, dict]
    loss_class: type[nn.Module] = nn.CrossEntropyLoss
    seed: None | int = None

    @property
    def loss(self):
        """Return an instance of the loss function"""
        return self.loss_class(self.params.loss)

    @property
    def model(self):
        """Return an instance of the model"""
        return self.model_class(self.params.model)

    @property
    def dataset(self):
        """Return an instance of the dataset"""
        return self.dataset_class(self.params.dataset)

    @staticmethod
    def to_yaml(dumper, data: "ClassificationTask"):
        """Dump a ClassificationTask to a YAML file (registerd as representer)"""
        return dumper.represent_mapping(
            "!" + full_name(data.__class__),
            {
                "model": {
                    "class": full_name(data.model_class),
                    "params": data.params.model,
                },
                "dataset": {
                    "class": full_name(data.dataset_class),
                    "params": data.params.dataset,
                },
                "loss": {
                    "class": full_name(data.loss_class),
                    "params": data.params.loss,
                },
                "seed": data.seed,
                "training": {"params": data.params.training},
            },
        )

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        """Load a ClassificationTask from a YAML file (registerd as constructor)"""
        data = loader.construct_mapping(node, deep=True)
        model_class = locate(data["model"]["class"])
        dataset_class = locate(data["dataset"]["class"])
        loss_class = locate(data["loss"]["class"])
        params = Params(
            model=data["model"]["params"],
            dataset=data["dataset"]["params"],
            loss=data["loss"]["params"],
        )
        seed = data.get("seed", None)
        return ClassificationTask(
            model_class=model_class,
            dataset_class=dataset_class,
            params=params,
            loss_class=loss_class,
            seed=seed,
        )


yaml.add_representer(ClassificationTask, ClassificationTask.to_yaml)
yaml.add_constructor("!ClassificationTask", ClassificationTask.from_yaml)

if __name__ == "__main__":
    # print(DATASETS)
    # print(MODELS)

    # task = ClassificationTask(CNN3, MNIST, Params(training={"batch_size": 128}))
    with open("task.yaml", "r", encoding="utf-8") as file:
        task = yaml.load(file, Loader=yaml.FullLoader)
