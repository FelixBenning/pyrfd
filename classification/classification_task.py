import importlib
import inspect
from pathlib import Path
import pkgutil
import sys
from typing import Any, Literal
from attr import dataclass
from ruamel.yaml import YAML

from torch import nn
import lightning as L


import classification
from classification.mnist.data import MNIST
from classification.mnist.models.cnn3 import CNN3


YAML_GLOBAL = YAML()


# def get_models_and_datasets():
#     models = {}
#     datasets = {}

#     ### Collect all classification datasets and models
#     for dataset in pkgutil.iter_modules(classification.__path__):
#         if not dataset.ispkg:
#             continue
#         data_module = importlib.import_module(f"classification.{dataset.name}.data")
#         models_module = importlib.import_module(f"classification.{dataset.name}.models")

#         # pylint: disable=cell-var-from-loop ## only use the lambda in the same loop
#         datasets.update(
#             {
#                 cls.__name__: cls
#                 for _, cls in inspect.getmembers(
#                     data_module,
#                     lambda obj: inspect.isclass(obj)
#                     and (obj.__module__.startswith(data_module.__name__))
#                     and issubclass(obj, L.LightningDataModule),
#                 )
#             }
#         )
#         models.update(
#             {
#                 cls.__name__: cls
#                 for _, cls in inspect.getmembers(
#                     models_module,
#                     lambda obj: inspect.isclass(obj)
#                     and (obj.__module__.startswith(models_module.__name__))
#                     and issubclass(obj, nn.Module),
#                 )
#             }
#         )

#     return models, datasets


# MODELS, DATASETS = get_models_and_datasets()
# # register_classes(YAML_GLOBAL, MODELS)
# # register_classes(YAML_GLOBAL, DATASETS)


def class_from_str(cls_string, predicate: callable = lambda _: True):
    module, cls_name = cls_string["model"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module), cls_name)
    assert predicate(cls)
    return cls

@dataclass
class Params(dict):
    """ Parameter dictionaries for ClasisficationTask """
    dataset:dict[str,Any]={}
    model:dict[str,Any]={}
    loss:dict[str,Any]={}
    training:dict[str,Any]={}

def class_tag(cls):
    return f"!{cls.__module__}.{cls.__qualname__}"

@YAML_GLOBAL.register_class
@dataclass
class ClassificationTask:
    model_class: type[nn.Module]
    dataset_class: type[L.LightningDataModule]
    params: Params[str, dict]
    loss_class: type[nn.Module] = nn.CrossEntropyLoss
    seed: None | int = None

    @classmethod
    def to_yaml(cls, representer, node:"ClassificationTask"):
        return representer.represent_mapping(
            class_tag(cls),
            {
                "model": representer.represent_mapping(
                    class_tag(node.model_class),
                    node.params.model,
                ),
                "dataset": representer.represent_mapping(
                    class_tag(node.dataset_class),
                    node.params.dataset,
                ),
                "loss": representer.represent_mapping(
                    class_tag(node.loss_class),
                    node.params.loss,
                ),
                "seed": node.seed,
                "params.training": node.params.training,
            },
        )

    @classmethod
    def from_yaml(cls, constructor, node:"ClassificationTask"):
        return cls(
            model_class=class_from_str(
                node["model"], predicate=lambda cls: issubclass(cls, nn.Module)
            ),
            datase_classt=class_from_str(
                node["dataset"],
                predicate=lambda cls: issubclass(cls, L.LightningDataModule),
            ),
            seed= node.get("seed", None),
            loss_class=class_from_str(
                node["loss"], predicate=lambda cls: issubclass(cls, nn.Module)
            ),
            params=Params(node["params"]),
        )

    @property
    def loss(self):
        """ Return an instance of the loss function """
        return self.loss_class(self.params.loss)

    @property
    def model(self):
        """ Return an instance of the model """
        return self.model_class(self.params.model)

    @property
    def dataset(self):
        """ Return an instance of the dataset """
        return self.dataset_class(self.params.dataset)


if __name__ == "__main__":
    # print(DATASETS)
    # print(MODELS)

    task = ClassificationTask(CNN3, MNIST, Params(training={"batch_size": 128}))
    YAML_GLOBAL.dump(task, sys.stdout)
