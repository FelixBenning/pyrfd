""" ClassificationTask serialization """

from pydoc import locate
from sys import stdout
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


def strip_exclamation(tag: str) -> str:
    """Strip the exclamation mark from a YAML tag"""
    assert tag.startswith("!"), f"Tag {tag} does not start with '!'"
    return tag[1:]


@dataclass
class ClassificationTask:
    model_class: type[nn.Module]
    dataset_class: type[L.LightningDataModule]
    loss_class: type[nn.Module] = nn.CrossEntropyLoss
    params: dict = {}

    @property
    def loss(self):
        """Return an instance of the loss function"""
        return self.loss_class(self.params["loss"])

    @property
    def model(self):
        """Return an instance of the model"""
        return self.model_class(self.params["model"])

    @property
    def dataset(self):
        """Return an instance of the dataset"""
        return self.dataset_class(self.params["dataset"])

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: "ClassificationTask"):
        """Dump a ClassificationTask to a YAML file (registerd as representer)"""
        params = data.params.copy()
        value = []
        for key in ("model", "dataset", "loss"):
            value.append(
                (
                    dumper.represent_str(key),
                    dumper.represent_mapping(
                        tag="!" + full_name(getattr(data, key + "_class")),
                        mapping=params.pop(key, {}),
                    ),
                )
            )

        value.extend(dumper.represent_data(params).value)
        return yaml.MappingNode(tag="!" + full_name(data.__class__), value=value)

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        """Load a ClassificationTask from a YAML file (registerd as constructor)"""
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        data = {"params": {}}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if key in ("model", "dataset", "loss"):
                data[key] = locate(strip_exclamation(value_node.tag))
                data["params"][key] = loader.construct_mapping(value_node, deep=True)
            else:
                data["params"][key] = loader.construct_object(value_node, deep=True)

        return ClassificationTask(
            model_class=data["model"],
            dataset_class=data["dataset"],
            params=data["params"],
            loss_class=data["loss"],
        )


yaml.add_representer(ClassificationTask, ClassificationTask.to_yaml)
yaml.add_constructor("!ClassificationTask", ClassificationTask.from_yaml)

if __name__ == "__main__":
    # print(DATASETS)
    # print(MODELS)

    # task = ClassificationTask(CNN3, MNIST, Params(training={"batch_size": 128}))
    with open("task.yaml", "r", encoding="utf-8") as file:
        task = yaml.load(file, Loader=yaml.FullLoader)
    
    yaml.dump(task, stdout)
