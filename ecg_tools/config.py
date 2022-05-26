import dataclasses
from enum import Enum
from typing import Dict, Union


class Mode(Enum):
    train = "train"
    eval = "eval"


@dataclasses.dataclass()
class DatasetConfig:
    batch_size: int = 64
    num_workers: int = 8
    path: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: "./data/mitbih_train.csv",
        Mode.eval: "./data/mitbih_test.csv"
    })
    transforms: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: lambda x: x, Mode.eval: lambda x: x})


@dataclasses.dataclass()
class ModelConfig:
    num_layers: int = 6
    signal_length: int = 187
    num_classes: int = 5
    input_channels: int = 1
    embed_size: int = 192
    num_heads: int = 8
    expansion: int = 4


@dataclasses.dataclass()
class EcgConfig:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    device: Union[int, str] = "cuda"
    lr: float = 2e-4
    num_epochs: int = 5
