import dataclasses
from enum import Enum
from typing import Dict


class Mode(Enum):
    train = "train"
    eval = "eval"


@dataclasses.dataclass()
class DatasetConfig:
    batch_size: int = 16
    num_workers: int = 8
    path: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: "./data/mitbih_train.csv",
        Mode.eval: "./data/mitbih_test.csv"
    })
    transforms: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: lambda x: x, Mode.eval: lambda x: x})


@dataclasses.dataclass()
class EcgConfig:
    dataset: DatasetConfig = DatasetConfig()
