from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from random import randrange
from typing import NewType


PROJECT_PATH = Path(__file__).parent
RAW_DATA_PATH = PROJECT_PATH.joinpath("raw_data")
READY_DATA_PATH = PROJECT_PATH.joinpath("ready_data")
MODELS_PATH = PROJECT_PATH.joinpath("models")

Layer = NewType("Layer", int)

Sets = namedtuple("Sets", ["trainX", "trainY", "testX", "testY"])


@dataclass
class SizeParams:
    min_: int
    max_: int
    step: int = 1

    def __post_init__(self):
        assert (self.max_-self.min_) % self.step == 0
        self.max_ += self.step

    def __iter__(self):
        return iter([self.min_, self.max_, self.step])


