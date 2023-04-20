from dataclasses import dataclass
from pathlib import Path
from random import randrange
from typing import NewType

from genetic_algorithm.population import Population

PROJECT_PATH = Path(__file__).parent
RAW_DATA_PATH = PROJECT_PATH.joinpath("raw_data")
READY_DATA_PATH = PROJECT_PATH.joinpath("ready_data")

Layer = NewType("Layer", int)


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


if __name__ == "__main__":
    b = Population(net_param=SizeParams(3, 10, 1))
    b.random_initialize(10, f_l_size=(300, 2),
                        layer_param=SizeParams(100, 300, 10))
    for indi in b:
        indi.acc = randrange(1, 100)
    print(b)
    print(b.get_best())
    for _ in range(1000):
        b = b.do_crossing()
        b.do_mutations()
    print(b)
