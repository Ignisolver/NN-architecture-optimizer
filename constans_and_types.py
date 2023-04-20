from dataclasses import dataclass
from pathlib import Path
from random import randrange, sample
from typing import NewType, List, Tuple, Self

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


class NetworkData:
    def __init__(self, layer_par: SizeParams):
        super().__init__()
        self.acc_ = None
        self.new = True
        self.list_: List[int] = []
        self.layer_par = layer_par

    @property
    def acc(self):
        return self.acc_

    @acc.setter
    def acc(self, val):
        self.acc_ = val
        self.new = False

    def random_initialize(self, size, f_l_size) -> Self:
        self.list_.append(f_l_size[0])
        for l_nr in range(1, size-1):
            l_size = randrange(*self.layer_par)
            self.list_.append(l_size)
        self.list_.append(f_l_size[1])
        return self

    def mutation(self, n=1) -> Self:
        mutated = NetworkData(self.layer_par)
        mutated.list_ = self.list_[:]
        mutated_places = [None]
        for i in range(n):
            layer_nr = None
            while layer_nr in mutated_places:
                layer_nr = randrange(1, len(self.list_) - 1)
            new_lay_size = randrange(*self.layer_par)
            while new_lay_size == self.list_[layer_nr]:
                new_lay_size = randrange(*self.layer_par)
            mutated.list_[layer_nr] = new_lay_size
        return mutated

    def cross(self: "NetworkData", other: "NetworkData",
              net_par: SizeParams) -> "NetworkData":
        a_max_el = len(self.list_) - 2
        b_max_el = len(other.list_) - 2
        a_b_min_el = 1
        min_, max_, step = net_par
        min_ -= 2
        max_ -= 2 + (step if (max_-min_) % step == 0 else 0)

        a_start = max(min_ - b_max_el, a_b_min_el) + 1
        a_end = min(max_ - a_b_min_el + 1, a_max_el + 1)
        a_point = randrange(a_start, a_end+1, 1)

        b_start = max(min_ - a_point + 1, a_b_min_el) + 1
        b_end = min(max_ - a_point + 2, b_max_el+1)
        if b_start == b_end:
            b_point = -b_start
        else:
            b_point = randrange(-b_end, -b_start+1)

        part_a = self.list_[:a_point]
        part_b = other.list_[b_point:]
        child = NetworkData(self.layer_par)
        child.list_ = part_a + part_b
        return child

    def __repr__(self):
        return "\t" + str(self.list_) + f" ACC: {self.acc}"

    def __iter__(self):
        return iter(self.list_)

    def __hash__(self):
        return hash(tuple(self.list_))

class Population:
    def __init__(self, net_param: SizeParams):
        super().__init__()
        self.list_: List[NetworkData] = []
        self.net_param = net_param

    def __add__(self, other) -> Self:
        self.list_ += other.list_
        return self

    def __iter__(self):
        return iter(self.list_)

    def random_initialize(self,
                          size: int,
                          f_l_size: Tuple[int, int],
                          layer_param: SizeParams) -> Self:
        self.list_.extend([NetworkData(layer_param)] * size)
        for indi_nr in range(size):
            size = randrange(*self.net_param)
            nd = NetworkData(layer_param)
            nd.random_initialize(size, f_l_size)
            self.list_[indi_nr] = nd
        return self

    def reset_acc(self):
        for net in self.list_:
            net.acc = None

    def do_crossing(self, n=None) -> "Population":
        if n is None:
            n = len(self.list_)
        indexes = range(len(self.list_))
        parents_1_idx = sample(indexes, n)
        crossed = Population(self.net_param)
        for p1_id in parents_1_idx:
            p1 = self.list_[p1_id]
            child = p1
            rest = self.list_[:p1_id]+self.list_[p1_id+1:]
            while p1.list_ == child.list_:
                p2 = sample(rest, 1)[0]
                child = p1.cross(p2, self.net_param)
            crossed.list_.append(child)
        return crossed

    def do_mutations(self, n_in_pop=None, n_in_indi=1):
        if n_in_pop is None:
            n_in_pop = len(self.list_)
        to_mutate = sample(self.list_, n_in_pop)
        mutated = Population(self.net_param)
        for indi in to_mutate:
            m = indi.mutation(n_in_indi)
            mutated.list_.append(m)
        return mutated

    def get_best(self, n=None):
        if n is None:
            n = len(self.list_)
        pop = Population(self.net_param)
        best = sorted(self.list_, key=lambda x: x.acc, reverse=True)[:n]
        pop.list_.extend(best)
        return pop

    def __repr__(self):
        str_ = "Population:\n"
        for net_data in self.list_:
            str_ += '\t' + str(net_data) + '\n'
        str_ += "\t___"
        return str_


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
