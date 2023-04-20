from dataclasses import dataclass
from pathlib import Path
from random import randrange, sample
from typing import NewType, List, Tuple

PROJECT_PATH = Path(__file__).parent
RAW_DATA_PATH = PROJECT_PATH.joinpath("raw_data")
READY_DATA_PATH = PROJECT_PATH.joinpath("ready_data")

Layer = NewType("Layer", int)


@dataclass
class SizeParams:
    min_: int
    max_: int
    step: int = 1

    def __iter__(self):
        return iter([self.min_, self.max_, self.step])


class NetworkData:
    def __init__(self, layer_par: SizeParams):
        super().__init__()
        self.acc = None
        self.list_: List[int] = []
        self.layer_par = layer_par

    def random_initialize(self, size, f_l_size):
        self.list_.append(f_l_size[0])
        for l_nr in range(1, size-1):
            min_, max_, step = self.layer_par
            l_size = randrange(min_, max_+1, step)
            self.list_.append(l_size)
        self.list_.append(f_l_size[1])
        return self

    def mutation(self):
        layer_nr = randrange(1, len(self.list_)-1)
        val = randrange(*self.layer_par)
        self.list_[layer_nr] = val
        return self
    
    def cross(self: "NetworkData", other: "NetworkData", net_par: SizeParams):
        a_max_el = len(self.list_) - 2
        b_max_el = len(other.list_) - 2
        a_b_min_el = 1
        min_, max_, _ = net_par
        min_ -= 2
        max_ -= 2
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
        return str(self.list_) + f" - ACC: {self.acc}"

    def __iter__(self):
        return iter(self.list_)


class Population:
    def __init__(self, net_param: SizeParams):
        super().__init__()
        self.list_: List[NetworkData] = []
        self.net_param = net_param

    def __add__(self, other):
        self.list_ += other.list_
        return self

    def __iter__(self):
        return iter(self.list_)

    def random_initialize(self,
                          size: int,
                          f_l_size: Tuple[int, int],
                          layer_param: SizeParams):
        self.list_.extend([NetworkData(layer_param)] * size)
        for indi_nr in range(size):
            min_, max_, step = self.net_param
            size = randrange(min_, max_+1, step)
            nd = NetworkData(layer_param)
            nd.random_initialize(size, f_l_size)
            self.list_[indi_nr] = nd
        return self

    def reset_acc(self):
        for net in self.list_:
            net.acc = None

    def do_crossing(self, n=None):
        if n is None:
            n = len(self.list_)
        parents_1 = sample(self.list_, n)
        parents_2 = sample(self.list_, n)
        new_pop = Population(self.net_param)
        for p1, p2 in zip(parents_1, parents_2):
            if p1 != p2:
                child = p1.cross(p2, self.net_param)
                new_pop.list_.append(child)
        return new_pop

    def do_mutations(self, n=None):
        if n is None:
            n = len(self.list_)
        to_mutate = sample(self.list_, n)
        for indi in to_mutate:
            indi.mutation()

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
    pass
    a = NetworkData(SizeParams(10, 300, 10))
    a.random_initialize(10, (300, 2))
    # print(a)
    # print(a.mutation())
    # print(a.mutation())
    # print(a.mutation())
    # print(a.mutation())

    b = Population(net_param=SizeParams(3, 10, 1))
    b.random_initialize(10, f_l_size=(300, 2), layer_param=SizeParams(100, 300, 10))
    for indi in b:
        indi.acc = randrange(1, 100)
    # print(b)
    # print(b.get_best())
    for _ in range(15):
        b += b.do_crossing()
        print(len(b.list_))
    n1 = NetworkData(SizeParams(5, 20, 1))
    n2 = NetworkData(SizeParams(5, 10, 1))
    n1.list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
    n2.list_ = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    n3 = n1.cross(n2, SizeParams(5, 10, 1))
    pop = Population(SizeParams(5, 10, 1))
    pop.list_.extend([n1, n2, n3])
    print(pop)