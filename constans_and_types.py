from dataclasses import dataclass
from pathlib import Path
from random import randrange, sample
from typing import NewType, List


PROJECT_PATH = Path(__file__).parent
RAW_DATA_PATH = PROJECT_PATH.joinpath("raw_data")
READY_DATA_PATH = PROJECT_PATH.joinpath("ready_data")

Layer = NewType("Layer", int)


class NetworkData:
    def __init__(self, min_max_step_layer_size):
        super().__init__()
        self.acc = None
        self.list_: List[int] = []
        self.min_max_step_layer_size = min_max_step_layer_size

    def random_initialize(self, size, f_l_size):
        self.list_.append(f_l_size[0])
        for l_nr in range(1, size-1):
            l_size = randrange(*self.min_max_step_layer_size)
            self.list_.append(l_size)
        self.list_.append(f_l_size[1])
        return self

    def mutation(self):
        layer_nr = randrange(1, len(self.list_)-1)
        val = randrange(*self.min_max_step_layer_size)
        self.list_[layer_nr] = val
        return self
    
    def cross(self: "NetworkData", other: "NetworkData", min_max_net_size):
        a_max_el = len(self.list_) - 2
        b_max_el = len(other.list_) - 2
        a_b_min_el = 1
        min_, max_, _ = min_max_net_size
        min_ -= 2
        max_ -= 2
        a_start = max(min_ - b_max_el, a_b_min_el) + 1
        a_end = min(max_ - a_b_min_el + 1, a_max_el)
        a_point = randrange(a_start, a_end+1, 1)
        b_start = max(min_ - a_point + 1, a_b_min_el) + 1
        print(a_point)
        b_end = min(max_ - a_point + 2, b_max_el)
        if b_start == b_end:
            b_point = -b_start
        else:
            b_point = randrange(-b_end, -b_start+1)
        part_a = self.list_[:a_point]
        part_b = other.list_[b_point:]
        child = NetworkData(self.min_max_step_layer_size)
        child.list_ = part_a + part_b
        print("a_max_el: ", a_max_el,"\n",
                "b_max_el: ", b_max_el,"\n",
                "a_b_min_el: ", a_b_min_el,"\n",
                "min_: ", min_,"\n",
                "max_: ", max_,"\n",
                "a_start: ", a_start,"\n",
                "a_end: ", a_end,"\n",
                "a_point: ", a_point,"\n",
                "b_start: ", b_start,"\n",
                "b_end: ", b_end,"\n",
                "b_point: ", b_point,"\n"
                'child: ', child)
        return child

    def __repr__(self):
        return str(self.list_) + f" - ACC: {self.acc}"


class Population(list):
    def __init__(self, min_max_step_net_size):
        super().__init__()
        self.list_: List[NetworkData] = []
        self.min_max_step_net_size = min_max_step_net_size

    def __add__(self, other):
        self.list_ += other.list_
        return self

    def random_initialize(self,
                          size,
                          f_l_size,
                          min_max_step_layer_size):
        self.list_.extend([[0]] * size)
        for indi_nr in range(size):
            size = randrange(*self.min_max_step_net_size)
            nd = NetworkData(min_max_step_layer_size)
            nd.random_initialize(size, f_l_size)
            self.list_[indi_nr] = nd
        return self

    def reset_acc(self):
        for net in self.list_:
            net.acc = None

    def do_crossing(self, num):
        parents_1 = sample(self.list_, num)
        parents_2 = sample(self.list_, num)
        min_max_net_size = self.min_max_step_net_size[0, 2]
        new_pop = Population(self.min_max_step_net_size)
        for p1, p2 in zip(parents_1, parents_2):
            if p1 != p2:
                child = p1.cross(p2, min_max_net_size)
                new_pop.append(child)
        return new_pop

    def do_mutations(self, num):
        to_mutate = sample(self, num)
        for indi in to_mutate:
            indi.mutatation()

    def get_best(self, n):
        return sorted(self.list_, key=lambda x: x.acc, reverse=True)[:n]

    def __repr__(self: List[NetworkData]):
        str_ = "Population:\n"
        for net_data in self:
            str_ += '\t' + str(net_data) + '\n'
        str_ += "\t___"
        return str_


if __name__ == "__main__":
    # a = NetworkData((10, 300, 10))
    # a.random_initialize(10, (300, 2))
    # print(a)
    # print(a.mutation())
    # print(a.mutation())
    # print(a.mutation())
    # print(a.mutation())
    #
    # b = Population(min_max_step_net_size=(3, 10, 1))
    # b.random_initialize(10, f_l_size=(300, 2),
    #                     min_max_step_layer_size=(100, 300, 10))
    # print(b)

    n1 = NetworkData((5, 20, 1))
    n2 = NetworkData((5, 10, 1))
    n1.list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
    n2.list_ = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    n1.cross(n2, (5, 10, 1))