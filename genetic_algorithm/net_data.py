from random import randrange
from typing import List, Self

from constans_and_types import SizeParams


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

    def __getitem__(self, item):
        return self.list_[item]

    def __repr__(self):
        return "\t" + str(self.list_) + f" ACC: {self.acc}"

    def __iter__(self):
        return iter(self.list_)

    def __hash__(self):
        return hash(tuple(self.list_))
