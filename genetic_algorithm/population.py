from random import randrange, sample
from typing import List, Tuple

from numpy.random import rand
from .net_data import NetworkData
from constans_and_types import SizeParams


class Population:
    def __init__(self, net_param: SizeParams):
        super().__init__()
        self.list_: List[NetworkData] = []
        self.net_param = net_param

    def __add__(self, other) :
        self.list_ += other.list_
        return self

    def __iter__(self):
        return iter(self.list_)

    def load_init_pop(self, pop_list, layer_param: SizeParams):
        self.list_ = [NetworkData(layer_param, init_list=init_list)
                      for init_list in pop_list]

    def random_initialize(self,
                          size: int,
                          f_l_size: Tuple[int, int],
                          layer_param: SizeParams):
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
        pop = self._sort_pop(n)
        return pop

    def get_best_stat(self, n=None):
        new_pop = Population(self.net_param)
        sorted_pop = self._sort_pop(n)
        max_acc = sorted_pop.list_[0].acc
        indi_acc = []
        for indi in sorted_pop:
            indi_acc.append((indi, indi.acc/max_acc))
        while len(new_pop.list_) < n:
            rand_nr = rand(1)
            for indi, acc in indi_acc:
                if acc >= rand_nr:
                    if indi not in new_pop.list_:
                        new_pop.list_.append(indi)
            new_pop = self._sort_pop(n)
        return new_pop

    def _sort_pop(self, n=None):
        if n is None:
            n = len(self.list_)
        pop = Population(self.net_param)
        best = sorted(self.list_, key=lambda x: x.acc, reverse=True)[:n]
        pop.list_.extend(best)
        return pop

    def __repr__(self):
        str_ = "Population:\n"
        for nr, net_data in enumerate(self.list_):
            str_ += ' \t' + str(nr) + ": " + str(net_data) + '\n'
        str_ += "\t___"
        return str_

    def print_to_restore(self):
        print('[', end='')
        for net_data in self.list_:
            print(net_data.list_, end=',\n')
        print(']')
