from dataclasses import dataclass
from functools import cache
from time import time, sleep
from typing import Tuple

from constans_and_types import SizeParams
from genetic_algorithm.population import Population
from genetic_algorithm.net_data import NetworkData
from neuronal_network import NN


@dataclass
class PopParam:
    size: int
    n_best: int
    leave: int
    cross: int
    mut_in_pop: int
    cross_mut: int
    mut_in_indi: int

    def __post_init__(self):
        assert sum([self.leave, self.cross,
                    self.mut_in_pop, self.cross_mut]) == self.size


class GeneticAlgorithm:
    def __init__(self, net_param,  f_l_size,layer_param, num_epoch,
                 pop_par: PopParam):
        self.net_param = net_param
        self.f_l_size = f_l_size
        self.layer_param = layer_param
        self.num_epoch = num_epoch
        self.pop_par = pop_par
        self.tested = {}

    def run_algorithm(self):
        pop = Population(net_param=self.net_param)
        pop.random_initialize(self.pop_par.size, self.f_l_size,
                              self.layer_param)
        self._evaluate_population(pop)
        for e_nr in range(self.num_epoch):
            best = pop.get_best(self.pop_par.n_best)
            print("BEST: ", best.list_[0])
            leave = best.get_best(self.pop_par.leave)
            cross = best.do_crossing(self.pop_par.cross)
            cross_mut = best.do_crossing(self.pop_par.cross_mut)
            cross_mut = cross_mut.do_mutations(n_in_indi=
                                               self.pop_par.mut_in_indi)
            mut = best.do_mutations(self.pop_par.mut_in_pop,
                                    self.pop_par.mut_in_indi)
            pop = leave + mut + cross + cross_mut
            print(f'\nGA epoch: {e_nr}: ', end='')
            self._evaluate_population(pop)
        return pop.get_best(1).list_[0]

    def _evaluate_population(self, pop: Population):
        for indi in pop:
            if indi.new:
                acc = self._evaluate_network(indi)
                indi.acc = acc
            print('.', end='')
        print()

    @staticmethod
    @cache
    def _evaluate_network(net: NetworkData):
        nn = NN(net)
        nn.train_network()
        # todo change
        # acc = nn.evaluate_network()
        acc = sum(net.list_)
        return acc


if __name__ == "__main__":
    net_par = SizeParams(min_=5, max_=10, step=1)
    f_l_size = (300, 2)
    lay_par = SizeParams(min_=60, max_=300, step=20)
    pop_par = PopParam(size=10, n_best=6, leave=2, cross=3,
                       mut_in_pop=2, cross_mut=3, mut_in_indi=2)
    ga = GeneticAlgorithm(net_par, f_l_size, lay_par, 100, pop_par)
    start = time()
    res = ga.run_algorithm()
    print("Time: ", time()-start)
    print("RESULT: ",  res)
