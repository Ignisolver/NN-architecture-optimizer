from dataclasses import dataclass
from functools import cache

from constans_and_types import Sets
from data_loader import load_data
from .population import Population
from .net_data import NetworkData
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
    def __init__(self, net_param, f_l_size,layer_param, num_epoch,
                 pop_par: PopParam):
        self.net_param = net_param
        self.f_l_size = f_l_size
        self.layer_param = layer_param
        self.num_epoch = num_epoch
        self.pop_par = pop_par
        self.tested = {}
        self.sets = Sets(*load_data())


    def run_algorithm(self):
        pop = Population(net_param=self.net_param)
        pop.random_initialize(self.pop_par.size, self.f_l_size,
                              self.layer_param)
        self._evaluate_population(pop)
        for e_nr in range(self.num_epoch):
            best = pop.get_best_stat(self.pop_par.n_best)
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

    @cache
    def _evaluate_network(self, net: NetworkData):
        nn = NN(net)
        nn.train_network(self.sets.trainX, self.sets.trainY)
        acc = nn.evaluate_network(self.sets.testX, self.sets.testY)
        return acc

