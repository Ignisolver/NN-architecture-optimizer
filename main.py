from time import time

from constans_and_types import SizeParams
from genetic_algorithm.genetic_algorithm import PopParam, GeneticAlgorithm
from neuronal_network import NN

if __name__ == "__main__":
    nn = NN([300, 200, 100, 300, 200, 100, 300, 60])
    nn.model.summary()
    nn.save_model("hms_ready_model")
    # net_par = SizeParams(min_=5, max_=10, step=1)
    # f_l_size = (300, 2)
    # lay_par = SizeParams(min_=60, max_=300, step=20)
    # pop_par = PopParam(size=10, n_best=6, leave=2, cross=3,
    #                    mut_in_pop=2, cross_mut=3, mut_in_indi=2)
    # ga = GeneticAlgorithm(net_par, f_l_size, lay_par, 100, pop_par)
    # start = time()
    # res = ga.run_algorithm()
    # print("Time: ", time()-start)
    # print(res)