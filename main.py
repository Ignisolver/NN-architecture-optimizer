import os

from time import time

from constans_and_types import SizeParams
from genetic_algorithm.genetic_algorithm import PopParam, GeneticAlgorithm
import tensorflow as tf

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    net_par = SizeParams(min_=4, max_=6, step=1)
    f_l_size = (-1, -1)
    lay_par = SizeParams(min_=50, max_=200, step=50)
    pop_par = PopParam(size=7, cross=2,
                       mut_in_pop=2, cross_mut=3, mut_in_indi=1)
    init_pop = [
        [-1, 200, 150, 100, 50, -1],
        [-1, 200, 200, 200, 200, -1],
        [-1, 100, 100, 100, 100, -1],
        [-1, 150, 150, 150, 150, -1],
        [-1, 50, 50, 50, 50, -1],
        [-1, 150, 150, 150, -1],
        [-1, 100, 100, 100, -1],
        [-1, 200, 200, 200, -1],
        [-1, 50, 50, 50, -1],
        [-1, 200, 200, -1],
        [-1, 150, 150, -1],
        [-1, 100, 100, -1],
        [-1, 50, 50, -1],

    ]


    ga = GeneticAlgorithm(net_par, f_l_size, lay_par, pop_par, num_epoch=50,
                          init_pop=init_pop)

    start = time()
    res = ga.run_algorithm()
    print("Time: ", time()-start)
    print(res)