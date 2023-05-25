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

    net_par = SizeParams(min_=5, max_=9, step=1)
    f_l_size = (-1, -1)
    lay_par = SizeParams(min_=32, max_=512, step=32)
    pop_par = PopParam(size=10, cross=3,
                       mut_in_pop=2, cross_mut=5, mut_in_indi=1)
    ga = GeneticAlgorithm(net_par, f_l_size, lay_par, pop_par, num_epoch=50)

    start = time()
    res = ga.run_algorithm()
    print("Time: ", time()-start)
    print(res)