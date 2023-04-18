from typing import Tuple

from constans_and_types import Population, NetworkData


class GeneticAlgorithm:
    def __init__(self):
        pass

    def run_algorithm(self):
        pass

    def _initialize_population(self) -> Population:
        pass

    def _mutation(self, net: NetworkData):
        """
        First and last Layer schould not be change
        :param net:
        :return:
        """
        pass

    def _cross(self, net_1: NetworkData, net_2: NetworkData):
        """
        First and last Layer schould not be change
        :param net:
        :return:
        """
        pass

    def _evaluate_population(self, pop: Population) -> Tuple[NetworkData,
                                                            float]:
        pass

    def _evaluate_network(self, net: NetworkData):
        pass

