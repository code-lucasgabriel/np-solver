from metaheuristics.GA.GA import GA, Chromosome
from metaheuristics.GA.Mixin import LatinHypercubeInitializerMixin
from problems.QBF.QBF import QBF
from interface.Solution import Solution
import time

class GA_QBF(
    GA[int, int]
):
    """
    A specific implementation of the Genetic Algorithm for solving the
    Quadratic Binary Function (QBF) problem.
    """

    def __init__(self, generations: int, pop_size: int, mutation_rate: float, filename: str):
        """
        Constructor for the GA_QBF solver.

        Args:
            generations (int): Maximum number of generations.
            pop_size (int): Size of the population.
            mutation_rate (float): The mutation rate.
            filename (str): Path to the QBF instance file.
        """
        # The constructor can raise IO exceptions if the file is not found.
        qbf_instance = QBF(filename)
        super().__init__(qbf_instance, generations, pop_size, mutation_rate)

    def _decode(self, chromosome: Chromosome[int]) -> Solution[int]:
        """
        Decodes a binary chromosome into a QBF solution.

        The solution is represented by the list of indices where the
        chromosome has a value of 1.
        """
        solution = Solution[int]()
        for locus, gene in enumerate(chromosome):
            if gene == 1:
                solution.append(locus)
        
        self.obj_function.evaluate(solution)
        return solution
        