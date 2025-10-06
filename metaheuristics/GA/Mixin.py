from interface.Evaluator import Evaluator
from interface.Solution import Solution
import random
import numbers
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Set

G = TypeVar('G', bound=numbers.Number)
F = TypeVar('F')
Chromosome = list
Population = List[Chromosome[G]]


"""
 <- Mixin Base Classes ->
"""
class LatinHypercubeInitializerMixin(Generic[G, F]):
    """
    Overrides the default _initialize_population with LHS.
    """
    def _initialize_population(self) -> Population[G]:
        print("Using Latin Hypercube initialization.")
        
        if self.pop_size % 2 != 0:
            print(f"Warning: Pop size {self.pop_size} is odd. Adjusting to {self.pop_size+1} for balance.")
            self.pop_size += 1
             
        population: Population[G] = [[] for _ in range(self.pop_size)]

        for d in range(self.chromosome_size):
            gene_column = [0] * (self.pop_size // 2) + [1] * (self.pop_size - (self.pop_size // 2))
            
            self.rng.shuffle(gene_column)
            
            for i in range(self.pop_size):
                population[i].append(gene_column[i])

        return population[:self.pop_size]


class StochasticUniversalSelectionMixin(Generic[G, F]):
    """
    Overrides the default Tournament Selection with Stochastic Universal Selection (SUS).
    """
    def _select_parents(self, population: Population[G]) -> Population[G]:
        """
        Implementation of SUS selection.
        """
        print("Using Stochastic Universal Selection.")
        # TODO: implement this method
        return


class UniformCrossoverMixin(Generic[G, F]):
    """
    Overrides the default crossover with Stochastic Uniform Crossover.
    """
    def _crossover(self):
        return
