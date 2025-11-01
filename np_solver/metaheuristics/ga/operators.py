import random
import numbers
from abc import ABC
from typing import TypeVar, Generic, List

G = TypeVar('G', bound=numbers.Number)
F = TypeVar('F')
Chromosome = list
Population = List[Chromosome[G]]


"""
 <- Mixin Base Classes ->
"""
class LatinHypercubeInitializer(Generic[G, F]):
    """
    Overrides the default _initialize_population with LHS.
    """

    def _status_lhs(self):
        return True

    def _initialize_population(self) -> Population[G]:
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


class StochasticUniversalSelection(Generic[G, F]):
    """
    Overrides the default Tournament Selection with Stochastic Universal Selection (SUS).
    """
    def _status_sus(self):
        return True

    def _select_parents(self, population: Population[G]) -> Population[G]:
        """
        Implementation of an efficient SUS selection.
        """
        
        parents = []
        
        fitness_scores = [self._fitness(individual) for individual in population]
        total_fitness = sum(fitness_scores)
        
        num_to_select = self.pop_size
        
        pointer_distance = total_fitness / num_to_select
        start_point = random.uniform(0, pointer_distance)
        
        current_pointer = start_point
        cumulative_fitness = 0
        population_index = 0
        
        for _ in range(num_to_select):
            while cumulative_fitness < current_pointer:
                cumulative_fitness += fitness_scores[population_index]
                population_index += 1
            
            parents.append(population[population_index - 1])
            current_pointer += pointer_distance
            
        return parents
        


class UniformCrossover(Generic[G, F]):
    """
    Overrides the default crossover with Stochastic Uniform Crossover.
    """
    def _crossover(self):
        return
