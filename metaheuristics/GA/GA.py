from interface.Evaluator import Evaluator
from interface.Solution import Solution
import random
import numbers
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Set

"""
TODO:
    * latin hypercube sampling
    * stochastic universal selection
    * uniform crossover
    * adaptative mutation
    * steady-state
"""

# Generic type variables for Genotype (G) and Phenotype (F)
G = TypeVar('G', bound=numbers.Number)
F = TypeVar('F')

# Type aliases for Chromosome and Population for better readability
Chromosome = list
Population = List[Chromosome[G]]

class GA(Generic[G, F], ABC):
    """
    Abstract base class for a Genetic Algorithm (GA) metaheuristic.

    This implementation is designed to maximize the chromosome fitness. It provides the core structure for selection, crossover, and mutation, while leaving problem-specific details (like decoding and fitness evaluation) to be implemented in subclasses.
    """
    verbose: bool = True
    rng: random.Random = random.Random(0)

    def __init__(
        self,
        obj_function: Evaluator[F],
        generations: int,
        pop_size: int,
        mutation_rate: float,
    ):
        """
        Initializes the Genetic Algorithm solver.

        Args:
            obj_function (Evaluator[F]): The objective function being optimized.
            generations (int): The maximum number of generations to execute.
            pop_size (int): The size of the population.
            mutation_rate (float): The probability of mutating a single gene.
        """
        self.obj_function: Evaluator[F] = obj_function
        self.generations: int = generations
        if pop_size<2:
            print("Warning: adjusting population size to 2 because of insuficient individuals.")
            self.pop_size = 2
        else:
            self.pop_size = pop_size
        self.mutation_rate: float = mutation_rate
        self.chromosome_size: int = self.obj_function.get_domain_size()
        self.best_sol: Optional[Solution[F]] = None
        self.best_chromosome: Optional[Chromosome[G]] = None

    """
    <- ABSTRACT METHODS (Problem-Specific - MUST be implemented in subclass) ->
    """
    @abstractmethod
    def _decode(self, chromosome: Chromosome[G]) -> Solution[F]: pass
    
    """
    <- Methods with default implementation and Mixins coupling
    """
    def _initialize_population(self) -> Population[G]:
        """
        * DEFAULT: Generates the initial population using simple uniform random generation.
        ! Overridden by an initialization mixin (e.g., LHS).
        """
        return [self._generate_random_chromosome() for _ in range(self.pop_size)]

    def _select_parents(self, population: Population[G]) -> Population[G]:
        """
        * DEFAULT: Selects parents using simple **Tournament Selection** (k=2).
        ! Overridden by a selection mixin (e.g., SUS).
        """
        parents = []
        for _ in range(self.pop_size):
            p1 = self.rng.choice(population)
            p2 = self.rng.choice(population)
            parents.append(p1 if self._fitness(p1) > self._fitness(p2) else p2)
        return parents

    def _status_lhs(self):
        return False

    def _status_sus(self):
        return False
        
    def _mixin_ctrl(self):
        if self._status_lhs():
            print("Info: Latin Hypercube Sampling initialization activated.")
        if self._status_sus():
            print("Info: Stochastic Universal Selection activated.")

    """
    <- Optional methods with default implementation ->
    """

    def _start(self) -> None:
        print("\n========================================")
        print("Starting optimization...")
        print("Problem:", self.obj_function.get_name())
        print("Methaheuristic: Genetic Algorithm")
        self._mixin_ctrl()
        print("========================================\n")
        

    def solve(self) -> Optional[Solution[F]]:
        """
        Executes the main Genetic Algorithm loop.

        This process involves initializing a population and then iterating through generations, performing parent selection, crossover, mutation and population updates to find the best solution.

        Returns:
            The best solution found after all generations are complete.
        """
        self._start()

        population = self._initialize_population()

        self.best_chromosome = self._get_best_chromosome(population)
        self.best_sol = self._decode(self.best_chromosome)
        print(f"(Gen. 0) BestSol = {self.best_sol}")

        for g in range(1, self.generations + 1):
            parents = self._select_parents(population)
            offspring = self._crossover(parents)
            mutants = self._mutate(offspring)
            population = self._select_new_population(mutants)

            current_best_chromosome = self._get_best_chromosome(population)
            if self._fitness(current_best_chromosome) > self.best_sol.cost:
                self.best_chromosome = current_best_chromosome
                self.best_sol = self._decode(self.best_chromosome)
                if self.verbose:
                    print(f"(Gen. {g}) BestSol = {self.best_sol}")

        return self.best_sol

    
    def _generate_random_chromosome(self) -> Chromosome[G]:
        chromosome = [random.randint(0, 1) for _ in range(self.chromosome_size)]
        return chromosome
    
    def _mutate_gene(self, chromosome: Chromosome[G], locus: int) -> None:
        """
        Mutates a gene by flipping its bit (0 becomes 1, and 1 becomes 0).
        """
        chromosome[locus] = 1 - chromosome[locus]

    def _fitness(self, chromosome: Chromosome[G]) -> float:
        # ? Maybe a future mixin implementation of fitness scalling is good
        return self._decode(chromosome).cost

    def _get_best_chromosome(self, population: Population[G]) -> Chromosome[G]:
        """Finds the chromosome with the highest fitness in a population."""
        return max(population, key=self._fitness)

    def _get_worst_chromosome(self, population: Population[G]) -> Chromosome[G]:
        """Finds the chromosome with the lowest fitness in a population."""
        return min(population, key=self._fitness)


    def _crossover(self, parents: Population[G]) -> Population[G]:
        """Performs 2-point crossover on pairs of parents to create offspring."""
        offspring_population = []
        self.rng.shuffle(parents)

        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < self.pop_size else parents[0]

            point1 = self.rng.randint(0, self.chromosome_size)
            point2 = self.rng.randint(0, self.chromosome_size)
            start, end = min(point1, point2), max(point1, point2)

            offspring1 = p1[:start] + p2[start:end] + p1[end:]
            offspring2 = p2[:start] + p1[start:end] + p2[end:]
            offspring_population.extend([Chromosome(offspring1), Chromosome(offspring2)])
            
        return offspring_population[:self.pop_size]

    def _mutate(self, offspring: Population[G]) -> Population[G]:
        """Applies mutation to each gene in the offspring population."""
        for chromosome in offspring:
            for locus in range(self.chromosome_size):
                if self.rng.random() < self.mutation_rate:
                    self._mutate_gene(chromosome, locus)
        return offspring

    def _select_new_population(self, offspring: Population[G]) -> Population[G]:
        """
        Updates the population for the next generation using elitism.

        The worst chromosome from the new offspring is replaced by the best
        chromosome from the previous generation.
        """
        if self.best_chromosome is not None:
            worst_offspring = self._get_worst_chromosome(offspring)
            if self._fitness(worst_offspring) < self._fitness(self.best_chromosome):
                new_population = list(offspring)
                new_population.remove(worst_offspring)
                new_population.append(self.best_chromosome)
                return new_population
        return offspring