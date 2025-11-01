from np_solver.core.evaluator import BaseEvaluator, ObjectiveSense
from np_solver.core.solution import BaseSolution
import random
import numbers
from typing import TypeVar, Generic, List, Optional, Callable, Any

# Generic type variables for Genotype (G) and Phenotype (F)
G = TypeVar('G', bound=numbers.Number)
F = TypeVar('F')

# Type aliases for Chromosome and Population for better readability
Chromosome = list
Population = List[Chromosome[G]]

class GeneticAlgorithm(Generic[G, F]):
    """
    A concrete implementation of a Genetic Algorithm (GA) metaheuristic.

    This class uses a composition-over-inheritance approach. It is
    initialized with a problem-specific 'evaluator' and 'decode_func'
    to guide the search.
    
    It respects the 'sense' (MINIMIZE/MAXIMIZE) from the evaluator.
    """
    verbose: bool = True
    rng: random.Random = random.Random(0)

    def __init__(
        self,
        evaluator: BaseEvaluator[F],
        decode_func: Callable[[Any, Chromosome[G]], BaseSolution[F]],
        generations: int,
        pop_size: int,
        mutation_rate: float,
    ):
        """
        Initializes the Genetic Algorithm solver.

        Args:
            evaluator (BaseEvaluator[F]): The objective function evaluator.
                This object must contain a 'problem' attribute.
            decode_func (Callable): A function that takes a problem instance 
                and a chromosome, and returns a BaseSolution.
                Signature: (problem: Any, chromosome: Chromosome) -> BaseSolution
            generations (int): The maximum number of generations to execute.
            pop_size (int): The size of the population.
            mutation_rate (float): The probability of mutating a single gene.
        """
        self.evaluator: BaseEvaluator[F] = evaluator
        self._decode_func = decode_func
        self.generations: int = generations
        
        if pop_size < 2:
            print("Warning: adjusting population size to 2 because of insuficient individuals.")
            self.pop_size = 2
        else:
            self.pop_size = pop_size
            
        self.mutation_rate: float = mutation_rate
        
        # Correctly get domain size from the problem object
        self.chromosome_size: int = self.evaluator.problem.get_domain_size()
        
        self.best_sol: Optional[BaseSolution[F]] = None
        self.best_chromosome: Optional[Chromosome[G]] = None

    """
    <- Core Solver Logic ->
    """

    def solve(self) -> Optional[BaseSolution[F]]:
        """
        Executes the main Genetic Algorithm loop.
        
        Returns:
            The best solution found after all generations are complete.
        """
        self._start()

        population = self._initialize_population()

        # Evaluate and set the initial best solution
        self.best_chromosome = self._get_best_chromosome(population)
        self.best_sol = self._decode(self.best_chromosome)
        self.best_sol.cost = self.evaluator.evaluate(self.best_sol)
        print(f"(Gen. 0) BestSol = {self.best_sol}")

        for g in range(1, self.generations + 1):
            parents = self._select_parents(population)
            offspring = self._crossover(parents)
            mutants = self._mutate(offspring)
            population = self._select_new_population(mutants)

            current_best_chromosome = self._get_best_chromosome(population)
            current_best_fitness = self._fitness(current_best_chromosome)

            # Check if this is better, respecting MIN/MAX sense
            if self._is_better(current_best_fitness, self.best_sol.cost):
                self.best_chromosome = current_best_chromosome
                self.best_sol = self._decode(self.best_chromosome)
                self.best_sol.cost = current_best_fitness # Assign the cost we just calculated
                
                if self.verbose:
                    print(f"(Gen. {g}) BestSol = {self.best_sol}")

        return self.best_sol

    """
    <- Helper methods ->
    """

    def _decode(self, chromosome: Chromosome[G]) -> BaseSolution[F]:
        """Calls the user-provided decode function."""
        # Pass the problem instance from the evaluator to the function
        return self._decode_func(self.evaluator.problem, chromosome)

    def _fitness(self, chromosome: Chromosome[G]) -> float:
        """Decodes and evaluates a chromosome, returning its cost."""
        solution = self._decode(chromosome)
        cost = self.evaluator.evaluate(solution)
        solution.cost = cost # Assumes BaseSolution has a 'cost' attribute
        return cost

    def _is_better(self, fitness1: float, fitness2: float) -> bool:
        """Compares two fitness values based on the evaluator's objective sense."""
        if self.evaluator.sense == ObjectiveSense.MAXIMIZE:
            return fitness1 > fitness2
        else: # ObjectiveSense.MINIMIZE
            return fitness1 < fitness2

    def _get_best_chromosome(self, population: Population[G]) -> Chromosome[G]:
        """Finds the 'best' chromosome based on the objective sense."""
        if self.evaluator.sense == ObjectiveSense.MAXIMIZE:
            return max(population, key=self._fitness)
        else:
            return min(population, key=self._fitness)

    def _get_worst_chromosome(self, population: Population[G]) -> Chromosome[G]:
        """Finds the 'worst' chromosome based on the objective sense."""
        if self.evaluator.sense == ObjectiveSense.MAXIMIZE:
            return min(population, key=self._fitness)
        else:
            return max(population, key=self._fitness)

    """
    <- GA Operators (Defaults) ->
    """

    def _initialize_population(self) -> Population[G]:
        """Generates the initial population using simple uniform random generation."""
        return [self._generate_random_chromosome() for _ in range(self.pop_size)]

    def _generate_random_chromosome(self) -> Chromosome[G]:
        """DEFAULT: Generates a binary chromosome."""
        chromosome = [random.randint(0, 1) for _ in range(self.chromosome_size)]
        return chromosome

    def _select_parents(self, population: Population[G]) -> Population[G]:
        """DEFAULT: Selects parents using simple Tournament Selection (k=2)."""
        parents = []
        for _ in range(self.pop_size):
            p1 = self.rng.choice(population)
            p2 = self.rng.choice(population)
            
            fit1 = self._fitness(p1)
            fit2 = self._fitness(p2)
            
            # Select the parent that is 'better'
            parents.append(p1 if self._is_better(fit1, fit2) else p2)
        return parents

    def _crossover(self, parents: Population[G]) -> Population[G]:
        """DEFAULT: Performs 2-point crossover."""
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
        """DEFAULT: Applies bit-flip mutation."""
        for chromosome in offspring:
            for locus in range(self.chromosome_size):
                if self.rng.random() < self.mutation_rate:
                    self._mutate_gene(chromosome, locus)
        return offspring

    def _mutate_gene(self, chromosome: Chromosome[G], locus: int) -> None:
        """DEFAULT: Mutates a gene by flipping its bit."""
        chromosome[locus] = 1 - chromosome[locus]

    def _select_new_population(self, offspring: Population[G]) -> Population[G]:
        """DEFAULT: Elitism - replaces the worst offspring with the best-so-far."""
        if self.best_chromosome is not None:
            worst_offspring = self._get_worst_chromosome(offspring)
            
            # If the best-so-far is better than the worst new offspring
            if self._is_better(self._fitness(self.best_chromosome), self._fitness(worst_offspring)):
                new_population = list(offspring)
                new_population.remove(worst_offspring)
                new_population.append(self.best_chromosome)
                return new_population
        return offspring

    """
    <- Logging and Mixin Control ->
    """
    def _start(self) -> None:
        print("\n========================================")
        print("Starting optimization...")
        # Get problem name from the evaluator's problem instance
        try:
            problem_name = self.evaluator.problem.get_problem_name()
            print(f"Problem: {problem_name}")
        except AttributeError:
            print("Problem: (name not available)")
            
        print("Methaheuristic: Genetic Algorithm")
        self._mixin_ctrl()
        print("========================================\n")

    def _status_lhs(self):
        return False # Placeholder for mixin logic

    def _status_sus(self):
        return False # Placeholder for mixin logic
        
    def _mixin_ctrl(self):
        if self._status_lhs():
            print("Info: Latin Hypercube Sampling initialization activated.")
        if self._status_sus():
            print("Info: Stochastic Universal Selection activated.")