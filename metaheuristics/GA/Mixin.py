from interface.Evaluator import Evaluator
from interface.Solution import Solution
import random
import time
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
    """EVOL2: Seleção por Stochastic Universal Selection (SUS)."""
    def _select_parents(self, population: Population[G]) -> Population[G]:
        fitnesses = [self._fitness(ind) for ind in population]
        # shift para garantir não-negatividade
        fmin = min(fitnesses)
        eps = 1e-12
        if fmin < 0:
            fitnesses = [f - fmin + eps for f in fitnesses]
        total = sum(fitnesses)
        if total <= 0 or any((f is None) for f in fitnesses):
            # fallback: cópia direta (evita crash em casos degenerados)
            return list(population)

        step = total / self.pop_size
        start = self.rng.uniform(0.0, step)
        pointers = [start + i * step for i in range(self.pop_size)]

        parents: Population[G] = []
        cum = 0.0
        idx = 0
        for ind, fit in zip(population, fitnesses):
            cum += fit
            while idx < self.pop_size and cum >= pointers[idx]:
                parents.append(ind)
                idx += 1

        # ajuste de tamanho
        if len(parents) < self.pop_size:
            parents.extend(population[:self.pop_size - len(parents)])
        elif len(parents) > self.pop_size:
            parents = parents[:self.pop_size]
        return parents


class UniformCrossoverMixin(Generic[G, F]):
    """EVOL3: Crossover uniforme """
    def _crossover(self, parents: Population[G]) -> Population[G]:
        offspring: Population[G] = []
        self.rng.shuffle(parents)
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < self.pop_size else parents[0]
            c1, c2 = [], []
            for j in range(self.chromosome_size):
                if self.rng.random() < 0.5:
                    c1.append(p1[j]); c2.append(p2[j])
                else:
                    c1.append(p2[j]); c2.append(p1[j])
            offspring.extend([Chromosome(c1), Chromosome(c2)])
        return offspring[:self.pop_size]
    


class SteadyStateSolveMixin(Generic[G, F]):
    """ EVOL4: Estratégia steady-state incremental (μ + λ pequeno)."""

    def solve(self):
        start = time.time()
        self._timed_out = False
        time_limit = getattr(self, "time_limit_s", None)

        offspring_per_iter = max(1, int(getattr(self, "offspring_per_iter", 2)))
        worst_p = float(getattr(self, "worst_p", 0.5))
        worst_p = min(1.0, max(0.0, worst_p))
        no_duplicates = bool(getattr(self, "no_duplicates", True))

        population = self._initialize_population()

        self.best_chromosome = self._get_best_chromosome(population)
        self.best_sol = self._decode(self.best_chromosome)
        if getattr(self, "verbose", True):
            print(f"(Gen. 0) BestSol = {self.best_sol}")

        def as_key(ind): return tuple(ind)
        pop_keys: Set = set(as_key(ind) for ind in population)

        g = 0
        while g < self.generations:
            g += 1
            if time_limit is not None and (time.time() - start) >= time_limit:
                self._timed_out = True
                if getattr(self, "verbose", True):
                    print(f"(Gen. {g}) Interrompido por limite de tempo ({time_limit:.3f}s).")
                break

            produced = 0
            while produced < offspring_per_iter:
                mating_pool = self._select_parents(population)
                if len(mating_pool) < 2:
                    mating_pool = list(population)

                p1 = mating_pool[self.rng.randrange(len(mating_pool))]
                p2 = mating_pool[self.rng.randrange(len(mating_pool))]

                parents_stub = []
                for _ in range(max(2, self.pop_size // 2)):
                    parents_stub.extend([p1, p2])
                kids = self._crossover(parents_stub)[:2]
                kids = self._mutate(kids)

                for child in kids:
                    if time_limit is not None and (time.time() - start) >= time_limit:
                        self._timed_out = True
                        if getattr(self, "verbose", True):
                            print(f"(Gen. {g}) Interrompido por limite de tempo ({time_limit:.3f}s).")
                        break

                    if no_duplicates and as_key(child) in pop_keys:
                        continue

                    best_idx = max(range(len(population)), key=lambda i: self._fitness(population[i]))

                    K = max(1, int(len(population) * worst_p))
                    worst_order = sorted(range(len(population)),
                                         key=lambda i: self._fitness(population[i]))
                    removal_pool = [i for i in worst_order[:K] if i != best_idx]
                    if not removal_pool:
                        removal_pool = [i for i in worst_order if i != best_idx]

                    rm_idx = self.rng.choice(removal_pool)

                    old_key = as_key(population[rm_idx])
                    if old_key in pop_keys:
                        pop_keys.remove(old_key)
                    population[rm_idx] = child
                    pop_keys.add(as_key(child))

                    produced += 1
                    if produced >= offspring_per_iter:
                        break

            current_best = self._get_best_chromosome(population)
            if self._fitness(current_best) > self.best_sol.cost:
                self.best_chromosome = current_best
                self.best_sol = self._decode(self.best_chromosome)
                if getattr(self, "verbose", True):
                    print(f"(Gen. {g}) BestSol = {self.best_sol}")

        return self.best_sol

        

class TimeLimitedSolveMixin(Generic[G, F]):
    def solve(self):
        start = time.time()
        self._timed_out = False
        time_limit = getattr(self, "time_limit_s", None)

        population = self._initialize_population()

        self.best_chromosome = self._get_best_chromosome(population)
        self.best_sol = self._decode(self.best_chromosome)
        print(f"(Gen. 0) BestSol = {self.best_sol}")

        for g in range(1, self.generations + 1):
            if time_limit is not None and (time.time() - start) >= time_limit:
                self._timed_out = True
                if getattr(self, "verbose", True):
                    print(f"(Gen. {g}) Interrompido por limite de tempo ({time_limit:.3f}s).")
                break

            parents = self._select_parents(population)
            offspring = self._crossover(parents)
            mutants = self._mutate(offspring)
            population = self._select_new_population(mutants)

            current_best_chromosome = self._get_best_chromosome(population)
            if self._fitness(current_best_chromosome) > self.best_sol.cost:
                self.best_chromosome = current_best_chromosome
                self.best_sol = self._decode(self.best_chromosome)
                if getattr(self, "verbose", True):
                    print(f"(Gen. {g}) BestSol = {self.best_sol}")

        return self.best_sol