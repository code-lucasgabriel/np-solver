# problems/SCQBF/solver/GA_SCQBF.py

from typing import Sequence
from metaheuristics.GA.GA import GA
from metaheuristics.GA.Mixin import (
    TimeLimitedSolveMixin,
    LatinHypercubeInitializerMixin,
    StochasticUniversalSelectionMixin,
    UniformCrossoverMixin,
    SteadyStateSolveMixin
)
from interface.Solution import Solution
from problems.SCQBF.SCQBF import SCQBF

class GA_MaxSCQBF_Base(TimeLimitedSolveMixin[int, int], GA[int, int]):
    def __init__(self, generations: int, pop_size: int, mutation_rate: float, filename: str):
        super().__init__(SCQBF(filename), generations, pop_size, mutation_rate)

    def _decode(self, chromosome: Sequence[int]) -> Solution[int]:
        f: SCQBF = self.obj_function
        genes = list(chromosome)

        sol = Solution[int]()
        for i, g in enumerate(genes):
            if g == 1:
                sol.append(i)

        cover = f.build_cover_count(sol)
        while not f.is_feasible_cover(cover):
            k = f.first_uncovered(cover)
            best_i, best_gain = None, float("-inf")
            for i, g in enumerate(genes):
                if g == 0 and k in f.subsets[i]:
                    gain = f.delta_insert(i, genes)
                    if gain > best_gain:
                        best_i, best_gain = i, gain
            if best_i is None:
                raise RuntimeError(f"Nenhum conjunto cobre o elemento {k}.")
            sol.append(best_i)
            genes[best_i] = 1
            for kk in f.subsets[best_i]:
                cover[kk] += 1

        sol.cost = f.value_of(genes)
        return sol

# EVOL1: inicialização Latin Hypercube
class GA_MaxSCQBF_LHS(LatinHypercubeInitializerMixin[int, int], GA_MaxSCQBF_Base):
    pass

# EVOL2: seleção SUS
class GA_MaxSCQBF_SUS(StochasticUniversalSelectionMixin[int, int], GA_MaxSCQBF_Base):
    pass

# EVOL3: crossover uniforme
class GA_MaxSCQBF_UX(UniformCrossoverMixin[int, int], GA_MaxSCQBF_Base):
    pass

# EVOL4: steady-state (μ+λ)
class GA_MaxSCQBF_SS(SteadyStateSolveMixin[int, int], GA_MaxSCQBF_Base):
    pass
