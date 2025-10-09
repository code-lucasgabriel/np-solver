from interface.Evaluator import Evaluator
from interface.Solution import Solution
from typing import List, Set
import numpy as np

class SCQBF(Evaluator[int]):
    """
    A Setc-Cover Quadratic Binary Function (SCQBF) problem evaluator.

    The function is of the form f(x) = x' * A * x, where x is a binary
    vector and A is a matrix of coefficients, where coefficient aij indicates the cost of addint set Si in conjunction with set Sj, all whilst subject to set-cover conditions (i.e, the union of all chosen sets has to be the universe set of the problem). This class loads the matrix A
    from a file and provides methods to evaluate solutions according to the
    SCQBF formula, implementing the Evaluator interface.
    """
    def __init__(self, filename: str):
        self.A: np.ndarray
        self.subsets: List[Set[int]]
        self.size = self._read_instance(filename)
        self.variables = np.zeros(self.size)

    
    def _read_instance(self, filename: str) -> int:
        """
        Reads a MAX-SC-QBF instance from a file.
        """
        with open(filename, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]

        # Read n
        _size: int = int(lines[0])

        # Read sizes of each subset
        subset_sizes: List[int] = list(map(int, lines[1].split()))
        if len(subset_sizes) != _size:
            raise ValueError(f"Expected {_size} subset sizes, got {len(subset_sizes)}.")

        # Read subsets
        self.subsets: List[Set[int]] = []
        idx = 2
        for size in subset_sizes:
            elements = set(map(int, lines[idx].split()))
            if len(elements) != size:
                raise ValueError(f"Expected {size} elements in subset, but got {len(elements)}.")
            self.subsets.append(elements)
            idx += 1

        # Read upper triangular matrix
        self.A: List[List[float]] = [[0.0] * _size for _ in range(_size)]
        row = 0
        while idx < len(lines) and row < _size:
            values = list(map(float, lines[idx].split()))
            for col, val in enumerate(values, start=row):
                if col >= _size:
                    break
                self.A[row][col] = val
            idx += 1
            row += 1
        return _size
    
    
    def get_domain_size(self):
        return self.size


    @staticmethod
    def _tri(A: List[List[float]], i: int, j: int) -> float:
        return A[i][j] if i <= j else A[j][i]

    def _genes_of(self, sol: Solution[int]) -> List[int]:
        g = [0] * self.size
        for i in sol:
            g[i] = 1
        return g

    # ----- função objetivo (maximização) -----
    def value_of(self, genes: List[int]) -> float:
        A, n = self.A, self.size
        val = 0.0
        for i in range(n):
            if genes[i]:
                val += A[i][i]
                for j in range(i + 1, n):
                    if genes[j]:
                        val += self._tri(A, i, j)
        return val

    def delta_insert(self, i: int, genes: List[int]) -> float:
        if genes[i]:
            return 0.0
        A, n = self.A, self.size
        s = A[i][i]
        for j in range(n):
            if j != i and genes[j]:
                s += self._tri(A, i, j)
        return s

    def delta_remove(self, i: int, genes: List[int]) -> float:
        if not genes[i]:
            return 0.0
        A, n = self.A, self.size
        s = -A[i][i]
        for j in range(n):
            if j != i and genes[j]:
                s -= self._tri(A, i, j)
        return s

    # ----- cobertura (set-cover) -----
    def build_cover_count(self, sol: Solution[int]) -> List[int]:
        cover = [0] * self.size
        for i in sol:
            for k in self.subsets[i]:
                cover[k] += 1
        return cover

    @staticmethod
    def is_feasible_cover(cover: List[int]) -> bool:
        return all(c > 0 for c in cover)

    @staticmethod
    def first_uncovered(cover: List[int]) -> int:
        for k, c in enumerate(cover):
            if c == 0:
                return k
        return -1

    def removal_breaks(self, i: int, cover: List[int]) -> bool:
        return any(cover[k] == 1 for k in self.subsets[i])

    def evaluate(self, sol: Solution[int]) -> float:
        genes = self._genes_of(sol)
        sol.cost = self.value_of(genes)
        return sol.cost

    def evaluate_insertion_cost(self, elem: int, sol: Solution[int]) -> float:
        genes = self._genes_of(sol)
        if genes[elem] == 1:
            return 0.0
        delta_val = self.delta_insert(elem, genes)
        return -delta_val
    
    def set_variables(self, sol: Solution[int]) -> None:
        """
        Converts a Solution (list of indices) into a binary vector representation.
        """
        self.variables.fill(0.0)
        if sol:
            self.variables[sol] = 1.0

    def _isFeasible(self):
        """
        Calculates if a given solution is feasible, given the set-cover constrains
        """
        unionSet = set()
        for i, s in enumerate(self.subsets):
            if self.variables[i]:
                unionSet.update(self.subsets[i])
        if len(unionSet) == self.size:
            return True
        return False

    def _isFeasibleRemoval(self, idx: int):
        """
        Calculates if a given solution is feasible after removing set at index idx, given the set-cover constrains
        """
        unionSet = set()
        for i, s in enumerate(self.subsets):
            if i==idx:
                continue
            if self.variables[i]:
                unionSet.update(self.subsets[i])
        if len(unionSet) == self.size:
            return True
        return False
    
    def _isFeasibleExchange(self, idx_rem: int, idx_add: int):
        """
        Calculates if a given solution is feasible after removing set at index idx, given the set-cover constrains
        """
        unionSet = set()
        for i, s in enumerate(self.subsets):
            if i==idx_rem:
                continue
            if i==idx_add:
                unionSet.update(self.subsets[i])
                continue
            if self.variables[i]:
                unionSet.update(self.subsets[i])
        if len(unionSet) == self.size:
            return True
        return False

    def evaluate(self, sol: Solution[int]) -> float:
        """
        Evaluates a solution by computing x' * A * x.
        This method updates the solution's cost.
        """
        self.set_variables(sol)
        if not self._isFeasible(sol):
            cost = float('inf')
            sol.cost = cost
            return cost
        cost = self.evaluate_scqbf()
        sol.cost = cost
        return cost
    
    def evaluate_scqbf(self) -> float:
        """
        Calculates the SCQBF value using efficient NumPy matrix multiplication.
        """
        # The expression `self.variables @ self.A @ self.variables` is the
        # NumPy equivalent of the matrix multiplication x' * A * x.
        return self.variables @ self.A @ self.variables

    def evaluate_insertion_cost(self, elem: int, sol: Solution[int]) -> float:
        """Calculates the change in cost if `elem` is added to the solution."""
        self.set_variables(sol)
        return self._evaluate_insertion_scqbf(elem)

    def _evaluate_insertion_scqbf(self, i: int) -> float:
        """Helper to find the marginal cost of adding variable `i`."""
        if self.variables[i] == 1:
            return 0.0
        return self._evaluate_contribution_scqbf(i)

    def evaluate_removal_cost(self, elem: int, sol: Solution[int]) -> float:
        """Calculates the change in cost if `elem` is removed from the solution."""
        self.set_variables(sol)
        return self._evaluate_removal_scqbf(elem)

    def _evaluate_removal_scqbf(self, i: int) -> float:
        """Helper to find the marginal cost of removing variable `i`."""
        if self.variables[i] == 0:
            return 0.0
        if not self._isFeasibleRemoval(i):
            return float('inf')
        return -self._evaluate_contribution_scqbf(i)
        
    def _evaluate_contribution_scqbf(self, i: int) -> float:
        """
        Calculates the total contribution of variable `i` to the objective function.
        """
        # Sum of off-diagonal terms: Sum_{j!=i} x_j * (A_ij + A_ji)
        term1 = np.dot(self.variables, self.A[i, :] + self.A[:, i])
        # We included the diagonal term A_ii in the dot product, so subtract it
        # once if x_i is 1, and add the single diagonal term A_ii back.
        # This simplifies to term1 - x_i*(A_ii + A_ii) + A_ii
        contribution = term1 - self.variables[i] * (self.A[i, i] + self.A[i, i]) + self.A[i, i]
        return contribution

    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution[int]) -> float:
        """Calculates the change in cost for swapping two elements."""
        self.set_variables(sol)
        
        if elem_in == elem_out:
            return 0.0
        if self.variables[elem_in] == 1.0: # elem_in is already in solution
            return self._evaluate_removal_scqbf(elem_out)
        if self.variables[elem_out] == 0.0: # elem_out is not in solution
            return self._evaluate_insertion_qbf(elem_in)
            
        # Cost change is: (cost to add 'in') - (cost to remove 'out') - (interaction term)
        sum_delta = 0.0
        sum_delta += self._evaluate_contribution_scqbf(elem_in)
        sum_delta -= self._evaluate_contribution_scqbf(elem_out)
        sum_delta -= (self.A[elem_in, elem_out] + self.A[elem_out, elem_in])
        return sum_delta
