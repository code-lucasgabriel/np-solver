from interface.Evaluator import Evaluator
from interface.Solution import Solution
from typing import List, Set
import numpy as np

class SCQBF(Evaluator[int]):
    """
    A Set-Cover Quadratic Binary Function (SCQBF) problem evaluator.

    The function is of the form f(x) = x' * A * x, where x is a binary
    vector and A is a matrix of coefficients, where coefficient aij indicates 
    the cost of adding set Si in conjunction with set Sj, all whilst subject 
    to set-cover conditions (i.e, the union of all chosen sets has to be 
    the universe set of the problem). This class loads the matrix A from a 
    file and provides methods to evaluate solutions according to the SCQBF 
    formula, implementing the Evaluator interface.
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

        _size: int = int(lines[0])

        subset_sizes: List[int] = list(map(int, lines[1].split()))
        if len(subset_sizes) != _size:
            raise ValueError(f"Expected {_size} subset sizes, got {len(subset_sizes)}.")

        self.subsets: List[Set[int]] = []
        idx = 2
        for size in subset_sizes:
            elements = set(map(int, lines[idx].split()))
            if len(elements) != size:
                raise ValueError(f"Expected {size} elements in subset, but got {len(elements)}.")
            self.subsets.append(elements)
            idx += 1

        self.A = np.zeros((_size, _size))
        row = 0
        while idx < len(lines) and row < _size:
            values = list(map(float, lines[idx].split()))
            for col, val in enumerate(values, start=row):
                if col >= _size:
                    break
                self.A[row][col] = val
            idx += 1
            row += 1
        self.A = self.A + self.A.T - np.diag(self.A.diagonal())
        
        return _size

    def get_domain_size(self):
        return self.size

    def set_variables(self, sol: Solution[int]) -> None:
        """
        Converts a Solution (list of indices) into a binary vector representation.
        """
        self.variables.fill(0.0)
        if sol:
            self.variables[sol] = 1.0

    def _isFeasible(self):
        """
        Calculates if a given solution is feasible, given the set-cover constrains.
        """
        unionSet = set()
        for i in np.where(self.variables == 1)[0]:
            unionSet.update(self.subsets[i])
        
        return len(unionSet) == self.size

    def _isFeasibleRemoval(self, idx: int):
        """
        Calculates if a given solution is feasible after removing set at index idx.
        """
        unionSet = set()
        for i in np.where(self.variables == 1)[0]:
            if i == idx:
                continue
            unionSet.update(self.subsets[i])
        
        return len(unionSet) == self.size

    def _isFeasibleExchange(self, idx_rem: int, idx_add: int):
        """
        Calculates if a solution is feasible after exchanging idx_rem with idx_add.
        This implementation is robust and avoids ambiguity by explicitly building
        the set of indices for the proposed new solution.
        """
        new_solution_indices = {i for i, v in enumerate(self.variables) if v == 1}
        new_solution_indices.remove(idx_rem)
        new_solution_indices.add(idx_add)

        unionSet = set()
        for i in new_solution_indices:
            unionSet.update(self.subsets[i])

        return len(unionSet) == self.size

    def evaluate(self, sol: Solution[int]) -> float:
        """
        Evaluates a solution by computing x' * A * x.
        This method updates the solution's cost.
        """
        self.set_variables(sol)
        if not self._isFeasible():
            cost = float('inf')
        else:
            cost = self.evaluate_scqbf()
        
        sol.cost = cost
        return cost

    def evaluate_scqbf(self) -> float:
        """
        Calculates the SCQBF value using NumPy matrix multiplication.
        """
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
        Contribution = A_ii + sum_{j!=i} (A_ij + A_ji) * x_j
        Since A is now symmetric, A_ij + A_ji = 2 * A_ij.
        """
        contribution = 2 * np.dot(self.A[i, :], self.variables) - self.A[i, i]
        return contribution

    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution[int]) -> float:
        """Calculates the change in cost for swapping two elements."""
        self.set_variables(sol)
        
        if elem_in == elem_out:
            return 0.0
        if self.variables[elem_in] == 1.0: # elem_in is already in solution
            return self._evaluate_removal_scqbf(elem_out)
        if self.variables[elem_out] == 0.0: # elem_out is not in solution
            return self._evaluate_insertion_scqbf(elem_in)
        
        if not self._isFeasibleExchange(elem_out, elem_in):
            return float('inf')

        sum_delta = 0.0
        sum_delta += self._evaluate_contribution_scqbf(elem_in)
        sum_delta -= self._evaluate_contribution_scqbf(elem_out)
        sum_delta -= 2 * self.A[elem_in, elem_out]
        return sum_delta