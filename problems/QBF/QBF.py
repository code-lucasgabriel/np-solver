from interface.Evaluator import Evaluator
from interface.Solution import Solution
import numpy as np
import random

class QBF(Evaluator[int]):
    """
    A Quadratic Binary Function (QBF) problem evaluator.

    The function is of the form f(x) = x' * A * x, where x is a binary
    vector and A is a matrix of coefficients. This class loads the matrix A
    from a file and provides methods to evaluate solutions according to the
    QBF formula, implementing the Evaluator interface.
    """

    def __init__(self, filename: str):
        """
        Initializes the QBF problem by reading an instance file.

        Args:
            filename (str): The path to the file containing the QBF instance.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        self.A: np.ndarray
        self.size = self._read_instance(filename)
        self.variables = np.zeros(self.size)

    def _read_instance(self, filename: str) -> int:
        """
        Reads the QBF instance file to populate the matrix A.

        The file format is expected to be the size `N` on the first line,
        followed by the numbers for the upper triangle of matrix A.
        """
        with open(filename, 'r') as file:
            tokens = file.read().split()
            
            _size = int(tokens[0])
            self.A = np.zeros((_size, _size))
            
            token_idx = 1
            for i in range(_size):
                for j in range(i, _size):
                    self.A[i, j] = float(tokens[token_idx])
                    token_idx += 1
        return _size

    def get_domain_size(self) -> int:
        """Returns the number of variables in the problem."""
        return self.size

    def set_variables(self, sol: Solution[int]) -> None:
        """
        Converts a Solution (list of indices) into a binary vector representation.
        """
        self.variables.fill(0.0)
        if sol:
            self.variables[sol] = 1.0

    def evaluate(self, sol: Solution[int]) -> float:
        """
        Evaluates a solution by computing x' * A * x.
        This method updates the solution's cost.
        """
        self.set_variables(sol)
        cost = self.evaluate_qbf()
        sol.cost = cost
        return cost

    def evaluate_qbf(self) -> float:
        """
        Calculates the QBF value using efficient NumPy matrix multiplication.
        """
        # The expression `self.variables @ self.A @ self.variables` is the
        # NumPy equivalent of the matrix multiplication x' * A * x.
        return self.variables @ self.A @ self.variables

    def evaluate_insertion_cost(self, elem: int, sol: Solution[int]) -> float:
        """Calculates the change in cost if `elem` is added to the solution."""
        self.set_variables(sol)
        return self._evaluate_insertion_qbf(elem)

    def _evaluate_insertion_qbf(self, i: int) -> float:
        """Helper to find the marginal cost of adding variable `i`."""
        if self.variables[i] == 1:
            return 0.0
        return self._evaluate_contribution_qbf(i)

    def evaluate_removal_cost(self, elem: int, sol: Solution[int]) -> float:
        """Calculates the change in cost if `elem` is removed from the solution."""
        self.set_variables(sol)
        return self._evaluate_removal_qbf(elem)

    def _evaluate_removal_qbf(self, i: int) -> float:
        """Helper to find the marginal cost of removing variable `i`."""
        if self.variables[i] == 0:
            return 0.0
        return -self._evaluate_contribution_qbf(i)
        
    def _evaluate_contribution_qbf(self, i: int) -> float:
        """
        Calculates the total contribution of variable `i` to the objective function.
        This is faster than re-evaluating the whole solution.
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
            return self._evaluate_removal_qbf(elem_out)
        if self.variables[elem_out] == 0.0: # elem_out is not in solution
            return self._evaluate_insertion_qbf(elem_in)
            
        # Cost change is: (cost to add 'in') - (cost to remove 'out') - (interaction term)
        sum_delta = 0.0
        sum_delta += self._evaluate_contribution_qbf(elem_in)
        sum_delta -= self._evaluate_contribution_qbf(elem_out)
        sum_delta -= (self.A[elem_in, elem_out] + self.A[elem_out, elem_in])
        return sum_delta
