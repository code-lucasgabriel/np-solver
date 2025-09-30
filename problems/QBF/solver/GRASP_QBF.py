from metaheuristics.GRASP import GRASP
from problems.QBF.QBF_Inverse import QBF_Inverse
from interface.Solution import Solution
from typing import List


class GRASP_QBF(GRASP[int]):
    """
    A concrete implementation of the GRASP metaheuristic for solving the
    Quadratic Binary Function (QBF) problem.

    Since the AbstractGRASP is designed for minimization, this implementation
    uses the QBF_Inverse objective function.
    """

    def __init__(self, alpha: float, iterations: int, filename: str):
        """
        Initializes the GRASP_QBF solver.

        Args:
            alpha (float): The GRASP greediness-randomness parameter (0 to 1).
            iterations (int): The number of GRASP iterations to perform.
            filename (str): The path to the QBF instance file.
        """
        super().__init__(QBF_Inverse(filename), alpha, iterations)

    def make_cl(self) -> List[int]:
        """
        Creates the Candidate List (CL), which initially contains all possible
        variables (from 0 to N-1).
        """
        return list(range(self.obj_function.get_domain_size()))

    def make_rcl(self) -> List[int]:
        """Creates an empty Restricted Candidate List (RCL)."""
        return []

    def update_cl(self) -> None:
        """
        Does nothing in this implementation, since all elements not in the
        solution are always considered viable candidates.
        """
        pass

    def create_empty_sol(self) -> Solution[int]:
        """
        Creates an empty solution with a cost of 0.0, which is the known
        cost for an all-zero QBF solution.
        """
        sol = Solution[int]()
        sol.cost = 0.0
        return sol

    def local_search(self) -> Solution[int]:
        """
        Performs a local search based on the "best improvement" strategy.

        It iteratively searches for the best possible move (insertion, removal,
        or exchange) and applies it, repeating until no further improvement
        can be made (i.e., a local optimum is reached).
        """
        min_delta_cost: float
        best_cand_in: int | None
        best_cand_out: int | None

        while True:
            min_delta_cost = float('inf')
            best_cand_in, best_cand_out = None, None

            # 1. Evaluate insertion moves
            for cand_in in self.cl:
                delta_cost = self.obj_function.evaluate_insertion_cost(cand_in, self.sol)
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in, best_cand_out = cand_in, None

            # 2. Evaluate removal moves
            for cand_out in self.sol:
                delta_cost = self.obj_function.evaluate_removal_cost(cand_out, self.sol)
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in, best_cand_out = None, cand_out

            # 3. Evaluate exchange moves
            for cand_in in self.cl:
                for cand_out in self.sol:
                    delta_cost = self.obj_function.evaluate_exchange_cost(cand_in, cand_out, self.sol)
                    if delta_cost < min_delta_cost:
                        min_delta_cost = delta_cost
                        best_cand_in, best_cand_out = cand_in, cand_out

            # 4. Implement the best move if it's an improvement
            if min_delta_cost < -1e-9:  # Use a small epsilon for float comparison
                if best_cand_out is not None:
                    self.sol.remove(best_cand_out)
                    self.cl.append(best_cand_out)
                if best_cand_in is not None:
                    self.sol.append(best_cand_in)
                    self.cl.remove(best_cand_in)
                self.obj_function.evaluate(self.sol)
            else:
                # No improving move found, break the loop
                break

        return self.sol
