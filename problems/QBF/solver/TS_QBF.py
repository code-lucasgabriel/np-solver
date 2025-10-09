from metaheuristics.TS import TS
from problems.QBF.QBF_Inverse import QBF_Inverse
from interface.Solution import Solution
import time
from collections import deque
from typing import Deque, List, Optional


class TS_QBF(TS[int]):
    """
    Tabu Search (TS) metaheuristic for solving the Quadratic Binary Function (QBF) problem.

    This implementation uses an inverse QBF instance to frame the problem as a
    minimization task. The neighborhood moves include insertion, removal, and
    exchange of variables in the solution.
    """

    def __init__(self, tenure: int, iterations: int, filename: str):
        """
        Initializes the TS_QBF solver.

        Args:
            tenure (int): The tabu tenure, defining how many iterations a move is
                          considered "tabu".
            iterations (int): The total number of iterations for the main search loop.
            filename (str): The path to the file containing the QBF instance.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        super().__init__(QBF_Inverse(filename), tenure, iterations)
        # A placeholder value for the tabu list, representing no move.
        self._fake: int = -1
        # The candidate list is initialized here as it is static for QBF.
        self.cl = self.make_cl()

    def make_cl(self) -> List[int]:
        """
        Creates the Candidate List (CL) for the QBF problem.

        The CL for QBF consists of all variable indices, from 0 to N-1.

        Returns:
            List[int]: A list containing all variable indices.
        """
        return list(range(self.obj_function.get_domain_size()))

    def make_rcl(self) -> List[int]:
        """
        Creates an empty Restricted Candidate List (RCL). This is not used
        in this specific Tabu Search implementation's neighborhood move.

        Returns:
            List[int]: An empty list.
        """
        return []

    def make_tl(self) -> Deque[int]:
        """
        Creates and initializes the Tabu List (TL).

        The list is filled with placeholder values to match its capacity.
        The capacity is twice the tenure to account for both elements that
        might enter and leave the solution in an exchange move.

        Returns:
            Deque[int]: The initialized Tabu List.
        """
        # The deque size is 2 * tenure to handle both in and out moves.
        tabu_list = deque([self._fake] * (2 * self.tenure), maxlen=(2 * self.tenure))
        return tabu_list

    def update_cl(self) -> None:
        """
        Updates the Candidate List. For the standard QBF problem, this method
        does nothing, as the set of non-solution elements is always the
        complement of the solution elements.
        """
        pass

    def create_empty_sol(self) -> Solution[int]:
        """
        Creates an empty solution for the QBF problem.

        An empty solution corresponds to all binary variables being set to zero,
        which has a known cost of 0.0.

        Returns:
            Solution[int]: An empty solution with its cost initialized to 0.0.
        """
        sol = Solution[int]()
        sol.cost = 0.0
        return sol

    def neighborhood_move(self) -> Solution[int]:
        """
        Performs a neighborhood move by exploring insertions, removals, and exchanges.

        It selects the best move (i.e., the one that results in the lowest
        solution cost) among all non-tabu moves. An aspiration criterion allows
        a tabu move if it leads to a solution better than any found so far.

        Returns:
            Solution[int]: The modified solution after the move.
        """
        min_delta_cost = float('inf')
        best_cand_in: Optional[int] = None
        best_cand_out: Optional[int] = None

        # Elements not in the solution are candidates for insertion.
        non_solution_elements = [c for c in self.cl if c not in self.sol]

        # Evaluate insertions
        for cand_in in non_solution_elements:
            delta_cost = self.obj_function.evaluate_insertion_cost(cand_in, self.sol)
            # Aspiration Criterion: accept move if it leads to a new best solution
            if (cand_in not in self.tl) or (self.sol.cost + delta_cost < self.best_sol.cost):
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in = cand_in
                    best_cand_out = None

        # Evaluate removals
        for cand_out in self.sol:
            delta_cost = self.obj_function.evaluate_removal_cost(cand_out, self.sol)
            if (cand_out not in self.tl) or (self.sol.cost + delta_cost < self.best_sol.cost):
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in = None
                    best_cand_out = cand_out

        # Evaluate exchanges
        for cand_in in non_solution_elements:
            for cand_out in self.sol:
                delta_cost = self.obj_function.evaluate_exchange_cost(cand_in, cand_out, self.sol)
                if ((cand_in not in self.tl) and (cand_out not in self.tl)) or \
                   (self.sol.cost + delta_cost < self.best_sol.cost):
                    if delta_cost < min_delta_cost:
                        min_delta_cost = delta_cost
                        best_cand_in = cand_in
                        best_cand_out = cand_out
        
        # Implement the best move found
        # First, handle the element leaving the solution
        self.tl.popleft()
        if best_cand_out is not None:
            self.sol.remove(best_cand_out)
            self.tl.append(best_cand_out)
        else:
            self.tl.append(self._fake)
        
        # Second, handle the element entering the solution
        self.tl.popleft()
        if best_cand_in is not None:
            self.sol.append(best_cand_in)
            self.tl.append(best_cand_in)
        else:
            self.tl.append(self._fake)

        self.obj_function.evaluate(self.sol)
        return self.sol
