from interface.Evaluator import Evaluator
from interface.Solution import Solution
import abc
import random
from collections import deque
from typing import Deque, Generic, List, TypeVar

E = TypeVar('E')

class TS(abc.ABC, Generic[E]):
    """
    Abstract base class for the Tabu Search (TS) metaheuristic.

    This class provides a template for implementing Tabu Search for minimization
    problems. It includes the main search loop, management of the tabu list, and
    hooks for problem-specific implementations like neighborhood moves and
    candidate list generation.

    Attributes:
        verbose (bool): A flag to control the printing of detailed search progress.
    """

    verbose: bool = True
    _rng: random.Random = random.Random(0)

    def __init__(self, obj_function: Evaluator[E], tenure: int, iterations: int):
        """
        Initializes the AbstractTS solver.

        Args:
            obj_function (Evaluator[E]): The objective function to be minimized.
            tenure (int): The tabu tenure, defining how many iterations a move is
                          considered "tabu".
            iterations (int): The total number of iterations for the main search loop.
        """
        self.obj_function: Evaluator[E] = obj_function
        self.tenure: int = tenure
        self.iterations: int = iterations
        self.best_sol: Solution[E]
        self.sol: Solution[E]
        self.cl: List[E]
        self.rcl: List[E]
        self.tl: Deque[E]

    @abc.abstractmethod
    def make_cl(self) -> List[E]:
        """
        Creates the initial Candidate List (CL).

        The CL contains all possible elements that can be part of a solution.

        Returns:
            List[E]: The generated Candidate List.
        """
        pass

    @abc.abstractmethod
    def make_rcl(self) -> List[E]:
        """
        Creates an empty Restricted Candidate List (RCL).

        The RCL will be populated during the search with the best candidate moves.

        Returns:
            List[E]: An empty Restricted Candidate List.
        """
        pass

    @abc.abstractmethod
    def make_tl(self) -> Deque[E]:
        """
        Creates an empty Tabu List (TL).

        The TL stores moves that are forbidden for a certain number of iterations
        (defined by the tenure).

        Returns:
            Deque[E]: An empty Tabu List.
        """
        pass

    @abc.abstractmethod
    def update_cl(self) -> None:
        """
        Updates the Candidate List based on the current solution.

        This method is responsible for managing which elements are still
        viable candidates to be included in the solution.
        """
        pass

    @abc.abstractmethod
    def create_empty_sol(self) -> Solution[E]:
        """
        Creates a new, empty solution.

        Returns:
            Solution[E]: An empty solution object.
        """
        pass

    @abc.abstractmethod
    def neighborhood_move(self) -> Solution[E]:
        """
        Performs the local search step by exploring the neighborhood of the
        current solution.

        This method defines the core logic of the local search, including how
        neighboring solutions are generated and evaluated. It should also respect
        the tabu list to prevent cycling.

        Returns:
            Solution[E]: The new solution after performing a move.
        """
        pass

    def _constructive_stop_criteria(self, previous_cost: float) -> bool:
        """
        Standard stopping criterion for the constructive heuristic.

        The heuristic stops when the current solution's cost is no longer
        improving (i.e., less than the previous cost).

        Args:
            previous_cost (float): The cost of the solution in the previous step.

        Returns:
            bool: True if the stopping criterion is met, False otherwise.
        """
        return self.sol.cost >= previous_cost

    def constructive_heuristic(self) -> Solution[E]:
        """
        Builds an initial feasible solution using a greedy approach.

        This heuristic iteratively adds the best candidate element to the solution
        until no further improvement can be made.

        Returns:
            Solution[E]: A feasible initial solution.
        """
        self.cl = self.make_cl()
        self.rcl = self.make_rcl()
        self.sol = self.create_empty_sol()
        cost = float('inf')

        # Main loop, continues as long as the solution cost is improving
        while not self._constructive_stop_criteria(cost):
            cost = self.sol.cost
            self.update_cl()

            min_cost = float('inf')

            # Find the minimum cost of inserting a candidate
            for c in self.cl:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost < min_cost:
                    min_cost = delta_cost

            # Add all candidates with the minimum insertion cost to the RCL
            for c in self.cl:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost <= min_cost:
                    self.rcl.append(c)

            # Choose a candidate randomly from the RCL and add it to the solution
            if not self.rcl:
                break # No more candidates to add
                
            in_cand = self._rng.choice(self.rcl)
            self.cl.remove(in_cand)
            self.sol.append(in_cand)
            self.obj_function.evaluate(self.sol)
            self.rcl.clear()

        return self.sol

    def solve(self) -> Solution[E]:
        """
        The main Tabu Search loop.

        It starts by creating an initial solution with a constructive heuristic,
        then iteratively improves it using neighborhood moves while respecting
        the tabu list. The best solution found is stored and returned.

        Returns:
            Solution[E]: The best feasible solution found by the search.
        """
        self.best_sol = self.create_empty_sol()
        self.constructive_heuristic()
        self.tl = self.make_tl()

        for i in range(self.iterations):
            self.neighborhood_move()
            if self.sol.cost < self.best_sol.cost:
                self.best_sol = Solution(self.sol)
                if self.verbose:
                    print(f"(Iter. {i}) BestSol = {self.best_sol}")

        return self.best_sol