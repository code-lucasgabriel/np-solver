from core.evaluator import BaseEvaluator
from core.solution import BaseSolution
import random
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

# Generic type for the elements that compose a solution
E = TypeVar('E')

class GRASP(Generic[E], ABC):
    """
    Abstract base class for the GRASP (Greedy Randomized Adaptive Search
    Procedure) metaheuristic.

    This class provides a framework for minimization problems, outlining the
    main GRASP loop which alternates between a constructive heuristic and a
    local search phase.
    """

    verbose: bool = True
    rng: random.Random = random.Random(0)

    def __init__(self, obj_function: BaseEvaluator[E], alpha: float, iterations: int):
        """
        Initializes the AbstractGRASP solver.

        Args:
            obj_function (BaseEvaluator[E]): The objective function to be minimized.
            alpha (float): The GRASP greediness-randomness parameter (0 to 1).
            iterations (int): The number of GRASP iterations to perform.
        """
        self.obj_function: BaseEvaluator[E] = obj_function
        self.alpha: float = alpha
        self.iterations: int = iterations
        self.best_sol: Optional[BaseSolution[E]] = None
        self.sol: Optional[BaseSolution[E]] = None
        self.cl: List[E] = []
        self.rcl: List[E] = []

    @abstractmethod
    def make_cl(self) -> List[E]:
        """Creates the initial Candidate List (CL)."""
        pass

    @abstractmethod
    def make_rcl(self) -> List[E]:
        """Creates an empty Restricted Candidate List (RCL)."""
        pass

    @abstractmethod
    def update_cl(self) -> None:
        """Updates the CL based on the current solution."""
        pass

    @abstractmethod
    def create_empty_sol(self) -> BaseSolution[E]:
        """Creates a new, empty solution."""
        pass

    @abstractmethod
    def local_search(self) -> BaseSolution[E]:
        """
        Performs the local search phase to improve the current solution
        until a local minimum is found.
        """
        pass

    def solve(self) -> Optional[BaseSolution[E]]:
        """
        The main GRASP loop.

        It iterates for a fixed number of times, each time generating a new
        solution with the constructive heuristic and then improving it with
        local search.

        Returns:
            The best solution found across all iterations.
        """
        self.best_sol = self.create_empty_sol()

        for i in range(self.iterations):
            self.sol = self.constructive_heuristic()
            self.sol = self.local_search()
            if self.best_sol.cost > self.sol.cost:
                self.best_sol = Solution(self.sol)  # Create a copy
                if self.verbose:
                    print(f"(Iter. {i}) BestSol = {self.best_sol}")
        
        return self.best_sol

    def constructive_heuristic(self) -> BaseSolution[E]:
        """
        Builds a feasible solution by iteratively selecting candidates from an RCL.
        """
        self.cl = self.make_cl()
        self.rcl = self.make_rcl()
        self.sol = self.create_empty_sol()

        # This loop continues as long as there are candidates to evaluate.
        # Note: The original Java code's stopping criterion was ambiguous and
        # likely flawed. A more standard approach is to continue until the CL
        # is empty, which is implemented here.
        while self.cl:
            min_cost = float('inf')
            max_cost = -float('inf')

            # Calculate costs for all candidates and find the min/max range
            costs = {c: self.obj_function.evaluate_insertion_cost(c, self.sol) for c in self.cl}
            if not costs: break # No more valid candidates
            
            min_cost = min(costs.values())
            max_cost = max(costs.values())
            
            # Build the RCL based on the alpha threshold
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            self.rcl.clear()
            for c, cost in costs.items():
                if cost <= threshold:
                    self.rcl.append(c)
            
            # If RCL is empty, something went wrong or no options are good.
            if not self.rcl: break

            # Choose a candidate randomly from the RCL and add it to the solution
            chosen_candidate = self.rng.choice(self.rcl)
            self.sol.append(chosen_candidate)
            self.cl.remove(chosen_candidate)
            
            # Update the candidate list for the next iteration
            self.update_cl()

        self.obj_function.evaluate(self.sol)
        return self.sol