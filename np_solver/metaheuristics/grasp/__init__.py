from np_solver.core.evaluator import BaseEvaluator
from np_solver.core.solution import BaseSolution
import random
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

# Generic type for the elements that compose a solution
E = TypeVar('E')

class BaseGRASP(Generic[E], ABC):
    """
    Abstract base class for the GRASP metaheuristic, designed for a 
    mixin-based framework.

    This class provides the main GRASP loop (`solve`) and the template for 
    the constructive heuristic. Specific behaviors, like how the RCL is
    built or how alpha is managed, are intended to be "mixed in".
    """

    verbose: bool = True
    rng: random.Random = random.Random(0)

    def __init__(
        self, 
        obj_function: BaseEvaluator[E], 
        alpha: float,
        iterations: int,
        sense: str = "min",
        **kwargs 
    ):
        """
        Initializes the AbstractGRASP solver.

        Args:
            obj_function (BaseEvaluator[E]): The objective function to be minimized.
            alpha (float): The *initial* GRASP greediness parameter (0 to 1).
            iterations (int): The number of GRASP iterations to perform.
        """
        super().__init__(**kwargs) # Call next in MRO
        self.obj_function: BaseEvaluator[E] = obj_function
        self.alpha: float = alpha
        self.iterations: int = iterations
        self.best_sol: Optional[BaseSolution[E]] = None
        self.sol: Optional[BaseSolution[E]] = None
        self.cl: List[E] = []
        self.rcl: List[E] = []

        self.sense: str = sense
        if self.sense == "min":
            self.infeasible_cost_val = float('inf')
            # Lambda: True if cost 'a' is better than cost 'b'
            self.best_is_better = lambda a, b: a < b
        else: # "max"
            self.infeasible_cost_val = float('-inf')
            # Lambda: True if cost 'a' is better than cost 'b'
            self.best_is_better = lambda a, b: a > b

    """
    <- ABSTRACT METHODS (Problem-Specific - MUST be implemented in subclass) ->
    """

    @abstractmethod
    def make_cl(self) -> List[E]:
        """Creates the initial Candidate List (CL)."""
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

    """
    <- ABSTRACT METHODS (Framework-Specific - MUST be implemented by a mixin) ->
    """
    
    @abstractmethod
    def _build_rcl(self) -> List[E]:
        """
        Builds the Restricted Candidate List (RCL).
        This method is intended to be implemented by a construction mixin
        (e.g., RandomPlusGreedy, SampledGreedy).
        """
        pass

    """
    <- Core Logic & Template Methods ->
    """

    def solve(self) -> Optional[BaseSolution[E]]:
        """
        The main GRASP loop.
        """
        self.best_sol = None
        self._initialize_reactive() 

        for i in range(self.iterations):
            self._update_alpha_before_construction(i) 
            
            self.sol = self.constructive_heuristic()
            
            if self.sol.cost == self.infeasible_cost_val:
                if self.verbose:
                    print(f"(Iter. {i}) Construction failed, skipping.")
                continue 
                
            self.sol = self.local_search()
            self._update_alpha_after_ls(self.sol) 

            if self.best_sol is None or self.best_is_better(self.sol.cost, self.best_sol.cost):
                self.best_sol = BaseSolution(self.sol) 
                if self.verbose:
                    print(f"(Iter. {i}) BestSol = {self.best_sol}")
        
        return self.best_sol

    def constructive_heuristic(self) -> BaseSolution[E]:
        """
        (Template Method) Builds a feasible solution.
        
        This method provides the loop structure, relying on abstract
        methods (`make_cl`, `_build_rcl`, etc.) to perform the steps.
        """
        self.cl = self.make_cl()
        self.sol = self.create_empty_sol()

        while self.cl:
            # 1. Build RCL (delegated to mixin)
            self.rcl = self._build_rcl()
            
            if not self.rcl:
                break # No more valid candidates

            # 2. Select from RCL (default implementation)
            chosen_candidate = self._select_from_rcl(self.rcl) 
            
            # 3. Add to solution and update problem state
            self.sol.append(chosen_candidate)
            self.cl.remove(chosen_candidate)
            self.update_cl() # Problem-specific update

        self.obj_function.evaluate(self.sol)
        return self.sol

    """
    <- Methods with default implementation (Hooks for Mixins) ->
    """

    def _select_from_rcl(self, rcl: List[E]) -> E:
        """
        Selects a candidate from the RCL. 
        Default is uniform random selection.
        """
        return self.rng.choice(rcl)

    def _initialize_reactive(self) -> None:
        """Hook for initializing reactive components."""
        pass

    def _update_alpha_before_construction(self, iteration: int) -> None:
        """Hook for updating alpha before constructing a solution."""
        pass 

    def _update_alpha_after_ls(self, solution: BaseSolution[E]) -> None:
        """Hook for updating alpha policies based on the found solution."""
        pass