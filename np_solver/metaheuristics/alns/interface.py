from abc import ABC, abstractmethod
from typing import List, Tuple
from np_solver.core import BaseSolution, BaseProblemInstance, BaseEvaluator

# --- 1. Operator Strategies ---

class ALNSDestroy(ABC):
    """
    Abstract interface for an ALNS "Destroy" operator.
    This is a "Strategy" object.
    
    Its responsibility is to take a complete solution and return a
    "partial" or "broken" solution (e.g., by removing customers).
    """
    @abstractmethod
    def destroy(self, solution: BaseSolution, problem: BaseProblemInstance) -> BaseSolution:
        """
        Destroys a part of the solution.

        ! Important: This method MUST return a new copy (e.g., `new_sol = solution.copy()`).
        Modifying the solution in-place will break the algorithm.
        
        Args:
            solution (BaseSolution): The current complete solution.
            problem (BaseProblemInstance): The problem instance for context.

        Returns:
            BaseSolution: A new, partial solution object.
        """
        pass

class ALNSRepair(ABC):
    """
    Abstract interface for an ALNS "Repair" operator.
    This is a "Strategy" object.
    
    Its responsibility is to take a partial solution (from a destroy
    operator) and rebuild it into a new, complete solution.
    """
    @abstractmethod
    def repair(self, 
             partial_solution: BaseSolution, 
             problem: BaseProblemInstance, 
             evaluator: BaseEvaluator) -> BaseSolution:
        """
        Repairs a partial solution to make it whole.

        ! Important: This method MUST return a new, complete solution object.
        The returned solution does *not* need to have its `.cost`
        attribute set, as the main ALNS loop will evaluate it.
        
        Args:
            partial_solution (BaseSolution): The broken solution from a destroy op.
            problem (BaseProblemInstance): The problem instance.
            evaluator (BaseEvaluator): The evaluator, for cost-based decisions.

        Returns:
            BaseSolution: A new, complete solution.
        """
        pass

# --- 2. Framework Strategies ---

class ALNSAcceptance(ABC):
    """
    Abstract interface for the ALNS "Acceptance Criterion".
    This is a "Strategy" object.
    
    It decides whether to accept a new candidate solution (s') as the
    next current solution (s_i+1), or to reject it and keep the
    current one (s_i).
    """
    
    def _initialize_run(self):
        """(Optional) Called by ALNS._initialize_run() to reset any state (e.g., temperature)."""
        pass
        
    @abstractmethod
    def accept(self, 
             candidate_cost: float, 
             current_cost: float, 
             best_cost: float, 
             sense: BaseEvaluator.ObjectiveSense) -> bool:
        """
        Decides whether to accept the candidate solution.
        
        Args:
            candidate_cost (float): The cost of the new solution (s').
            current_cost (float): The cost of the current solution (s_i).
            best_cost (float): The cost of the best-so-far solution (s*).
            sense (ObjectiveSense): MINIMIZE or MAXIMIZE.

        Returns:
            bool: True to accept s' as the next current solution, False to reject.
        """
        pass
        
    def step(self):
        """(Optional) Called by ALNS._iterate() to update internal state (e.g., cool down)."""
        pass

class ALNSWeightManager(ABC):
    """
    Abstract interface for the ALNS "Weight and Selection Manager".
    This is a "Strategy" object.
    
    It is responsible for:
    1. Tracking all destroy/repair operators.
    2. Tracking their weights and scores.
    3. Selecting which operators to use in an iteration.
    4. Updating scores based on iteration outcomes.
    5. Updating weights at the end of a "segment".
    """
    
    @abstractmethod
    def _initialize_run(self, 
                        destroy_ops: List[ALNSDestroy], 
                        repair_ops: List[ALNSRepair]):
        """
        Called by ALNS._initialize_run() to receive the operators
        and reset all weights and scores.
        """
        pass
        
    @abstractmethod
    def select_operators(self) -> Tuple[ALNSDestroy, ALNSRepair]:
        """
        Selects one destroy and one repair operator based on current weights.
        
        Returns:
            Tuple[ALNSDestroy, ALNSRepair]: The chosen operators.
        """
        pass
        
    @abstractmethod
    def update_scores(self, 
                      destroy_op: ALNSDestroy, 
                      repair_op: ALNSRepair, 
                      candidate_cost: float, 
                      current_cost: float, 
                      best_cost: float,
                      sense: BaseEvaluator.ObjectiveSense,
                      is_accepted: bool,
                      is_new_best: bool):
        """
        Updates the *scores* for the given operators based on the
        outcome of the iteration.
        """
        pass
        
    @abstractmethod
    def step(self):
        """
        Called once per iteration. The manager should use this to
        track segment length and trigger `update_weights()` when needed.
        """
        pass