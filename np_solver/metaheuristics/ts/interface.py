from abc import abstractmethod, ABC
from typing import List, Any
from np_solver.core import BaseSolution, BaseEvaluator

class TSNeighborhood(ABC):
    """
    Abstract interface for defining a neighborhood structure.
    This is a "Strategy" object.

    A "move" is represented as a hashable object (e.g., a tuple)
    that defines the change from one solution to another.
    Example: ('swap', client1, client2)
    """
    @abstractmethod
    def generate_moves(self, solution: BaseSolution) -> List[Any]:
        """
        Generates all possible moves from the current solution.
        
        Returns:
            A list of hashable "move" objects.
        """
        pass

    @abstractmethod
    def apply_move(self, solution: BaseSolution, move: Any) -> BaseSolution:
        """
        Applies a move to a solution and returns a *new* solution object.

        ! Important: This method MUST return a new copy (e.g., `new_sol = solution.copy()`).
        Modifying the solution in-place will break the algorithm, as
        `BaseMetaheuristic` does not copy the solution when updating `best_solution`.
        """
        pass

    @abstractmethod
    def evaluate_move(self, solution: BaseSolution, move: Any, evaluator: BaseEvaluator) -> float:
        """
        Efficiently calculates the *cost of the new solution* after the move.

        This method SHOULD use the fast delta-evaluation methods
        from the BaseEvaluator (e.g., `evaluator.evaluate_swap_cost(...)`).
        
        It must return the *full cost* of the new solution,
        e.g.: `return solution.cost + delta_cost`
        """
        pass


class TSTabuList(ABC):
    """
    Abstract interface for a tabu list (the short-term memory).
    This is a "Strategy" object.
    """
    
    @abstractmethod
    def add(self, move: Any):
        """Adds a move to the tabu list."""
        pass

    @abstractmethod
    def is_tabu(self, move: Any) -> bool:
        """Checks if a move is currently tabu."""
        pass

    @abstractmethod
    def clear(self):
        """Clears all moves from the tabu list."""
        pass


class TSAspiration(ABC):
    """
    Abstract interface for the aspiration criteria.
    This is a "Strategy" object.
    """
    
    @abstractmethod
    def is_aspirated(self, neighbor_cost: float, best_cost: float, 
                     sense: BaseEvaluator.ObjectiveSense) -> bool:
        """
        Checks if a tabu move should be allowed (aspirated).
        
        Args:
            neighbor_cost: The cost of the new neighbor solution.
            best_cost: The cost of the best-so-far solution.
            sense: The optimization objective (MINIMIZE or MAXIMIZE).
        
        Returns:
            True if the move is good enough to override its tabu status.
        """
        pass