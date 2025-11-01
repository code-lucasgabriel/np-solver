from np_solver.core import BaseSolution, BaseProblemInstance
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod

E = TypeVar('E')



class BaseEvaluator(Generic[E], ABC):
    """
    Abstract base class for an objective function in an optimization problem.

    This class defines the "interface" that a problem needs to follow to be
    solved by a metaheuristic. It provides methods to get the problem size and
    evaluate the cost of a full solution or the change in cost from making
    small modifications (like adding, removing, or swapping elements).

    Attributes:
        sense (ObjectiveSense): The optimization objective.
            Must be set by all concrete subclasses to either
            `ObjectiveSense.MINIMIZE` or `ObjectiveSense.MAXIMIZE`.
    """
    class ObjectiveSense(Enum):
        """Defines the optimization direction."""
        MINIMIZE = auto()
        MAXIMIZE = auto()

    sense: ObjectiveSense = None

    def __init__(self, problem: BaseProblemInstance):
        """
        Initializes the evaluator.

        Raises:
            NotImplementedError: If the subclass does not set the `sense` class attribute.
        """
        
        if self.sense is None:
            raise NotImplementedError(
                f"Class {self.__class__.__name__} must set the 'sense' "
                f"class attribute to 'ObjectiveSense.MINIMIZE' or "
                f"'ObjectiveSense.MAXIMIZE'."
            )

        self.problem = problem

    def _get_infeasible_cost(self):
        """
        Helper method.
        
        Returns the appropriate 'infinite' cost penalty based on the
        optimization sense.
        """
        if self.sense == self.ObjectiveSense.MINIMIZE:
            return float("inf")
        else:
            return float("-inf")

    @abstractmethod
    def constraints(self, sol: BaseSolution[E]) -> bool:
        """
        Pass a given solution canditate to the set of constraints of the problem.
        If the solution violates some of the constraints, it returns False, othrewise, True.

        Recommended implementation:
        def consteains(sol):
            return all_clients_served(sol) and hamiltonian_cycle(sol) and 

        Args:
            sol (BaseSolution): The solution to be evaluated.

        Returns:
            bool: Feasibility of the solution, given the constraints.
        """
        pass
    
    @abstractmethod
    def objective_function(self, sol: BaseSolution) -> float:
        """
        Implementation of the objective function to calculate the cost of the solution.

        This method assumes `constraints(sol)` has already returned True.
        Do not call this directly; use `evaluate()`.

        Args:
            sol (BaseSolution[E]): The solution to be evaluated.

        Returns:
            float: The objective function value (cost) of the solution
        """
        pass

    def evaluate(self, sol: BaseSolution[E]) -> float:
        """
        Calculates the objective function value for a given solution.

        Args:
            sol (BaseSolution[E]): The solution to be evaluated.

        Returns:
            float: The objective function value (cost) of the solution.

        ! Note: if the solution if not possible due to problem constraints, the returned objective function value (cost) should be 'float(inf)' (or '-float(inf)', if dealing with maximization objective function)!
        """
        if self.constraints(sol):
            # solutin respects all constraints (is possible)
            return self.objective_function(sol)

        # solution is not possible
        return self._get_infeasible_cost()

# -----------------------------------------------------------------
#! Delta-evaluation: Default "fallback" implementations
#! These should be overridden for performance
# -----------------------------------------------------------------

    def evaluate_insertion_cost(self, elem_to_insert: E, elem_new_neighbor: Optional[E], sol: BaseSolution[E]) -> float:
        """
        (Default Fallback) Evaluates the cost delta for inserting an element.

        This is a SLOW, generic implementation that copies the solution and
        calls self.evaluate().

        For performance, subclasses SHOULD override this method with a
        fast, O(1) delta calculation if possible.
        """
        cost_old = self.evaluate(sol)

        sol_new = sol.copy()

        if elem_new_neighbor is not None:
            idx_neighbor = sol._find_node(elem_new_neighbor)
        else:
            idx_neighbor = len(sol)

        sol_new.insert(idx_neighbor, elem_to_insert)
        
        cost_new = self.evaluate(sol_new)
        
        infeasible_cost = self._get_infeasible_cost()
        if cost_new == infeasible_cost and cost_old == infeasible_cost:
            return 0.0
            
        return cost_new - cost_old

    def evaluate_removal_cost(self, elem: E, sol: BaseSolution[E]) -> float:
        """
        (Default Fallback) Evaluates the cost delta for removing an element.

        This is a SLOW, generic implementation.

        For performance, subclasses SHOULD override this.
        """
        cost_old = self.evaluate(sol)

        sol_new = sol.copy()
        sol_new.remove(elem)
    
        cost_new = self.evaluate(sol_new)

        infeasible_cost = self._get_infeasible_cost()
        if cost_new == infeasible_cost and cost_old == infeasible_cost:
            return 0.0
            
        return cost_new - cost_old

    def evaluate_exchange_cost(
        self, elem_in: E, elem_out: E, sol: BaseSolution[E]
    ) -> float:
        """
        (Default Fallback) Evaluates the cost delta for an exchange (swap).

        This is a SLOW, generic implementation.

        For performance, subclasses SHOULD override this.
        """
        cost_old = self.evaluate(sol)
    
        sol_new = sol.copy()
        sol_new.remove(elem_out)
        sol_new.add(elem_in)

        cost_new = self.evaluate(sol_new)

        infeasible_cost = self._get_infeasible_cost()
        if cost_new == infeasible_cost and cost_old == infeasible_cost:
            return 0.0
            
        return cost_new - cost_old

    def evaluate_swap_cost(self, elem1: E, elem2: E, sol: BaseSolution[E]) -> float:
        """
        (Default Fallback) Evaluates the cost delta for swapping two elements
        already within the solution.

        This is a SLOW, generic implementation.

        For performance, subclasses SHOULD override this.
        """
        cost_old = self.evaluate(sol)

        sol_new = sol.copy()
        sol_new.swap(elem1, elem2)

        cost_new = self.evaluate(sol_new)

        infeasible_cost = self._get_infeasible_cost()
        if cost_new == infeasible_cost and cost_old == infeasible_cost:
            return 0.0
            
        return cost_new - cost_old

    def evaluate_relocation_cost(
        self, elem_to_move: E, elem_new_neighbor: E, sol: BaseSolution[E]
    ) -> float:
        """
        (Default Fallback) Evaluates the cost delta for relocating an element
        to a new position within the solution (e.g., move elem_to_move
        to be after elem_new_neighbor).

        This is a SLOW, generic implementation.

        For performance, subclasses SHOULD override this.
        """
        cost_old = self.evaluate(sol)
    
        sol_new = sol.copy()
        sol_new.relocate(elem_to_move, elem_new_neighbor) 

        cost_new = self.evaluate(sol_new)

        infeasible_cost = self._get_infeasible_cost()
        if cost_new == infeasible_cost and cost_old == infeasible_cost:
            return 0.0
            
        return cost_new - cost_old