from interface.Solution import Solution
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

E = TypeVar('E')


class Evaluator(Generic[E], ABC):
    """
    Abstract base class for an objective function in an optimization problem.

    This class defines the "interface" that a problem needs to follow to be
    solved by a metaheuristic. It provides methods to get the problem size and
    evaluate the cost of a full solution or the change in cost from making
    small modifications (like adding, removing, or swapping elements).
    """

    @abstractmethod
    def get_domain_size(self) -> int:
        """
        Returns the size of the problem domain.

        This typically corresponds to the number of potential elements or
        decision variables in the optimization problem.

        Returns:
            int: The size of the problem domain.
        """
        pass

    @abstractmethod
    def evaluate(self, sol: Solution[E]) -> float:
        """
        Calculates the objective function value for a given solution.

        Args:
            sol (Solution[E]): The solution to be evaluated.

        Returns:
            float: The objective function value (cost) of the solution.
        """
        pass

    @abstractmethod
    def evaluate_insertion_cost(self, elem: E, sol: Solution[E]) -> float:
        """
        Evaluates the change in cost from inserting an element into a solution.

        This is useful for quickly assessing potential moves in local search.

        Args:
            elem (E): The element being considered for insertion.
            sol (Solution[E]): The solution into which the element might be inserted.

        Returns:
            float: The change in cost that would result from the insertion.
        """
        pass

    @abstractmethod
    def evaluate_removal_cost(self, elem: E, sol: Solution[E]) -> float:
        """
        Evaluates the change in cost from removing an element from a solution.

        This is useful for quickly assessing potential moves in local search.

        Args:
            elem (E): The element being considered for removal.
            sol (Solution[E]): The solution from which the element might be removed.

        Returns:
            float: The change in cost that would result from the removal.
        """
        pass

    @abstractmethod
    def evaluate_exchange_cost(
        self, elem_in: E, elem_out: E, sol: Solution[E]
    ) -> float:
        """
        Evaluates the cost variation of exchanging one element in the solution
        with one element that is currently not in the solution.

        Args:
            elem_in (E): The element to be inserted into the solution.
            elem_out (E): The element to be removed from the solution.
            sol (Solution[E]): The solution being modified.

        Returns:
            float: The change in cost that would result from the exchange.
        """
        pass