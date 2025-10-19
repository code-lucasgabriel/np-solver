from __future__ import annotations
from typing import TypeVar, Generic, Optional

E = TypeVar('E')


class BaseSolution(list, Generic[E]):
    """
    A class representing a solution to an optimization problem.

    It extends Python's built-in `list` to hold the elements of the solution and adds a `cost` attribute to store its objective function value.
    """

    def __init__(self, other: Optional[BaseSolution[E]] = None):
        """
        Initializes a Solution object.

        - If `other` is not provided, it creates an empty solution with an
          infinite cost (default constructor).
        - If `other` is an existing Solution object, it creates a copy of that
          solution with the same elements and cost (copy constructor).

        Args:
            other (Optional[BaseSolution[E]]): Another solution to copy from.
        """
        if other is not None:
            # Copy constructor behavior: copy elements and cost
            super().__init__(other)
            self.cost: float = other.cost
        else:
            # Default constructor behavior: empty list, infinite cost
            super().__init__()
            self.cost: float = float('inf')

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of the Solution.
        """
        elements_str = super().__str__()
        return (f"Solution: cost=[{self.cost}], size=[{len(self)}], "
                f"elements={elements_str}")