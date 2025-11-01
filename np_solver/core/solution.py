from __future__ import annotations
from typing import TypeVar, Generic, Optional, List

E = TypeVar('E')


class BaseSolution(list, Generic[E]):
    """
    A class representing a solution to an optimization problem.

    It extends Python's built-in `list` to hold the elements of the solution 
    and adds a `cost` attribute to store its objective function value.
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
    
    def _find_node(self, node_to_find):
        """Finds (route_idx, node_idx) of a node. Returns (None, None) if not found."""
        try:
            n_idx = self.index(node_to_find)
            return n_idx
        except ValueError:
            print("Element not found in the solution.")
            return None

    @classmethod
    def from_list(cls, elements: List[E], evaluator) -> BaseSolution[E]:
        """
        Creates a new Solution object from a given list of elements and a cost.

        Args:
            elements: list-like object
            evaluator: BaseEvaluator object

        """
        # Create a new, empty solution instance
        new_solution = cls() 
        
        # Populate it with the elements
        new_solution.extend(elements)
        
        # Set the cost
        new_solution.cost = evaluator.evaluate(new_solution)
        
        return new_solution

    def copy(self) -> BaseSolution[E]:
        """
        Creates a deep copy of this solution.

        This method leverages the copy constructor logic already
        defined in __init__.

        Returns:
            BaseSolution[E]: A new Solution object with the same 
                             elements and cost.
        """
        return BaseSolution(self)

    def add(self, elem: E):
        """
        Adds an element to the solution's list (in-place).

        This is a convenience method that calls `append()`.
        
        It does NOT update the 'cost' attribute.

        Args:
            elem (E): The element to add.
        """
        self.append(elem)
    
    def remove(self, elem: E):
        """
        Removes the first occurrence of an element from the solution's 
        list (in-place).

        This is a convenience method that calls `list.remove()`.

        It does NOT update the 'cost' attribute.

        Args:
            elem (E): The element to remove.
        
        Raises:
            ValueError: If the element is not found in the solution.
        """
        self.remove(elem)

    def insert_element(self, element_to_insert, element_new_neightbor):
        """
        Inserts a new element to be immediately after elem_new_neighbor.

        It does NOT update the 'cost' attribute.

        Args:
            element_to_insert: element to insert in the solution
            element_new_neighbor: element in the solution which will be the neighbor of the element_to_insert
        """
        idx_new_neighbor = self.index(element_new_neightbor)
        self.insert(idx_new_neighbor+1, element_to_insert)
    

    def exchange(self, elem_to_add, elem_to_remove):
        """
        Swaps an element inside the solution (elem_to_remove) with an element outside of it (elem_to_add).
        
        This is just a special case of swap where one element (elem_to_remove)
        is not yet in the solution.

        Args:
            elem_to_remove: Element in the solution
            elem_to_add: Element out of the solution

        Raises:
            ValueError: If either element_to_remove is not found in the solution.
        """

        idx_remove = self.index(elem_to_remove)
        
        if idx_remove is None:
             raise ValueError("Element to remove (elem_to_add) not found in solution")
        
        # 2. Perform the swap
        self[idx_remove] = elem_to_add

    def swap(self, elem1: E, elem2: E):
        """
        Swaps the positions of two elements *already in* the solution (in-place).

        It does NOT update the 'cost' attribute.

        Args:
            elem1 (E): The first element to swap.
            elem2 (E): The second element to swap.

        Raises:
            ValueError: If either element is not found in the solution.
        """
        idx1 = self.index(elem1)
        idx2 = self.index(elem2)

        _aux = self[idx1]
        self[idx1] = self[idx2]
        self[idx2] = _aux

    def relocate(self, elem_to_move: E, elem_new_neighbor: E):
        """
        Moves `elem_to_move` to the position immediately *after* `elem_new_neighbor` (in-place).

        It does NOT update the 'cost' attribute.

        Args:
            elem_to_move (E): The element to relocate.
            elem_new_neighbor (E): The element to insert after.

        Raises:
            ValueError: If either element is not found in the solution.
        """
        if elem_to_move == elem_new_neighbor:
            return

        self.remove(elem_to_move)

        idx_neighbor = self.index(elem_new_neighbor)

        self.insert(idx_neighbor + 1, elem_to_move)

    