from collections import deque
from np_solver.metaheuristics.ts.interface import TSTabuList, TSAspiration
from np_solver.core import BaseEvaluator
from typing import Any

class DequeTabuList(TSTabuList):
    """
    A concrete TSTabuList that uses a simple deque (FIFO queue)
    to maintain tabu tenure. It stores the last `tenure` moves.
    """
    def __init__(self, tenure: int):
        """
        Args:
            tenure (int): The number of iterations a move stays tabu.
        """
        if tenure < 0:
            raise ValueError("Tabu tenure cannot be negative.")
        self.tenure = tenure
        # maxlen=0 creates a deque that always discards
        self.deque = deque(maxlen=tenure if tenure > 0 else 0)

    def add(self, move: Any):
        """Adds the move to the deque. If full, the oldest move is dropped."""
        if self.tenure > 0:
            self.deque.append(move)

    def is_tabu(self, move: Any) -> bool:
        """Checks if the move is currently in the deque."""
        if self.tenure == 0:
            return False
        # This check requires the 'move' object to be hashable
        # and implement __eq__ (tuples are perfect for this).
        return move in self.deque

    def clear(self):
        """Empties the deque."""
        self.deque.clear()
        
    def __str__(self) -> str:
        return f"DequeTabuList(tenure={self.tenure}, size={len(self.deque)})"


class AspirationByBest(TSAspiration):
    """
    A concrete TSAspiration criterion.
    
    It allows (aspirates) a tabu move if and only if that move results
    in a new best-so-far solution.
    """
    def is_aspirated(self, neighbor_cost: float, best_cost: float, 
                     sense: BaseEvaluator.ObjectiveSense) -> bool:
        
        if sense == BaseEvaluator.ObjectiveSense.MINIMIZE:
            # Allow if this neighbor is strictly better than the all-time best
            return neighbor_cost < best_cost
        else:
            # MAXIMIZE
            return neighbor_cost > best_cost