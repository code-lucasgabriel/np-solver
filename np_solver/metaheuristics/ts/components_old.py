# ts_components.py
from abc import ABC, abstractmethod
from collections import deque
import random
from np_solver.core import BaseProblemInstance, BaseEvaluator, BaseSolution

# --- Base "Move" Class ---
class BaseMove(ABC):
    """
    Abstract class for a move. A move should be hashable to be stored
    in the tabu list. User can subclass this (e.g., SwapMove(i, j)).
    """
    def __hash__(self):
        raise NotImplementedError("Moves must be hashable")
    
    def __eq__(self, other):
        raise NotImplementedError("Moves must be comparable")

# --- Strategy 1: Initialization ---

class BaseInitialization(ABC):
    @abstractmethod
    def execute(self, problem: BaseProblemInstance, evaluator: BaseEvaluator) -> BaseSolution:
        pass

class RandomInitialization(BaseInitialization):
    """A simple random initialization strategy."""
    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def execute(self, problem: BaseProblemInstance, evaluator: BaseEvaluator) -> BaseSolution:
        # Assumes get_problem_space() returns something that can create a solution
        # This is highly problem-dependent.
        print("Using Random Initialization")
        sol_data = problem.get_problem_space().sample() # Simplified
        solution = BaseSolution(data=sol_data)
        solution.cost = evaluator.objective_function(solution)
        return solution

# --- Strategy 2: Neighborhood ---

class BaseNeighborhood(ABC):
    @abstractmethod
    def generate(self, solution: BaseSolution) -> list[BaseMove]:
        """Returns a list of all possible moves from the current solution."""
        pass

# --- Strategy 3: Tabu List Management ---

class BaseTabuList(ABC):
    @abstractmethod
    def add(self, move: BaseMove):
        """Adds a move to the tabu list."""
        pass

    @abstractmethod
    def is_tabu(self, move: BaseMove) -> bool:
        """Checks if a move is currently tabu."""
        pass

class DequeTabuList(BaseTabuList):
    """A standard tabu list using a deque with fixed tenure."""
    def __init__(self, tenure: int):
        self.tenure = tenure
        self.tabu_list = deque(maxlen=tenure)
    
    def add(self, move: BaseMove):
        self.tabu_list.append(move)
    
    def is_tabu(self, move: BaseMove) -> bool:
        return move in self.tabu_list

# --- Strategy 4: Aspiration Criteria ---

class BaseAspirationCriteria(ABC):
    @abstractmethod
    def is_satisfied(self, move_cost: float, best_cost: float) -> bool:
        """
        Checks if a tabu move is allowed.
        Takes the *cost* of the solution after the move,
        and the *cost* of the all-time best solution.
        """
        pass

class AspirationByImprovement(BaseAspirationCriteria):
    """
    Standard criterion: allows a tabu move if it leads to a
    solution better than the current best-so-far.
    (Assumes minimization)
    """
    def is_satisfied(self, move_cost: float, best_cost: float) -> bool:
        return move_cost < best_cost

# --- Strategy 5: Move Selection ---

class BaseMoveSelection(ABC):
    @abstractmethod
    def select(self, 
                 solution: BaseSolution, 
                 moves: list[BaseMove], 
                 tabu_list: BaseTabuList, 
                 aspiration: BaseAspirationCriteria, 
                 evaluator: BaseEvaluator,
                 best_solution_cost: float) -> BaseMove | None:
        """Selects the best admissible move from the list."""
        pass

class BestAdmissibleMoveSelection(BaseMoveSelection):
    """
    Selects the best non-tabu move.
    If all moves are tabu, selects the best tabu move that
    meets the aspiration criteria.
    """
    def select(self, 
                 solution: BaseSolution, 
                 moves: list[BaseMove], 
                 tabu_list: BaseTabuList, 
                 aspiration: BaseAspirationCriteria, 
                 evaluator: BaseEvaluator,
                 best_solution_cost: float) -> BaseMove | None:

        best_move = None
        best_cost = float('inf')
        
        best_tabu_move = None
        best_tabu_cost = float('inf')

        for move in moves:
            # Use delta-evaluation for speed
            move_cost = evaluator.evaluate_move(solution, move) 
            
            if not tabu_list.is_tabu(move):
                if move_cost < best_cost:
                    best_cost = move_cost
                    best_move = move
            else: # Move is tabu
                if move_cost < best_tabu_cost:
                    best_tabu_cost = move_cost
                    best_tabu_move = move
        
        # 1. Best case: We found a non-tabu move.
        if best_move is not None:
            return best_move
            
        # 2. No non-tabu moves. Check aspiration for the best tabu move.
        if best_tabu_move is not None:
            if aspiration.is_satisfied(best_tabu_cost, best_solution_cost):
                return best_tabu_move
        
        # 3. No admissible move found
        return None