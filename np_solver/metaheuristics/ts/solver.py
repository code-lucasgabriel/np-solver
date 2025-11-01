from typing import Optional
from np_solver.core import BaseSolution
from np_solver.metaheuristics import BaseMetaheuristic
from np_solver.metaheuristics.ts.interface import TSNeighborhood, TSTabuList, TSAspiration
from np_solver.metaheuristics.ts.components import DequeTabuList, AspirationByBest

class TabuSearch(BaseMetaheuristic):
    """
    A Tabu Search (TS) metaheuristic implementation using the Template Method
    ("BaseMetaheuristic") and the Strategy Design Pattern.

    This class implements the "_initialize" and "_iterate" methods
    from "BaseMetaheuristic". It coordinates the search by delegating
    the actual logic to the provided strategy objects.
    """
    def __init__(self, 
                 tenure: int,
                 neighborhood_strategy: TSNeighborhood,
                 tabu_list: TSTabuList=None,
                 aspiration_criteria: TSAspiration=None,
                 **kwargs):
        """
        Initializes the Tabu Search algorithm.

        Args:
            evaluator: The user's solution evaluator.
            initial_solution_strategy: Strategy object for creating s0.
            neighborhood_strategy: Strategy object for moves, eval, and application.
            tabu_list: Strategy object for managing tabu status.
            aspiration_criteria: Strategy object for overriding tabu status.
            **kwargs: Passed up to BaseMetaheuristic (e.g., 'time_limit').
        """
        super().__init__(**kwargs)

        self.neighborhood = neighborhood_strategy
        self.tabu_list = tabu_list if tabu_list is not None else DequeTabuList(tenure)
        self.aspiration = aspiration_criteria if aspiration_criteria is not None else AspirationByBest()

        self.current_solution = None

    def _initialize_run(self):
        """
        (Template Method) Creates the initial solution s0 using the
        provided strategy and sets it as the current and best solution.
        """
        self.tabu_list.clear()


    def _iterate(self) -> Optional[BaseSolution]:
        """
        (Template Method) Performs one complete iteration of Tabu Search.
        
        1. Generates all moves from the current solution.
        2. Evaluates each move, checking its tabu status.
        3. Checks aspiration criteria for tabu moves.
        4. Selects the best *allowed* move (best non-tabu or best aspirated-tabu).
        5. Applies the move, updates the tabu list, and returns the new solution.
        
        Returns:
            The new "BaseSolution" for the next iteration, or "None" if no
            allowed move was found.
        """
        
        best_allowed_move = None
        best_allowed_cost = self._infeasible_cost
        
        # 1. Generate all moves
        print(f"Sover current solution: {self.current_solution}")
        all_moves = self.neighborhood.generate_moves(self.current_solution, self.problem)

        if not all_moves:
            print("No neighbors.")
            return None # Stuck, no neighbors

        # 2. Evaluate all moves and find the best *allowed* one
        for move in all_moves:
            
            # 2a. Evaluate the move
            neighbor_cost = self.neighborhood.evaluate_move(
                self.current_solution, move, self.evaluator
            )

            # 2b. Check tabu status
            is_tabu = self.tabu_list.is_tabu(move)
            
            # 2c. Check aspiration criteria
            is_aspirated = self.aspiration.is_aspirated(
                neighbor_cost, self.best_solution.cost, self.evaluator.sense
            )

            # 2d. Determine if the move is allowed
            is_allowed = (not is_tabu) or is_aspirated

            if is_allowed:
                # 3. Check if this allowed move is the best one found so far
                if (self.is_minimizing and neighbor_cost < best_allowed_cost) or \
                   (not self.is_minimizing and neighbor_cost > best_allowed_cost):
                    
                    best_allowed_cost = neighbor_cost
                    best_allowed_move = move
        
        # 4. If no allowed move was found (e.g., all moves are tabu and
        #    none are aspirated), we are stuck.
        if best_allowed_move is None:
            print("Tabu search stuck: No allowed moves found.")
            return None 
        
        # 5. We have a winner. Apply the move.
        
        # Add the chosen move to the tabu list
        self.tabu_list.add(best_allowed_move)
        
        # Create the new solution by *applying* the move
        # (The neighborhood strategy MUST return a copy)
        new_solution = self.neighborhood.apply_move(
            self.current_solution, best_allowed_move
        )
        
        # Set its cost (we already calculated it)
        new_solution.cost = best_allowed_cost
        
        # Update the current solution for the *next* iteration
        self.current_solution = new_solution
        
        # Return the new solution to the main "solve()" loop,
        # which will pass it to "_update_solution()".
        return new_solution