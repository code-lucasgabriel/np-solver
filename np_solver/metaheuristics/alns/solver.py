from typing import List, Optional
from np_solver.core import BaseSolution
from np_solver.metaheuristics import BaseMetaheuristic
from np_solver.metaheuristics.alns.interface import (
    ALNSDestroy, ALNSRepair, ALNSWeightManager, ALNSAcceptance
)

class ALNS(BaseMetaheuristic):
    """
    An Adaptive Large Neighborhood Search (ALNS) metaheuristic implementation
    using the Template Method ("BaseMetaheuristic") and the Strategy 
    Design Pattern.

    This class implements the "_initialize_run" and "_iterate" methods
    from "BaseMetaheuristic". It coordinates the search by delegating
    the logic to the provided strategy objects.
    """
    
    def __init__(self, 
                 destroy_operators: List[ALNSDestroy], 
                 repair_operators: List[ALNSRepair], 
                 weight_manager: ALNSWeightManager, 
                 acceptance_criteria: ALNSAcceptance, 
                 **kwargs):
        """
        Initializes the ALNS algorithm.

        Args:
            destroy_operators (List[ALNSDestroy]): A list of destroy strategies.
            repair_operators (List[ALNSRepair]): A list of repair strategies.
            weight_manager (ALNSWeightManager): Strategy for selecting ops
                                                and managing weights.acceptance_criteria (ALNSAcceptance): Strategy for accepting/rejecting new solutions.
            **kwargs: Passed up to BaseMetaheuristic (e.g., 'time_limit').
        """
        super().__init__(**kwargs)
        
        if not destroy_operators or not repair_operators:
            raise ValueError("Must provide at least one destroy and one repair operator.")
            
        self.destroy_operators = destroy_operators
        self.repair_operators = repair_operators
        self.weight_manager = weight_manager
        self.acceptance_criteria = acceptance_criteria
        
    def _initialize_run(self):
        """
        (Template Method) Initializes the strategies, such as
        resetting operator weights and acceptance criteria (e.g., temperature).
        """
        self.weight_manager._initialize_run(self.destroy_operators, self.repair_operators)
        self.acceptance_criteria._initialize_run()

    def _iterate(self) -> Optional[BaseSolution]:
        """
        (Template Method) Performs one complete iteration of ALNS.
        
        1. Selects destroy/repair operators using the WeightManager.
        2. Applies destroy operator to the current solution.
        3. Applies repair operator to the partial solution.
        4. Evaluates the new candidate solution.
        5. Decides to accept/reject using the AcceptanceCriteria.
        6. Updates operator scores using the WeightManager.
        7. Steps the strategies (for weight updates and cooling).
        8. Returns the *candidate* solution (for best-tracking).
        
        Returns:
            The new candidate "BaseSolution", or "None" if an error occurs.
        """
        
        # 1. Select Operators
        destroy_op, repair_op = self.weight_manager.select_operators()
        
        try:
            # 2. Destroy Solution
            partial_sol = destroy_op.destroy(self.current_solution, self.problem)
            
            # 3. Repair Solution
            candidate_sol = repair_op.repair(partial_sol, self.problem, self.evaluator)
            
            # 4. Evaluate the new candidate
            candidate_cost = self.evaluator.evaluate(candidate_sol)
            candidate_sol.cost = candidate_cost
            
        except Exception as e:
            print(f"Error during destroy/repair: {e}. Skipping iteration.")
            return None # Skip this iteration

        # 5. Decide Acceptance
        is_accepted = self.acceptance_criteria.accept(
            candidate_cost,
            self.current_solution.cost,
            self.best_solution.cost,
            self.evaluator.sense
        )
        
        # Check for new best *before* updating scores
        is_new_best = False
        if (self.is_minimizing and candidate_cost < self.best_solution.cost) or \
           (not self.is_minimizing and candidate_cost > self.best_solution.cost):
            is_new_best = True

        # 6. Update Scores
        self.weight_manager.update_scores(
            destroy_op,
            repair_op,
            candidate_cost,
            self.current_solution.cost,
            self.best_solution.cost,
            self.evaluator.sense,
            is_accepted,
            is_new_best
        )
        
        # 7. Update Current Solution (if accepted)
        if is_accepted:
            self.current_solution = candidate_sol
            
        # 8. Step the strategies (for weight updates / cooling)
        self.weight_manager.step()
        self.acceptance_criteria.step()
        
        # 9. Return the candidate solution to the main "solve()" loop,
        #    which will pass it to "_update_solution()".
        return candidate_sol