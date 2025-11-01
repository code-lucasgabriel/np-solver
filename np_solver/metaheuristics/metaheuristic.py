from abc import ABC, abstractmethod
from np_solver.core import BaseProblemInstance, BaseEvaluator, BaseSolution
import time

class BaseMetaheuristic(ABC):
    """
    The abstract base class for all metaheuristic algorithms.

    This class uses the **Template Method Pattern**. It provides the main "solve()" loop, which defines the fixed "skeleton" of a metaheuristic:
    1. Initialize
    2. Loop (Iterate -> Update Best)
    3. Terminate

    To create a new algorithm (e.g., TabuSearch, GeneticAlgorithmn, GRASP, ALNS, etc), you MUST subclass this class and implement the two abstract methods:
      * "_initialize_run()"
      * "_iterate()"
    
    The base class handles all the boilerplate logic, including
    time limits, iteration counting, and tracking the best-found solution.

    Attributes:
        problem (BaseProblemInstance): The user-defined problem instance evaluator (BaseEvaluator): The user-defined evaluator (objective func). time_limit (int): Maximum run time in seconds. max_iterations (Optional[int]): Maximum number of iterations. best_solution (BaseSolution): Stores the best solution found so far. current_solution (Optional[BaseSolution]): Stores the solution for the current iteration.
    """
    def __init__(self, **kwargs):
        """
        Initializes the base metaheuristic.

        :param problem: The user's problem instance.
        :param evaluator: The user's solution evaluator.
        :param kwargs: Optional keyword arguments.
                       - 'time_limit' (int): Max time in seconds. (Default: 600)
                       - 'max_iterations' (int): Max iterations. (Default: 1000)
        """
        self.time_limit = kwargs.get("time_limit", 600) # Default is 10min
        self.max_iterations = kwargs.get("max_iterations", None) # Default is None

    def _initialize_run(self):
        """
        (Optional) Initializes the algorithm.

        This method is called once at the beginning of "solve()". Its **primary responsibility** is to set the foundation of before the execution of the search loop.
        """
        pass

    @abstractmethod
    def _iterate(self):
        """
        (Abstract) Performs one complete iteration of the algorithm.

        This method is called inside the main "solve()" loop. It should contain the core logic of the metaheuristic (e.g., generate neighbors, create a new generation).
        
        The returned solution will be automatically checked against the best-so-far by the "_update_solution()" method.

        :return: A "BaseSolution" object (or "None") representing the candidate solution found in this iteration.
        """
        pass

    def _terminate(self):
        """
        Checks if the algorithm should terminate.

        This is called at the start of each loop in "solve()". It checks against "self.time_limit" and "self.max_iterations".

        A user *can* override this to add custom termination (e.g., convergence checks), but it's not required.

        :return: True if the algorithm should stop, False otherwise.
        """
        if time.time()>(self.start_time+self.time_limit):
            return True
        
        if self.max_iterations is not None:
            if self.current_iteration>self.max_iterations: # maximum iterations reached
                return True
        return False
    
    def _update_solution(self, candidate_sol: BaseSolution): 
        """
        Compares a candidate solution to the best-so-far.
        """
        if self.is_minimizing:
            if candidate_sol.cost < self.best_solution.cost:
                self.best_solution = candidate_sol
        else: # Maximizing
            if candidate_sol.cost > self.best_solution.cost:
                self.best_solution = candidate_sol

    def _state_setup(self, problem: BaseProblemInstance, evaluator: BaseEvaluator, initial_solution: BaseSolution):
        """
        Initializes all state related to a specific problem/evaluator.
        Called once at the start of solve().
        """
        self.problem = problem
        self.evaluator = evaluator
        self.current_solution = initial_solution

        # evaluate the initial solution cost, in case it was not set before
        self.current_solution.cost = self.evaluator.evaluate(self.current_solution)

        self.best_solution = self.current_solution.copy()
        
        self.is_minimizing = self.evaluator.sense == BaseEvaluator.ObjectiveSense.MINIMIZE

        if self.is_minimizing:
            self._infeasible_cost = float('inf')
        else:
            self._infeasible_cost = float('-inf')

    def solve(self, problem: BaseProblemInstance, evaluator: BaseEvaluator, initial_solution: BaseSolution) -> BaseSolution:
        """
        The main entry point to run the metaheuristic.

        This method is the "template" that executes the algorithm's skeleton. You should **not** override this method in subclasses.

        The process is:
        1. Start timer and counters.
        2. Call "_initialize_run()" to set the foundations before executing the main solver loop.
        3. Update best solution with the initial one passed by the user.
        4. Loop until "_terminate()" is True:
           - Call "_iterate()" to get a new candidate.
           - Call "_update_solution()" with the candidate.
           - Increment iteration counter.
        5. Call "problem.report_experiment()"
        6. Return "self.best_solution".

        :return: The best "BaseSolution" found by the algorithm.
        """
        self._state_setup(problem, evaluator, initial_solution)

        self.start_time = time.time()
    
        self.current_iteration = 0

        self._initialize_run()
        
        while not self._terminate():
            candidate_sol = self._iterate()
            try:
                self._update_solution(candidate_sol)
            except:
                print("Coundn't update solution. Trying again.")
                continue
            self.current_iteration+=1
        
        self.problem.report_experiments(self.best_solution)

        return self.best_solution
