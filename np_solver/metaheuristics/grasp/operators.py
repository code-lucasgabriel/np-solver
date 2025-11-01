"""
TODO
1. Random plus greedy
2. Sampled greedy construction
3. Reactive GRASP
4. Cost perturbations
5. Bias functions
6. Intelligent construction
7. POP in construction
"""


from typing import TypeVar, Generic, List, Dict, Optional
from np_solver.core.solution import BaseSolution
import random

E = TypeVar('E')

class RandomPlusGreedyConstruction(Generic[E]):
    """
    Implements the standard GRASP construction heuristic.
    
    This provides the "classic" implementation for _build_rcl, where 
    the RCL is built using the alpha parameter on the *full* candidate list.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Pass args up the MRO
        if getattr(self, 'verbose', False):
            print("Using RandomPlusGreedyConstructionMixin")

    def _build_rcl(self) -> List[E]:
        costs = {c: self.obj_function.evaluate_insertion_cost(c, self.sol) for c in self.cl}
        if not costs: 
            return []
        
        min_cost = min(costs.values())
        max_cost = max(costs.values())

        if max_cost == min_cost:
            return list(costs.keys())
        
        if self.sense == "min":
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            rcl = [c for c, cost in costs.items() if cost <= threshold]
        else: 
            threshold = max_cost - self.alpha * (max_cost - min_cost)
            rcl = [c for c, cost in costs.items() if cost >= threshold]

        return rcl

class SampledGreedyConstruction(Generic[E]):
    """
    Implements the Sampled Greedy construction heuristic.
    
    This overrides _build_rcl to only evaluate a *sample* of the 
    candidates in CL, building the RCL from that smaller set.
    """
    def __init__(self, *args, sample_size: int = 10, **kwargs):
        """
        Adds the sample_size parameter.
        
        Args:
            sample_size (int): The number of candidates to sample from CL.
        """
        super().__init__(*args, **kwargs) # Pass other args up the MRO
        self.sample_size: int = sample_size
        if getattr(self, 'verbose', False):
            print(f"Using SampledGreedyConstructionMixin with k={self.sample_size}")

    def _build_rcl(self) -> List[E]:
        """
        Builds the RCL by sampling 'k' candidates from CL.
        'self' refers to the combined GRASP instance.
        """
        if not self.cl:
            return []

        # Ensure sample size is not larger than the candidate list
        k = min(self.sample_size, len(self.cl))
        
        # Sample k candidates from CL
        sampled_cl = self.rng.sample(self.cl, k)
        
        # Calculate costs *only* for the sampled candidates
        costs = {c: self.obj_function.evaluate_insertion_cost(c, self.sol) for c in sampled_cl}
        
        if not costs: 
            return []

        min_cost = min(costs.values())
        max_cost = max(costs.values())

        if max_cost == min_cost:
            threshold = min_cost
        else:
            threshold = min_cost + self.alpha * (max_cost - min_cost)
        
        rcl = [c for c, cost in costs.items() if cost <= threshold]

        if self.sense == "min":
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            rcl = [c for c, cost in costs.items() if cost <= threshold]
        else:
            threshold = max_cost - self.alpha * (max_cost - min_cost)
            rcl = [c for c, cost in costs.items() if cost >= threshold]

        return rcl

class ReactiveGRASP(Generic[E]):
    """
    Implements the Reactive GRASP mechanism by adapting the 'alpha'
    parameter dynamically based on solution quality.
    
    This mixin overrides the '_initialize_reactive', 
    '_update_alpha_before_construction', and '_update_alpha_after_ls' hooks.
    """
    def __init__(self, 
                 *args, 
                 reactive_alphas: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                 reactive_update_period: int = 25,
                 reactive_beta: float = 6.0,
                 **kwargs):
        """
        Initializes the Reactive GRASP parameters.
        
        Args:
            reactive_alphas (List[float]): The discrete set of alpha values.
            reactive_update_period (int): Iterations between probability updates.
            reactive_beta (float): Exponent for the quality update calculation.
        """
        super().__init__(*args, **kwargs) # Pass other args up the MRO
        self.reactive_alphas: List[float] = reactive_alphas
        self.reactive_update_period: int = reactive_update_period
        self.reactive_beta: float = reactive_beta
        
        if getattr(self, 'verbose', False):
            print(f"Using ReactiveGRASPMixin with {len(self.reactive_alphas)} alphas.")
    
    def _initialize_reactive(self) -> None:
        """Initializes probabilities and average costs for each alpha."""
        n_alphas = len(self.reactive_alphas)
        self.reactive_probs: List[float] = [1.0 / n_alphas] * n_alphas
        self.reactive_avg_costs: List[float] = [0.0] * n_alphas
        self.reactive_counts: List[int] = [0] * n_alphas
        self._current_alpha_index: int = 0
        # Init average costs with a high value to encourage exploration
        # This requires a "dummy" evaluation, or just setting to infinity
        # For simplicity, we'll let 0.0 be the starting point.

    def _update_alpha_before_construction(self, iteration: int) -> None:
        """Selects an alpha based on current probabilities."""
        # Select alpha index
        self._current_alpha_index = self.rng.choices(
            population=range(len(self.reactive_alphas)),
            weights=self.reactive_probs,
            k=1
        )[0]
        
        # Set the 'self.alpha' that _build_rcl will use
        self.alpha = self.reactive_alphas[self._current_alpha_index]

        # Update probabilities periodically
        if iteration > 0 and iteration % self.reactive_update_period == 0:
            self._update_reactive_probabilities()

    def _update_alpha_after_ls(self, solution: BaseSolution[E]) -> None:
        """Updates the average cost for the alpha that was just used."""
        if solution is None or solution.cost is None:
            return

        idx = self._current_alpha_index
        current_cost = solution.cost
        
        # Update the running average cost for this alpha
        count = self.reactive_counts[idx]
        avg_cost = self.reactive_avg_costs[idx]
        
        self.reactive_avg_costs[idx] = ((avg_cost * count) + current_cost) / (count + 1)
        self.reactive_counts[idx] += 1

    def _update_reactive_probabilities(self) -> None:
        """Recalculates the selection probabilities for each alpha."""
        if self.best_sol is None or self.best_sol.cost is None:
            return 

        z_best = self.best_sol.cost
        # Avoid division by zero if best cost is 0
        z_best = z_best if z_best != 0 else 1e-9 
        
        qualities: List[float] = []
        for i, avg_cost in enumerate(self.reactive_avg_costs):
            if self.reactive_counts[i] == 0:
                qualities.append(1.0) # Encourage exploration
            else:
                # Use 1e-9 for avg_cost if it's 0 to avoid zero division
                safe_avg_cost = avg_cost if avg_cost != 0 else 1e-9
                
                # --- MODIFIED ---
                if self.sense == "min":
                    # Quality = (best / avg). Lower avg -> higher quality
                    q = (z_best / safe_avg_cost) ** self.reactive_beta
                else: # "max"
                    # Quality = (avg / best). Higher avg -> higher quality
                    q = (safe_avg_cost / z_best) ** self.reactive_beta
                # --- END MODIFIED ---
                qualities.append(q)
        
        sum_qualities = sum(qualities)
        
        if sum_qualities == 0:
            # Should not happen if we guard against z_best=0, but as a fallback:
            n_alphas = len(self.reactive_alphas)
            self.reactive_probs = [1.0 / n_alphas] * n_alphas
        else:
            self.reactive_probs = [q / sum_qualities for q in qualities]
        
        if self.verbose:
            print(f"  (Reactive) Alphas: {self.reactive_alphas}")
            print(f"  (Reactive) Costs:  {[round(c, 2) for c in self.reactive_avg_costs]}")
            print(f"  (Reactive) Probs:  {[round(p, 3) for p in self.reactive_probs]}")
