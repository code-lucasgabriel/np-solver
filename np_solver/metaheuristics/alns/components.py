import random
import math
from typing import List, Dict, Any, Tuple
from np_solver.core import BaseEvaluator
from np_solver.metaheuristics.alns.interface import (
    ALNSAcceptance, ALNSWeightManager, ALNSDestroy, ALNSRepair
)

class SimulatedAnnealingAcceptance(ALNSAcceptance):
    """
    A concrete ALNSAcceptance strategy that uses a
    Simulated Annealing (SA) criterion.
    
    It always accepts better solutions. It accepts worse solutions
    with a probability p = exp(-delta / temperature).
    The temperature cools over time.
    """
    def __init__(self, 
                 initial_temp: float, 
                 cooling_rate: float, 
                 min_temp: float = 0.1):
        """
        Args:
            initial_temp (float): The starting temperature.
            cooling_rate (float): Multiplier for temp (e.g., 0.999).
            min_temp (float): The lowest temperature.
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_temp = initial_temp
        
    def _initialize_run(self):
        """Resets the temperature to its initial value."""
        self.current_temp = self.initial_temp
        
    def accept(self, 
             candidate_cost: float, 
             current_cost: float, 
             best_cost: float, 
             sense: BaseEvaluator.ObjectiveSense) -> bool:
        
        if sense == BaseEvaluator.ObjectiveSense.MINIMIZE:
            delta = candidate_cost - current_cost
        else: # MAXIMIZE
            delta = current_cost - candidate_cost # Invert delta

        # Always accept improving solutions
        if delta < 0:
            return True
            
        # Avoid math domain error if temp is 0
        if self.current_temp <= 0:
            return False
            
        # Accept worsening solutions with probability
        try:
            probability = math.exp(-delta / self.current_temp)
            return random.random() < probability
        except OverflowError:
            return False # Probability is essentially 0

    def step(self):
        """Cools the temperature."""
        self.current_temp = max(
            self.current_temp * self.cooling_rate, 
            self.min_temp
        )


class RouletteWheelManager(ALNSWeightManager):
    """
    A concrete ALNSWeightManager that uses Roulette Wheel selection
    and updates weights based on performance over a "segment"
    of iterations.
    """
    DEFAULT_REWARDS = {
        "new_best": 5,
        "better_than_current": 2,
        "accepted": 1,
        "rejected": 0
    }

    def __init__(self, 
                 segment_size: int = 100, 
                 decay: float = 0.8, 
                 reward_points: Dict[str, float] = None):
        """
        Args:
            segment_size (int): Iterations before weights are recalculated.
            decay (float): The decay factor (lambda) for weights.
                           new_w = w * decay + (1 - decay) * avg_score
            reward_points (Dict): Scores for different outcomes.
        """
        self.segment_size = segment_size
        self.decay = decay
        self.rewards = reward_points if reward_points is not None else self.DEFAULT_REWARDS
        
        self.iterations_in_segment = 0
        
    def _initialize_run(self, 
                        destroy_ops: List[ALNSDestroy], 
                        repair_ops: List[ALNSRepair]):
        """Initializes all weights to 1 and scores/usages to 0."""
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        
        self.d_weights = [1.0] * len(destroy_ops)
        self.d_scores = [0.0] * len(destroy_ops)
        self.d_usages = [0.0] * len(destroy_ops)
        
        self.r_weights = [1.0] * len(repair_ops)
        self.r_scores = [0.0] * len(repair_ops)
        self.r_usages = [0.0] * len(repair_ops)
        
        self.iterations_in_segment = 0
        
    def select_operators(self) -> Tuple[ALNSDestroy, ALNSRepair]:
        """Selects ops using random.choices (roulette wheel)."""
        destroy_op = random.choices(self.destroy_ops, weights=self.d_weights, k=1)[0]
        repair_op = random.choices(self.repair_ops, weights=self.r_weights, k=1)[0]
        return destroy_op, repair_op

    def update_scores(self, 
                      destroy_op: ALNSDestroy, 
                      repair_op: ALNSRepair, 
                      candidate_cost: float, 
                      current_cost: float, 
                      best_cost: float, 
                      sense: BaseEvaluator.ObjectiveSense,
                      is_accepted: bool,
                      is_new_best: bool):
        
        # 1. Determine the score for this outcome
        score = 0.0
        if is_new_best:
            score = self.rewards["new_best"]
        elif is_accepted:
            # Check if it was accepted because it was better than current
            # (but not better than best)
            is_better_than_current = (sense == BaseEvaluator.ObjectiveSense.MINIMIZE and candidate_cost < current_cost) or \
                                     (sense == BaseEvaluator.ObjectiveSense.MAXIMIZE and candidate_cost > current_cost)
            
            if is_better_than_current:
                score = self.rewards["better_than_current"]
            else: # Accepted, but worse (SA)
                score = self.rewards["accepted"]
        else: # Rejected
            score = self.rewards["rejected"]

        # 2. Add score and usage to the operators
        try:
            d_idx = self.destroy_ops.index(destroy_op)
            self.d_scores[d_idx] += score
            self.d_usages[d_idx] += 1
            
            r_idx = self.repair_ops.index(repair_op)
            self.r_scores[r_idx] += score
            self.r_usages[r_idx] += 1
        except ValueError:
            # Should not happen if ops are managed correctly
            print("Warning: Could not find operator in list to update score.")
            pass
            
    def step(self):
        """Tracks segment iterations and triggers weight update."""
        self.iterations_in_segment += 1
        
        if self.iterations_in_segment >= self.segment_size:
            self._update_weights()
            self.iterations_in_segment = 0 # Reset counter

    def _update_weights(self):
        """
        Recalculates all operator weights based on their average
        scores during the last segment.
        """
        # Update Destroy Weights
        for i in range(len(self.d_weights)):
            if self.d_usages[i] > 0:
                avg_score = self.d_scores[i] / self.d_usages[i]
                self.d_weights[i] = self.d_weights[i] * self.decay + (1 - self.decay) * avg_score
            else:
                # If not used, just decay weight
                self.d_weights[i] = self.d_weights[i] * self.decay
            # Ensure weight is non-negative
            self.d_weights[i] = max(0.01, self.d_weights[i])

        # Update Repair Weights
        for i in range(len(self.r_weights)):
            if self.r_usages[i] > 0:
                avg_score = self.r_scores[i] / self.r_usages[i]
                self.r_weights[i] = self.r_weights[i] * self.decay + (1 - self.decay) * avg_score
            else:
                self.r_weights[i] = self.r_weights[i] * self.decay
            self.r_weights[i] = max(0.01, self.r_weights[i])

        # Reset scores and usages for the next segment
        self.d_scores = [0.0] * len(self.destroy_ops)
        self.d_usages = [0.0] * len(self.destroy_ops)
        self.r_scores = [0.0] * len(self.repair_ops)
        self.r_usages = [0.0] * len(self.repair_ops)