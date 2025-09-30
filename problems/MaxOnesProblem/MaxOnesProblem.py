from v0.problem.Problem import Problem
from typing import List
import random

class MaxOnesProblem(Problem):
    """
    A simple example problem: find a binary vector with the maximum number of 1s.
    The goal is to maximize the sum of the vector elements.
    """
    def __init__(self, vector_length: int):
        super().__init__()
        self.vector_length = vector_length

    def get_initial_solution(self) -> List[int]:
        """Generates a random binary vector."""
        return [random.randint(0, 1) for _ in range(self.vector_length)]

    def evaluate(self, solution: List[int]) -> float:
        """The score is the sum of the elements."""
        return sum(solution)

    def get_neighbors(self, solution: List[int]) -> List[List[int]]:
        """Neighbors are created by flipping a single bit."""
        neighbors = []
        for i in range(self.vector_length):
            neighbor = solution[:]
            neighbor[i] = 1 - neighbor[i] # Flip the bit
            neighbors.append(neighbor)
        return neighbors