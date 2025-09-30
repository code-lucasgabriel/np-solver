from problems.QBF.QBF import QBF
from interface.Solution import Solution


class QBF_Inverse(QBF):
    """
    Represents the inverse of the Quadratic Binary Function (QBF).

    This class is used to transform the QBF maximization problem into a
    minimization problem by simply negating the value of the objective
    function. This is useful for metaheuristics that are designed as
    minimization procedures by default.
    """

    def __init__(self, filename: str):
        """
        Initializes the QBF_Inverse problem by loading the instance file.

        Args:
            filename (str): The path to the file containing the QBF instance.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        super().__init__(filename)

    def evaluate_qbf(self) -> float:
        """
        Returns the negated value of the QBF evaluation.
        """
        return -super().evaluate_qbf()

    def _evaluate_insertion_qbf(self, i: int) -> float:
        """
        Returns the negated value of the insertion cost.
        Note: This overrides a protected-like method from the parent QBF class.
        """
        return -super()._evaluate_insertion_qbf(i)

    def _evaluate_removal_qbf(self, i: int) -> float:
        """
        Returns the negated value of the removal cost.
        Note: This overrides a protected-like method from the parent QBF class.
        """
        return -super()._evaluate_removal_qbf(i)
    
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution[int]) -> float:
        """
        Returns the negated value of the exchange cost.
        """
        return -super().evaluate_exchange_cost(elem_in, elem_out, sol)