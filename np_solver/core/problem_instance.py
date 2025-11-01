# in solver/problem.py
from abc import ABC, abstractmethod
from np_solver.core.solution import BaseSolution

class BaseProblemInstance(ABC):
    """
    Abstract base class for a problem instance.

    This class is a "data container". It is responsible for
    loading, holding, and parsing all the static data for a
    problem instance (e.g., distances, demands, time windows).
    """

    @abstractmethod
    def read_instance(self, filename: str) -> None:
        """
        Reads, parses, and stores the problem data from a file.
        
        This method should populate the instance's attributes
        (e.g., self.distance_matrix, self.customer_list, etc.).
        """
        pass
    
    @abstractmethod
    def get_instance_name(self) -> str:
        """
        Returns a unique name for the instance.
        """
        pass

    @abstractmethod
    def get_domain_size(self) -> int:
        """
        Returns the size of the problem domain.
        
        For a GA, this is often the chromosome_size.
        For a VRP, this might be the number of customers.
        """
        pass
    
    @abstractmethod
    def report_experiment(self, filename: str, solution: BaseSolution) -> None:
        """
        Writes the given solution to a formatted results to a JSON file.
        """
        pass