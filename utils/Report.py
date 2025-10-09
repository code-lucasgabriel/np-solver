from datetime import datetime
import os

class Report:
    """
    A class to handle the creation and management of execution reports.

    This class creates a timestamped file in a specified directory and provides
    methods to write formatted results to it. It is designed to be used with
    a 'with' statement to ensure the file is properly closed.
    """
    def __init__(self, report_directory, problem_name):

        os.makedirs(report_directory, exist_ok=True)

        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        self.filename = f"{problem_name}_{timestamp}.txt"
        self.filepath = os.path.join(report_directory, self.filename)

        self.file_handler = open(self.filepath, 'w')
        
    def write_result(self, instance_name, best_solution, execution_time):
        """
        Writes the result of a single instance execution to the report file.

        Args:
            instance_name (str): The name of the instance file.
            best_solution (any): The best solution object returned by the solver.
            execution_time (float): The time taken to solve the instance in seconds.
        """
        self.file_handler.write(f"--- Instance: {instance_name} ---\n")
        self.file_handler.write(f"Best Solution: {best_solution}\n")
        self.file_handler.write(f"Execution Time: {execution_time:.3f} seconds\n\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_handler.close()