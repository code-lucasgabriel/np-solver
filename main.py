from problems.SCQBF.SCQBF import SCQBF
from problems.SCQBF.solver.GA_SCQBF import GA_SCQBF
import time
import os
from reporting.logger import Report
from metaheuristics import assemble
from metaheuristics.ga.operators import LatinHypercubeInitializer, StochasticUniversalSelection

def main():
    """
    Main function to run the genetic algorithm on all instances and generate a report.
    """
    base_path = os.getcwd()
    instances_path = os.path.join(base_path, "problems/SCQBF/instances")
    report_path = os.path.join(instances_path, "report")
    mixin_configs = [[LatinHypercubeInitializer, StochasticUniversalSelection], [LatinHypercubeInitializer], [StochasticUniversalSelection], []]
    
    for mixins in mixin_configs:
        with Report(report_path, "SCQBF") as report_manager:
            print(f"Report will be saved to: {report_manager.filepath}")
            
            try:
                instance_files = [f for f in os.listdir(instances_path) if os.path.isfile(os.path.join(instances_path, f))]
            except FileNotFoundError:
                print(f"Error: The directory '{instances_path}' was not found.")
                return

            if not instance_files:
                print("No instance files found in the directory.")
                return
                
            for filename in instance_files:
                filepath = os.path.join(instances_path, filename)
                print(f"\nProcessing instance: {filename}...")

                start_time = time.time()

                try:
                    ga = assemble(
                        GA_SCQBF,
                        mixins,
                        generations=10000,
                        pop_size=100,
                        mutation_rate=1.0 / 100.0,
                        filename=filepath
                    )

                    best_sol = ga.solve()
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    print(f"  -> Best Solution: {best_sol}")
                    print(f"  -> Time = {total_time:.3f} seconds")

                    report_manager.write_result(filename, best_sol, total_time)

                except Exception as e:
                    error_message = f"Failed to process {filename}: {e}"
                    print(f"  -> {error_message}")
                    report_manager.write_result(filename, f"ERROR: {e}", 0)


    print(f"\nAll instances processed. Report saved successfully!")


if __name__ == "__main__":
    main()