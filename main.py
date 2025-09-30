from problems.SCQBF.SCQBF import SCQBF
import time
import os

instances_path = os.path.join(os.getcwd(), "problems/SCQBF/instances")
filepath = os.path.join(instances_path, "scqbf025")

SCQBF(filepath)

# start_time = time.time()

# try:
# 1. Initialize the Genetic Algorithm solver for QBF
# ga = GA_QBF(
#     generations=1000,
#     pop_size=100,
#     mutation_rate=1.0 / 100.0,
#     filename=filepath
# )



# # 2. Run the solver to find the best solution
# best_sol = ga.solve()
# print(f"{best_sol}")

# except FileNotFoundError:
#     print("Error: Instance file 'instances/qbf/qbf100' could not be found.")
# except Exception as e:
#     print(f"An error occurred: {e}")

# # Record the ending time and calculate the duration
# end_time = time.time()
# total_time = end_time - start_time

# print(f"Time = {total_time:.3f} seconds")
