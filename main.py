from problems.SCQBF.SCQBF import SCQBF
from problems.SCQBF.solver.GA_SCQBF import GA_SCQBF
import time
import os

instances_path = os.path.join(os.getcwd(), "problems/SCQBF/instances")
filepath = os.path.join(instances_path, "scqbf025")

start_time = time.time()

ga = GA_SCQBF(
    generations=1000,
    pop_size=100,
    mutation_rate=1.0 / 100.0,
    filename=filepath
)

best_sol = ga.solve()
print(f"{best_sol}")

end_time = time.time()
total_time = end_time - start_time

print(f"Time = {total_time:.3f} seconds")
