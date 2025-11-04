# 🧬 NP Solver

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NP Solver** is a flexible, extensible Python framework for solving complex combinatorial optimization problems using state-of-the-art metaheuristic algorithms. Since finding exact optimal solutions to NP-hard problems is often computationally intractable, this library provides production-ready implementations of powerful metaheuristics that find high-quality near-optimal solutions in practical time.

## 🌟 Features

- **Multiple Metaheuristics**: Implementations of GRASP, Tabu Search (TS), Genetic Algorithms (GA), and Adaptive Large Neighborhood Search (ALNS)
- **Template Method Pattern**: Clean, extensible architecture that separates algorithm logic from problem-specific details
- **Strategy Pattern**: Pluggable components for neighborhoods, tabu lists, selection operators, and more
- **Delta Evaluation**: Optional fast incremental cost evaluation for performance-critical applications
- **Type Safety**: Full type hints with generic typing for better IDE support and fewer bugs
- **Production Ready**: Includes logging, reporting, and experiment tracking capabilities

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Framework Architecture](#-framework-architecture)
- [Project Structure](#-project-structure)
- [How to Use](#-how-to-use)
- [Available Metaheuristics](#-available-metaheuristics)
- [Engineering Challenges & Trade-offs](#-engineering-challenges--trade-offs)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Installation

### From PyPI (Recommended)

```bash
pip install np-solver
```

### From Source

```bash
# Clone the repository
git clone https://github.com/code-lucasgabriel/np-solver.git
cd np-solver

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.9 or higher
- NumPy >= 1.21.0

---

## ⚡ Quick Start

Here's a minimal example to get you started:

```python
from np_solver.core import BaseProblemInstance, BaseEvaluator, BaseSolution
from np_solver.metaheuristics.ts import TabuSearch
from np_solver.metaheuristics.ts.components import SwapNeighborhood

# 1. Define your problem instance
class MyProblem(BaseProblemInstance):
    def read_instance(self, filename: str) -> None:
        # Load your problem data
        pass
    
    def get_instance_name(self) -> str:
        return "my_instance"
    
    def get_domain_size(self) -> int:
        return 100  # e.g., number of cities, items, etc.
    
    def report_experiment(self, filename: str, solution: BaseSolution) -> None:
        # Save results
        pass

# 2. Define your evaluator (objective function)
class MyEvaluator(BaseEvaluator):
    sense = BaseEvaluator.ObjectiveSense.MINIMIZE
    
    def constraints(self, sol: BaseSolution) -> bool:
        # Check if solution is feasible
        return True
    
    def objective_function(self, sol: BaseSolution) -> float:
        # Calculate solution cost
        return sum(sol)  # Example: simple sum

# 3. Create an initial solution
problem = MyProblem()
evaluator = MyEvaluator(problem)
initial_solution = BaseSolution.from_list([1, 2, 3, 4, 5], evaluator)

# 4. Configure and run the metaheuristic
solver = TabuSearch(
    tenure=10,
    neighborhood_strategy=SwapNeighborhood(),
    time_limit=60,  # 60 seconds
    max_iterations=1000
)

best_solution = solver.solve(problem, evaluator, initial_solution)
print(f"Best solution found: {best_solution}")
```

---

## 🏗️ Framework Architecture

NP Solver is built on solid software engineering principles:

### Design Patterns

1. **Template Method Pattern** (`BaseMetaheuristic`)
   - Defines the skeleton of the optimization algorithm
   - Subclasses implement `_initialize_run()` and `_iterate()`
   - Framework handles timing, iteration counting, and best solution tracking

2. **Strategy Pattern** (Neighborhoods, Tabu Lists, Selection Operators)
   - Pluggable components for algorithm customization
   - Easy to swap implementations without changing core logic

3. **Abstract Factory Pattern** (Problem Instance, Evaluator, Solution)
   - Clean separation between problem definition and solving logic
   - Type-safe generic implementations

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   BaseMetaheuristic                     │
│  (Template Method: solve(), _initialize(), _iterate())  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬──────────────┐
         │                       │              │
    ┌────▼─────┐         ┌──────▼──────┐  ┌───▼────┐
    │   GRASP  │         │ TabuSearch  │  │   GA   │
    └──────────┘         └─────────────┘  └────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
            ┌───────▼────────┐      ┌───────▼────────┐
            │  Neighborhood  │      │   TabuList     │
            │   (Strategy)   │      │   (Strategy)   │
            └────────────────┘      └────────────────┘
```

---

## 📂 Project Structure

```
np-solver/
├── README.md                      # This file
├── pyproject.toml                 # Project metadata and dependencies
├── LICENSE                        # MIT License
│
└── np_solver/                     # Main package
    │
    ├── core/                      # Core abstractions
    │   ├── __init__.py
    │   ├── problem_instance.py    # BaseProblemInstance (data container)
    │   ├── evaluator.py           # BaseEvaluator (objective function)
    │   └── solution.py            # BaseSolution (solution representation)
    │
    ├── metaheuristics/            # Metaheuristic implementations
    │   ├── __init__.py
    │   ├── metaheuristic.py       # BaseMetaheuristic (template method)
    │   │
    │   ├── grasp/                 # GRASP implementation
    │   │   ├── __init__.py
    │   │   └── operators.py       # Construction operators & variants
    │   │
    │   ├── ts/                    # Tabu Search implementation
    │   │   ├── __init__.py
    │   │   ├── solver.py          # TabuSearch main class
    │   │   ├── interface.py       # Strategy interfaces
    │   │   └── components.py      # Concrete strategies
    │   │
    │   ├── ga/                    # Genetic Algorithm implementation
    │   │   ├── __init__.py
    │   │   ├── initial_implementation.py  # BaseGA class
    │   │   └── operators.py       # Crossover, mutation, selection
    │   │
    │   └── alns/                  # ALNS implementation
    │       ├── __init__.py
    │       ├── solver.py          # ALNS main class
    │       ├── interface.py       # Destroy/Repair interfaces
    │       └── components.py      # Concrete operators
    │
    └── reporting/                 # Logging and experiment tracking
        ├── __init__.py
        └── logger.py              # Report class for results
```

---

## 📖 How to Use

### Step 1: Define Your Problem

Inherit from `BaseProblemInstance` to create a container for your problem data:

```python
from np_solver.core import BaseProblemInstance, BaseSolution

class TravelingSalesmanProblem(BaseProblemInstance):
    def __init__(self):
        self.distance_matrix = None
        self.num_cities = 0
    
    def read_instance(self, filename: str) -> None:
        """Load distance matrix from file"""
        with open(filename, 'r') as f:
            self.num_cities = int(f.readline())
            self.distance_matrix = [
                [float(x) for x in line.split()] 
                for line in f.readlines()
            ]
    
    def get_instance_name(self) -> str:
        return "TSP_instance"
    
    def get_domain_size(self) -> int:
        return self.num_cities
    
    def report_experiment(self, filename: str, solution: BaseSolution) -> None:
        """Save solution to file"""
        with open(f"{filename}_result.json", 'w') as f:
            json.dump({
                'tour': list(solution),
                'cost': solution.cost
            }, f)
```

### Step 2: Define Your Evaluator

Inherit from `BaseEvaluator` to implement your objective function:

```python
from np_solver.core import BaseEvaluator, BaseSolution

class TSPEvaluator(BaseEvaluator):
    sense = BaseEvaluator.ObjectiveSense.MINIMIZE
    
    def constraints(self, sol: BaseSolution) -> bool:
        """Check if tour visits all cities exactly once"""
        return (len(sol) == self.problem.num_cities and 
                len(set(sol)) == self.problem.num_cities)
    
    def objective_function(self, sol: BaseSolution) -> float:
        """Calculate total tour distance"""
        total_distance = 0
        for i in range(len(sol)):
            city_a = sol[i]
            city_b = sol[(i + 1) % len(sol)]
            total_distance += self.problem.distance_matrix[city_a][city_b]
        return total_distance
    
    # Optional: Override for performance (O(1) instead of O(n))
    def evaluate_swap_cost(self, city1: int, city2: int, 
                          sol: BaseSolution) -> float:
        """Fast delta evaluation for swapping two cities"""
        # Implement O(1) cost difference calculation
        # (details omitted for brevity)
        pass
```

### Step 3: Create an Initial Solution

```python
import random

problem = TravelingSalesmanProblem()
problem.read_instance("instances/tsp_50.txt")

evaluator = TSPEvaluator(problem)

# Random initial tour
cities = list(range(problem.num_cities))
random.shuffle(cities)
initial_solution = BaseSolution.from_list(cities, evaluator)
```

### Step 4: Choose and Configure a Metaheuristic

```python
from np_solver.metaheuristics.ts import TabuSearch
from np_solver.metaheuristics.ts.components import (
    SwapNeighborhood, 
    DequeTabuList, 
    AspirationByBest
)

solver = TabuSearch(
    tenure=15,
    neighborhood_strategy=SwapNeighborhood(),
    tabu_list=DequeTabuList(tenure=15),
    aspiration_criteria=AspirationByBest(),
    time_limit=300,      # 5 minutes
    max_iterations=5000
)

best_solution = solver.solve(problem, evaluator, initial_solution)
print(f"Best tour: {list(best_solution)}")
print(f"Tour length: {best_solution.cost}")
```

---

## 🔧 Available Metaheuristics

### 1. **GRASP** (Greedy Randomized Adaptive Search Procedure)

Two-phase iterative method:
- **Construction Phase**: Build a solution using a greedy randomized approach
- **Local Search Phase**: Improve the solution using local optimization

**Variants Available**:
- Random + Greedy Construction
- Sampled Greedy Construction
- Reactive GRASP (adaptive alpha parameter)

```python
from np_solver.metaheuristics.grasp import GRASP

solver = GRASP(
    alpha=0.3,           # RCL parameter
    time_limit=60,
    max_iterations=100
)
```

### 2. **Tabu Search (TS)**

Memory-based metaheuristic that avoids cycling by maintaining a tabu list of recent moves.

**Key Components**:
- Neighborhood strategies (Swap, Insert, Relocate)
- Tabu list management (Deque-based, Set-based)
- Aspiration criteria

```python
from np_solver.metaheuristics.ts import TabuSearch
from np_solver.metaheuristics.ts.components import SwapNeighborhood

solver = TabuSearch(
    tenure=10,
    neighborhood_strategy=SwapNeighborhood(),
    time_limit=60
)
```

### 3. **Genetic Algorithm (GA)**

Population-based evolutionary algorithm using selection, crossover, and mutation.

**Features**:
- Tournament selection (default)
- Stochastic Universal Sampling (SUS) mixin
- Latin Hypercube Sampling (LHS) initialization
- 2-point crossover
- Elitism

```python
from np_solver.metaheuristics.ga import BaseGA

class MyGA(BaseGA):
    def _decode(self, chromosome):
        # Convert chromosome to solution
        pass

solver = MyGA(
    evaluator=evaluator,
    generations=100,
    pop_size=50,
    mutation_rate=0.01
)
```

### 4. **ALNS** (Adaptive Large Neighborhood Search)

Iteratively destroys and repairs solutions using multiple operators with adaptive weights.

```python
from np_solver.metaheuristics.alns import ALNS
from np_solver.metaheuristics.alns.components import (
    RandomRemoval,
    GreedyInsertion
)

solver = ALNS(
    destroy_operators=[RandomRemoval()],
    repair_operators=[GreedyInsertion()],
    time_limit=60
)
```

---

## ⚖️ Engineering Challenges & Trade-offs

### 1. **Abstraction vs. Performance**

**Challenge**: Balancing clean abstractions with computational efficiency.

**Solution**: 
- Provide default implementations in base classes that are correct but slow
- Allow users to override with problem-specific optimizations
- Delta evaluation methods (`evaluate_swap_cost`, etc.) enable O(1) operations instead of O(n)

**Trade-off**: More code for users to write, but 10-1000× speedups for large problems.

### 2. **Type Safety with Generics**

**Challenge**: Supporting diverse solution representations (permutations, binary strings, graphs) in a type-safe way.

**Solution**: 
```python
E = TypeVar('E')  # Element type

class BaseSolution(list, Generic[E]):
    """Can hold any element type"""
```

**Trade-off**: More complex type signatures, but catches bugs at development time instead of runtime.

### 3. **Strategy Pattern Complexity**

**Challenge**: Making algorithms customizable without overwhelming users.

**Solution**:
- Provide sensible defaults for all strategy components
- Allow gradual complexity: start simple, customize as needed
- Clear interfaces (e.g., `TSNeighborhood`, `TSTabuList`)

**Trade-off**: More classes to understand, but maximum flexibility for advanced users.

### 4. **Termination Criteria**

**Challenge**: When should the algorithm stop?

**Solution**: Multiple criteria checked in `_terminate()`:
- Time limit (wall-clock time)
- Maximum iterations
- Custom criteria (overridable)

**Trade-off**: Cannot guarantee finding optimal solution, but ensures practical runtime.

### 5. **Solution Representation**

**Challenge**: Solutions inherit from `list` but need additional metadata (cost, feasibility).

**Solution**: 
```python
class BaseSolution(list, Generic[E]):
    def __init__(self, other=None):
        super().__init__(other or [])
        self.cost = other.cost if other else float('inf')
```

**Trade-off**: Slight memory overhead, but natural list operations work seamlessly.

### 6. **Memory Management in Population-Based Methods**

**Challenge**: GAs maintain large populations; memory can explode.

**Solution**:
- Deep copy only when necessary
- Reuse population arrays
- Provide `copy()` method for explicit control

**Trade-off**: Users must be careful about when objects are shared vs. copied.

### 7. **Logging and Debugging**

**Challenge**: Algorithms can run for hours; need visibility into progress.

**Solution**:
- Optional `verbose` flag in implementations
- `Report` class for structured experiment logging
- Automatic best solution tracking

**Trade-off**: Some performance overhead when logging is enabled.

---

## 💡 Examples

### Example 1: Knapsack Problem with GRASP

```python
from np_solver.core import BaseProblemInstance, BaseEvaluator, BaseSolution
from np_solver.metaheuristics.grasp import GRASP

class KnapsackProblem(BaseProblemInstance):
    def __init__(self, values, weights, capacity):
        self.values = values
        self.weights = weights
        self.capacity = capacity
    
    def get_domain_size(self):
        return len(self.values)
    
    # ... implement other methods ...

class KnapsackEvaluator(BaseEvaluator):
    sense = BaseEvaluator.ObjectiveSense.MAXIMIZE
    
    def constraints(self, sol: BaseSolution) -> bool:
        total_weight = sum(self.problem.weights[i] for i in sol)
        return total_weight <= self.problem.capacity
    
    def objective_function(self, sol: BaseSolution) -> float:
        return sum(self.problem.values[i] for i in sol)

# Solve
problem = KnapsackProblem([60, 100, 120], [10, 20, 30], 50)
evaluator = KnapsackEvaluator(problem)
initial = BaseSolution()  # Start empty

solver = GRASP(alpha=0.2, max_iterations=100)
solution = solver.solve(problem, evaluator, initial)
```

### Example 2: Scheduling with Tabu Search

See the `examples/` directory in the repository for complete implementations of:
- Traveling Salesman Problem (TSP)
- Vehicle Routing Problem (VRP)
- Job Shop Scheduling (JSP)
- Quadratic Assignment Problem (QAP)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs**: Open an issue with a minimal reproducible example
2. **Suggest features**: Propose new metaheuristics or improvements
3. **Submit PRs**: 
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/amazing-feature`)
   - Commit your changes (`git commit -m 'Add amazing feature'`)
   - Push to the branch (`git push origin feature/amazing-feature`)
   - Open a Pull Request

### Development Setup

```bash
git clone https://github.com/code-lucasgabriel/np-solver.git
cd np-solver
pip install -e ".[dev]"  # Install with dev dependencies
pytest                    # Run tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Lucas Gabriel Monteiro da Costa** - [code.lucasgabriel@gmail.com](mailto:code.lucasgabriel@gmail.com)
- **Pietro Grazzioli Golfeto** - [pggolfeto@gmail.com](mailto:pggolfeto@gmail.com)

---

## 📚 References & Further Reading

- **Metaheuristics**: Gendreau, M., & Potvin, J. Y. (2010). *Handbook of Metaheuristics*. Springer.
- **GRASP**: Resende, M. G., & Ribeiro, C. C. (2003). *Greedy Randomized Adaptive Search Procedures*.
- **Tabu Search**: Glover, F., & Laguna, M. (1998). *Tabu Search*. Springer.
- **Genetic Algorithms**: Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
- **ALNS**: Pisinger, D., & Ropke, S. (2010). *Large Neighborhood Search*. Handbook of Metaheuristics.

---

## 🔗 Links

- **Homepage**: [https://github.com/code-lucasgabriel/np-solver](https://github.com/code-lucasgabriel/np-solver)
- **PyPI**: [https://pypi.org/project/np-solver/](https://pypi.org/project/np-solver/)
- **Issue Tracker**: [https://github.com/code-lucasgabriel/np-solver/issues](https://github.com/code-lucasgabriel/np-solver/issues)
- **Documentation**: Coming soon!

---

<div align="center">

**If you find this framework useful, please consider giving it a ⭐ on GitHub!**

Made with ❤️ by optimization enthusiasts

</div>

