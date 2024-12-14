# MIP-Solver

`mip-solver` is a Python-based framework designed to facilitate the experimentation and analysis of Mixed Integer Programming (MIP) problems. The framework supports loading MPS files, permuting problem variables, and comparing the performance of solvers on original and permuted problems.

## Features

- **Load and Solve MIP Problems**: Read problems from MPS files and solve them using the Gurobi solver.
- **Random Permutations**: Generate permuted versions of MIP problems to analyze solver performance under different variable orderings.
- **Experimentation Framework**: Log results, compare solutions, and evaluate differences in objective values between original and permuted problems.

## Folder Structure

```plaintext
mip-solver/
├── core/
│   ├── logging_handler.py           # Logging utility for structured experiment logs
│   ├── problem_permutator.py        # Handles variable permutation for MIP problems
│   └── optimization_experiment.py   # Main class for running optimization experiments
├── input/
│   └── example3.mps                 # Example MPS file for experimentation
├── main.py                          # Entry point to run the optimization experiment
├── experiment.ipynb                 # Jupyter notebook for interactive experimentation
```

## Requirements

- Python 3.8+
- Gurobi Solver (with a valid license)
- Required Python packages (see [Installation](#installation))

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mip-solver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Gurobi and configure your license:
   Follow the instructions on the [Gurobi website](https://www.gurobi.com/documentation/).

## Usage

### Running the Experiment

1. Place your MPS file in the `input/` directory.
2. Update the `file_path` in `main.py` to point to your MPS file:
   ```python
   file_path = "input/example3.mps"
   ```
3. Run the experiment:
   ```bash
   python main.py
   ```

### Interactive Experimentation

You can also use the `experiment.ipynb` notebook for interactive testing. Open the notebook in Jupyter and follow the steps to load and solve problems or to analyze permuted versions.

## Modules

### `core/logging_handler.py`
Provides a centralized logging mechanism for recording experiment results, errors, and comparison metrics.

### `core/problem_permutator.py`
Handles the generation of permuted versions of MIP problems by randomly shuffling variables while preserving problem structure.

### `core/optimization_experiment.py`
Defines the main class for conducting optimization experiments, comparing the performance of solvers on original and permuted problems.

## Example Workflow

1. **Load a Problem**: The experiment reads a MIP problem from an MPS file.
2. **Solve Original Problem**: Solve the original problem using Gurobi.
3. **Permute Variables**: Generate a new problem with permuted variables.
4. **Solve Permuted Problem**: Solve the permuted problem and compare results with the original.

## Logging and Results

Logs are saved in the `experiments/` directory by default. These logs include:
- Problem details (variables, constraints, objective sense)
- Solver status and objective values for original and permuted problems
- Comparison metrics (absolute and relative differences)

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

