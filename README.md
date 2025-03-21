# MIP-Solver

`mip-solver` is a Python-based framework designed to facilitate the experimentation and analysis of Mixed Integer Programming (MIP) problems. The framework supports loading MPS files, permuting problem variables, and comparing the performance of solvers on original and permuted problems.



## Pull the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/lucapdr1/mip-solver
cd mip-solver
```

## Installation

### Option 1: Using Conda

1. **Create the Conda environment**

   Use the provided `environment-py311.yml` file to create a new Conda environment:

   ```bash
   conda env create -f environment-py311.yml
   ```

2. **Activate the environment**

   Activate the new environment (here named `mip311`):

   ```bash
   conda activate mip311
   ```

### Option 2: Using Pip (Python venv)

1. **Create the virtual environment**

   Create a new virtual environment called `mip311`:

   ```bash
   python -m venv mip311
   ```

2. **Activate the environment**

   - On macOS/Linux:

     ```bash
     source mip311/bin/activate
     ```

   - On Windows:

     ```bash
     mip311\Scripts\activate
     ```

3. **Install dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Activate the environment**

   Ensure your environment is activated (either via Conda or the virtual environment).

Below is an updated README section that shows how to run the script both directly and via GNU parallel, along with a note about processing all input instances versus a subset:

---

## 2. Run the Batch Jobs

The script will process **all** files (instances) found in the input folder you specify. If you want to run the job on only a subset of instances, simply create a folder containing only the desired files and pass that folder as the input directory.

### Direct Execution

In the root directory of the project, you can run the script with named parameters. For example:

```bash
bash ./runLocalBatchOfJobs.sh --input-dir=./mip_lib/ --output-dir=./batch_output/ --parallel-instances=4
```

Since the file is executable, you can also omit the `bash` command:

```bash
./runLocalBatchOfJobs.sh --input-dir=./mip_lib/ --output-dir=./batch_output/ --parallel-instances=4
```

- `--input-dir=./mip_lib/` is the folder containing the miplib problems.
- `--output-dir=./batch_output/` is the folder where the results will be saved.
- `--parallel-instances=4` sets the number of parallel jobs within the script (note that 1 is a valid option).
- **Note:** Remember to include the closing `/` at the end of each folder name.

### Execution Using GNU Parallel

You can also run multiple experiments in parallel with different parameters. For example, if you want to test experiments with two or more different granularity values (e.g., 5, 6, 8, 10, 12, 15, 20, 33, and all), use GNU parallel as follows:

```bash
parallel ./runLocalBatchOfJobs.sh --input-dir=./mip_lib/ --output-dir=./batch_output/granularity_{} --parallel-instances=4 --permute-granularity={} --time-limit=3600 ::: 5 6 8 10 12 15 20 33 all
```

In this command:
- `{}` is replaced by the granularity value for each job.
- The output directories will be created as, for example, `./batch_output/granularity_5/` and `./batch_output/granularity_all/` respectively.
- The script will process all files in the `./mip_lib/` folder. If you wish to run the experiment on only a subset of instances, create a folder with only those instances and provide that folder as the `--input-dir` parameter.

--------------

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

