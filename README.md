# MIP-Solver

`mip-solver` is a Python-based framework designed to facilitate the experimentation and analysis of Mixed Integer Programming (MIP) problems. The framework supports loading MPS files, permuting problem variables, and comparing the performance of solvers on original and permuted problems.



## Pull the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/lucapdr1/mip-solver
cd mip-solver
```
## Requirements

- Python 3.9+
- Gurobi Solver (with a valid license)
- Required Python packages (see [Installation](#installation))

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
- `--threads=8` sets the number of threads withing a job (8 in not specified).
- **Note:** Remember to include the closing `/` at the end of each folder name.

### Execution Using GNU Parallel

You can also run multiple experiments in parallel with different parameters. For example, if you want to test experiments with two or more different granularity values (e.g., 4, 5, 6, 8, 10, 12, 15, 20, 33, and all), use GNU parallel as follows:

```bash
parallel ./runLocalBatchOfJobs.sh --input-dir=./mip_lib/ --output-dir=./batch_output/granularity_{} --parallel-instances=4 --permute-granularity={} --time-limit=3600 ::: 4 5 6 8 10 12 15 20 33 all
```

In this command:
- `{}` is replaced by the granularity value for each job.
- The output directories will be created as, for example, `./batch_output/granularity_5/` and `./batch_output/granularity_all/` respectively.
- The script will process all files in the `./mip_lib/` folder. If you wish to run the experiment on only a subset of instances, create a folder with only those instances and provide that folder as the `--input-dir` parameter.
--------------

## Features

* **Load MIP Problems**: Read problems from MPS files.
* **Structured Permutations**: Apply variable permutations at different granularity levels (global, block-wise, etc.).
* **Rule-Based Reordering**: Reorder variables and constraints using custom-defined rules (e.g. size, sparsity, structure).
* **Solver Integration**: Solve the original and transformed problems using Gurobi (or other solvers).
* **Variability Analysis**: Compute variability matrices to quantify the impact of permutations and reorderings.
* **Result Postprocessing**: Analyze experiment logs to extract summary statistics and evaluate performance changes.

## Modules

### `core/problem_transformation/`

Handles the generation of transformed versions of MIP problems by applying permutations to problem matrices, scaling transformations, and provides helper methods for computing distances between matrices.

### `core/ordering/`

Implements logic and rules for reordering variables and constraints to produce canonical forms of MIP problems.

### `results_analysis/`

Contains all classes and utilities for post-processing experiment logs and computing performance and structural statistics.

## Example Workflow

1. **Load**
   Read a MIP problem from an MPS file.

2. **Permute with Granularity**
   Apply permutations at different levels (e.g. random, block-wise, structure-aware).

3. **Reorder with Rules**
   Sort variables or constraints using defined rules (e.g. number of nonzeros, constraint tightness, block size).

4. **Solve and Analyze**
   Solve the original and transformed problems, and compute variability matrices to capture solver sensitivity.

5. **Postprocess Logs**
   Extract statistics from experiment logs to evaluate structural and performance differences across transformations.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

