# main.py

from core.optimization_experiment import OptimizationExperiment
from utils.config import INPUT_PROBLEM

if __name__ == "__main__":    
    try:
        experiment = OptimizationExperiment(INPUT_PROBLEM)
        results = experiment.run_experiment(4)
    except Exception as e:
        print(f"Experiment failed: {e}")
