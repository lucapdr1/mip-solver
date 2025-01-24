# main.py

from core.optimization_experiment import OptimizationExperiment

if __name__ == "__main__":
    file_path = "input/example-min.mps"  # Replace with your MPS file path
    
    try:
        experiment = OptimizationExperiment(file_path)
        results = experiment.run_experiment(4)
    except Exception as e:
        print(f"Experiment failed: {e}")
