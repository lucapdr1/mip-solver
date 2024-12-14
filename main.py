# main.py

from core.optimization_experiment import OptimizationExperiment

if __name__ == "__main__":
    file_path = "input/example3.mps"  # Replace with your MPS file path
    
    try:
        experiment = OptimizationExperiment(file_path)
        results = experiment.run_experiment()
    except Exception as e:
        print(f"Experiment failed: {e}")
