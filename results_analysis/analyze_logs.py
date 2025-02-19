import os
import sys
import pandas as pd
from log_parser.model import LogMetrics
from log_parser.parsing import parse_model_info, parse_iterations
from log_parser.aggregation import compute_aggregated_metrics

def parse_log_file(file_path: str) -> LogMetrics:
    """Parse a single log file and return a LogMetrics object."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Parse the different sections using our separate data classes
    model_info = parse_model_info(content, os.path.basename(file_path))
    iterations = parse_iterations(content)
    aggregated = compute_aggregated_metrics(iterations)
    
    return LogMetrics(model_info=model_info, iterations=iterations, aggregated_metrics=aggregated)

def process_logs_folder(folder_path: str) -> pd.DataFrame:
    """Process all log files in a folder and return a DataFrame of results."""
    logs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.log'):
            file_path = os.path.join(folder_path, file_name)
            try:
                log_metrics = parse_log_file(file_path)
                # Merge our data classes into a single dictionary
                log_data = {**log_metrics.model_info.__dict__,
                            **log_metrics.aggregated_metrics.__dict__}
                logs.append(log_data)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    return pd.DataFrame(logs)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_logs.py <log_folder>")
        sys.exit(1)
    
    log_folder = sys.argv[1]
    df = process_logs_folder(log_folder)
    
    # Save the aggregated results to CSV
    output_file = os.path.join(log_folder, 'analysis_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")

    # Print summary statistics
    summary = df.describe(include='all')
    print("\nSummary Statistics:")
    print(summary)
