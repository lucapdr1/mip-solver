import os
import sys
import pandas as pd
from log_parser.model import LogMetrics
from log_parser.parsing import parse_model_info, parse_iterations, parse_granularity_stats
from log_parser.aggregation import compute_aggregated_metrics
from log_parser.visualization import plot_aggregated_comparisons

def parse_log_file(file_path: str) -> LogMetrics:
    """Parse a single log file and return a LogMetrics object."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Parse the different sections using our separate data classes
    model_info = parse_model_info(content, os.path.basename(file_path))
    iterations = parse_iterations(content)
    aggregated = compute_aggregated_metrics(iterations)
    granularity_stats = parse_granularity_stats(content)
    
    return LogMetrics(model_info=model_info, iterations=iterations, aggregated_metrics=aggregated, granularity_stats=granularity_stats)

def process_logs_folder(folder_path: str) -> pd.DataFrame:
    logs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.log'):
            file_path = os.path.join(folder_path, file_name)
            try:
                log_metrics = parse_log_file(file_path)
                # Merge model_info and aggregated_metrics
                log_data = {**log_metrics.model_info.__dict__,
                            **log_metrics.aggregated_metrics.__dict__}
                # Include granularity stats if available
                if log_metrics.granularity_stats:
                    log_data.update(log_metrics.granularity_stats.__dict__)
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

     # Create a single image with all comparisons.
    output_image = os.path.join(log_folder, 'aggregated_comparison.png')
    plot_aggregated_comparisons(df, output_file=output_image)
