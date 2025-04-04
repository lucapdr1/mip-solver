# compare_results.py

import os
import sys
import pandas as pd
import numpy as np
from log_parser.compare_results_visualization import plot_all_trends

def load_csvs_separately(folder_path: str) -> dict:
    """Load all CSV files (with '_analysis_results' in their name) from a folder into separate DataFrames."""
    dataframes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and '_analysis_results' in file_name:
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                key = os.path.splitext(file_name)[0]  # remove .csv extension
                dataframes[key] = df
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    if not dataframes:
        raise ValueError("No analysis CSV files found in folder.")
    return dataframes

def compute_geom_mean(series: pd.Series) -> float:
    """Compute the geometric mean for a series of ratio values, clipping to avoid log(0)."""
    clipped = series.clip(lower=1e-10)
    return np.exp(np.mean(np.log(clipped)))

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute summary metrics for one CSV.
    Assumes that the CSV has columns:
      - 'std_perm_distance_reduction_pct'
      - 'std_pairwise_distance_reduction_pct' (optional)
      - 'std_solve_time_reduction_pct'
      - 'std_work_units_reduction_pct'
    The ratios are computed as: ratio = 1 - (pct/100)
    """
    metrics = {}
    # Permutation Distance Ratio
    if 'std_perm_distance_reduction_pct' in df.columns:
        perm_ratios = df['std_perm_distance_reduction_pct'].apply(lambda x: 1 - x/100 if pd.notnull(x) else 1)
        metrics['geom_perm_ratio'] = compute_geom_mean(perm_ratios)
    else:
        metrics['geom_perm_ratio'] = np.nan

    # Pairwise Distance Ratio (if available)
    if 'std_pairwise_distance_reduction_pct' in df.columns:
        pairwise_ratios = df['std_pairwise_distance_reduction_pct'].apply(lambda x: 1 - x/100 if pd.notnull(x) else 1)
        metrics['geom_pairwise_ratio'] = compute_geom_mean(pairwise_ratios)
    else:
        metrics['geom_pairwise_ratio'] = np.nan

    # Solve Time Ratio
    if 'std_solve_time_reduction_pct' in df.columns:
        solve_ratios = df['std_solve_time_reduction_pct'].apply(lambda x: 1 - x/100 if pd.notnull(x) else 1)
        metrics['geom_solve_ratio'] = compute_geom_mean(solve_ratios)
    else:
        metrics['geom_solve_ratio'] = np.nan

    # Work Units Ratio
    if 'std_work_units_reduction_pct' in df.columns:
        work_ratios = df['std_work_units_reduction_pct'].apply(lambda x: 1 - x/100 if pd.notnull(x) else 1)
        metrics['geom_work_units_ratio'] = compute_geom_mean(work_ratios)
    else:
        metrics['geom_work_units_ratio'] = np.nan

    return metrics

def extract_parameter(file_key: str) -> float:
    """
    Extracts the parameter from the file key.
    For filenames like "granularity_1000_analysis_results", it extracts 1000.
    For filenames starting with "granularity_all", it returns 10000 to indicate a value above 3000.
    """
    try:
        parts = file_key.split('_')
        if file_key.startswith("granularity_all"):
            return 10000.0
        elif file_key.startswith("granularity") and len(parts) >= 2:
            # e.g., "granularity_1000_analysis_results" -> parts[1] is "1000"
            return float(parts[1])
        elif "param" in parts:
            idx = parts.index("param") + 1
            return float(parts[idx])
        else:
            # fallback: iterate through parts and return the first numeric token
            for token in parts:
                try:
                    return float(token)
                except ValueError:
                    continue
            return float('nan')
    except Exception as e:
        print(f"Could not extract parameter from {file_key}: {e}")
        return float('nan')

def aggregate_all_metrics(dataframes: dict) -> pd.DataFrame:
    """Compute metrics for each CSV and return a summary DataFrame."""
    rows = []
    for key, df in dataframes.items():
        metrics = compute_metrics(df)
        param = extract_parameter(key)
        # If filename indicates granularity_all, use a special label.
        if key.startswith("granularity_all"):
            param_label = "all"
        else:
            param_label = str(param)
        metrics['parameter'] = param      # numeric value used for sorting
        metrics['parameter_label'] = param_label  # label to display on x-axis
        metrics['source'] = key
        rows.append(metrics)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_results.py <results_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    print(f"Loading CSV files from: {folder_path}")
    dfs = load_csvs_separately(folder_path)
    print(f"Loaded {len(dfs)} CSV files.")

    # Compute aggregated metrics for each CSV.
    summary_df = aggregate_all_metrics(dfs)
    print("Aggregated Metrics:")
    print(summary_df)

    # Save the aggregated metrics to a CSV for reference.
    summary_csv = os.path.join(folder_path, "aggregated_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Aggregated metrics saved to {summary_csv}")

    # Create one single image with all trend plots.
    trends_output = os.path.join(folder_path, "all_trends.png")
    plot_all_trends(summary_df, trends_output)
