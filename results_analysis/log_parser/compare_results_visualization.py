# visualization.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_trend(ax, df: pd.DataFrame, parameter: str, parameter_label: str, metric: str, ylabel: str, title: str):
    """
    Plots a single trend line for a given metric vs parameter.
    The x-axis is treated as discrete categories with evenly spaced points.
    """
    # Drop rows with missing parameter or metric values, then sort by numeric parameter.
    plot_df = df.dropna(subset=[parameter, metric]).sort_values(parameter)
    # Create equally spaced x positions.
    x_positions = range(len(plot_df))
    
    # Plot the metric values at these x positions.
    ax.plot(x_positions, plot_df[metric], marker='o', linestyle='-')
    # Set the x-axis ticks to be the parameter labels.
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(plot_df[parameter_label].astype(str))
    
    ax.set_xlabel("Parameter Setting")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

def plot_all_trends(summary_df: pd.DataFrame, output_file: str):
    """
    Creates a single figure with multiple subplots, one for each trend.
    The summary_df must contain 'parameter' (numeric for sorting) and 'parameter_label' (for display).
    """
    # Define the configurations for each subplot.
    plots = [
        {
            'metric': 'geom_perm_ratio',
            'ylabel': 'Geom Mean (Permutation Ratio)',
            'title': 'Permutation Distance Ratio Trend'
        },
        {
            'metric': 'geom_pairwise_ratio',
            'ylabel': 'Geom Mean (Pairwise Ratio)',
            'title': 'Pairwise Distance Ratio Trend'
        },
        {
            'metric': 'geom_solve_ratio',
            'ylabel': 'Geom Mean (Solve Time Ratio)',
            'title': 'Solve Time Ratio Trend'
        },
        {
            'metric': 'geom_work_units_ratio',
            'ylabel': 'Geom Mean (Work Units Ratio)',
            'title': 'Work Units Ratio Trend'
        }
    ]
    
    num_plots = len(plots)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), squeeze=False)
    
    for i, config in enumerate(plots):
        metric = config['metric']
        if metric not in summary_df.columns:
            continue  # Skip if this metric is not present
        ax = axs[i, 0]
        plot_trend(
            ax,
            summary_df,
            parameter='parameter',
            parameter_label='parameter_label',
            metric=metric,
            ylabel=config['ylabel'],
            title=config['title']
        )
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"All trends plot saved to {output_file}")
