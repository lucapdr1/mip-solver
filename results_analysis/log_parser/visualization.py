import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams['text.usetex'] = False

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_perm_distance_variability(ax, df, labels, x, width):
    """
    Plots Permutation Distance Variability as a grouped bar chart.
    """
    left_values = df['std_perm_distance_before'].abs()
    right_values = df['std_perm_distance_after'].abs()
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    ax.bar(x - width/2, left_values, width, label='Before Canonicalization', color='lightblue')
    ax.bar(x + width/2, right_values, width, label='After Canonicalization', color=right_colors)
    ax.set_title('Permutation Distance Variability (Sample STD)')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for color in right_colors if color == 'green')
    percentage_green = (num_green / len(right_colors)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    
def plot_pairwise_distance_variability(ax, df, labels, x, width):
    """
    Plots Permutation Distance Variability as a grouped bar chart.
    """
    if 'std_pairwise_distance_before' not in df.columns:
        print("Column 'std_pairwise_distance_before' not found in DataFrame. Cannot plot granularity.")
        return
    
    left_values = df['std_pairwise_distance_before'].abs()
    right_values = df['std_pairwise_distance_after'].abs()
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    ax.bar(x - width/2, left_values, width, label='Before Canonicalization', color='lightblue')
    ax.bar(x + width/2, right_values, width, label='After Canonicalization', color=right_colors)
    ax.set_title('All-Pairs Permutation Distance Variability (Sample STD)')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for color in right_colors if color == 'green')
    percentage_green = (num_green / len(right_colors)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

def plot_solve_time_variability(ax, df, labels, x, width):
    """
    Plots Solve Time Variability as a grouped bar chart.
    """
    left_values = df['std_all_permutation_solve_time']
    right_values = df['std_all_canonical_solve_time']
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    ax.bar(x - width/2, left_values, width, label='Permutation', color='lightblue')
    ax.bar(x + width/2, right_values, width, label='Canonical', color=right_colors)
    ax.set_title('Solve Time Variability')
    ax.set_ylabel('Standard Deviation (Solve Time)')
    ax.legend()
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for color in right_colors if color == 'green')
    percentage_green = (num_green / len(right_colors)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

def plot_work_units_variability(ax, df, labels, x, width):
    """
    Plots Work Units Variability as a grouped bar chart.
    """
    left_values = df['std_all_permutation_work_units']
    right_values = df['std_all_canonical_work_units']
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    ax.bar(x - width/2, left_values, width, label='Permutation', color='lightblue')
    ax.bar(x + width/2, right_values, width, label='Canonical', color=right_colors)
    ax.set_title('Work Units Variability')
    ax.set_ylabel('Standard Deviation (Work Units)')
    ax.legend()
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for color in right_colors if color == 'green')
    percentage_green = (num_green / len(right_colors)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

def plot_perm_distance_ratio(ax, df, labels, x, bar_width):
    """
    Plots Permutation Distance Ratio as a single bar chart.
    """
    reduction_values = df['std_perm_distance_reduction_pct']
    ratio_values = reduction_values.apply(lambda x: 1 - x/100 if x is not None else 1)
    ratio_colors = ['green' if (x is not None and x >= 0) else 'red' for x in reduction_values]
    ax.bar(x, ratio_values, bar_width, color=ratio_colors)
    # Compute geometric mean (avoid log(0) by clipping)
    geom_mean = np.exp(np.mean(np.log(ratio_values.clip(lower=1e-10))))
    ax.axhline(geom_mean, color='black', linestyle='--', label=f'Geom Mean: {geom_mean:.3f}')
    ax.set_title('Permutation Distance Ratio (After/Before)')
    ax.set_ylabel('Ratio (After/Before)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for x in reduction_values if x is not None and x >= 0)
    percentage_green = (num_green / len(reduction_values)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()

def plot_pairwise_distance_ratio(ax, df, labels, x, bar_width):
    """
    Plots Permutation Distance Ratio as a single bar chart.
    """
    if 'std_pairwise_distance_reduction_pct' not in df.columns:
        print("Column 'std_pairwise_distance_reduction_pct' not found in DataFrame. Cannot plot granularity.")
        return
    
    reduction_values = df['std_pairwise_distance_reduction_pct']
    ratio_values = reduction_values.apply(lambda x: 1 - x/100 if x is not None else 1)
    ratio_colors = ['green' if (x is not None and x >= 0) else 'red' for x in reduction_values]
    ax.bar(x, ratio_values, bar_width, color=ratio_colors)
    # Compute geometric mean (avoid log(0) by clipping)
    geom_mean = np.exp(np.mean(np.log(ratio_values.clip(lower=1e-10))))
    ax.axhline(geom_mean, color='black', linestyle='--', label=f'Geom Mean: {geom_mean:.3f}')
    ax.set_title('All-Pairs Permutation Distance Ratio (After/Before)')
    ax.set_ylabel('Ratio (After/Before)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for x in reduction_values if x is not None and x >= 0)
    percentage_green = (num_green / len(reduction_values)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()

def plot_solve_time_ratio(ax, df, labels, x, bar_width):
    """
    Plots Solve Time Ratio as a single bar chart.
    """
    reduction_values = df['std_solve_time_reduction_pct']
    ratio_values = reduction_values.apply(lambda x: 1 - x/100 if x is not None else 1)
    ratio_colors = ['green' if (x is not None and x >= 0) else 'red' for x in reduction_values]
    ax.bar(x, ratio_values, bar_width, color=ratio_colors)
    geom_mean = np.exp(np.mean(np.log(ratio_values.clip(lower=1e-10))))
    ax.axhline(geom_mean, color='black', linestyle='--', label=f'Geom Mean: {geom_mean:.3f}')
    ax.set_title('Solve Time Ratio (After/Before)')
    ax.set_ylabel('Ratio (After/Before)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for x in reduction_values if x is not None and x >= 0)
    percentage_green = (num_green / len(reduction_values)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()

def plot_work_units_ratio(ax, df, labels, x, bar_width):
    """
    Plots Work Units Ratio as a single bar chart.
    """
    reduction_values = df['std_work_units_reduction_pct']
    ratio_values = reduction_values.apply(lambda x: 1 - x/100 if x is not None else 1)
    ratio_colors = ['green' if (x is not None and x >= 0) else 'red' for x in reduction_values]
    ax.bar(x, ratio_values, bar_width, color=ratio_colors)
    geom_mean = np.exp(np.mean(np.log(ratio_values.clip(lower=1e-10))))
    ax.axhline(geom_mean, color='black', linestyle='--', label=f'Geom Mean: {geom_mean:.3f}')
    ax.set_title('Work Units Ratio (After/Before)')
    ax.set_ylabel('Ratio (After/Before)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for x in reduction_values if x is not None and x >= 0)
    percentage_green = (num_green / len(reduction_values)) * 100
    ax.text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()


def plot_aggregated_comparisons(df: pd.DataFrame, output_file: str) -> None:
    """
    Aggregates selected comparison graphs into one image.
    
    Parameters:
        df (pd.DataFrame): DataFrame with aggregated metrics.
        output_file (str): Path to save the resulting image.
    """
    # Use the "instance" column if available; otherwise use "file_name"
    labels = df['instance'] if 'instance' in df.columns else df['file_name']
    num_instances = len(labels)
    x = np.arange(num_instances)  # label locations
    width = 0.35   # width for grouped bar charts
    bar_width = 0.6  # width for single bar charts

    # Build a list of plots to include (in desired order)
    plots = []
    
    # Pair comparisons
    plots.append(plot_perm_distance_variability)
    plots.append(plot_pairwise_distance_variability)
    plots.append(plot_solve_time_variability)
    plots.append(plot_work_units_variability)
    
    # Ratios
    plots.append(plot_perm_distance_ratio)
    plots.append(plot_pairwise_distance_ratio)
    plots.append(plot_solve_time_ratio)
    plots.append(plot_work_units_ratio)

    num_plots = len(plots)
    if num_plots == 0:
        print("No graphs selected for plotting.")
        return

    # Create a figure with one subplot per selected graph
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), squeeze=False)
    
    # Iterate over the selected plot functions and call each with its axis.
    for i, plot_func in enumerate(plots):
        # Determine width to pass: grouped bar charts use 'width', ratio charts use 'bar_width'
        # We check the function name as a simple heuristic.
        plot_width = width if 'variability' in plot_func.__name__ else bar_width
        ax = axs[i, 0]
        plot_func(ax, df, labels, x, plot_width)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Aggregated comparison plot saved to {output_file}")


def plot_granularity_combined(df: pd.DataFrame, output_file: str = None) -> None:
    """
    Creates a single image with two vertically-arranged subplots:
      1. The top subplot displays the average granularity (average items per block)
         for each instance using a logarithmic y-scale and a horizontal line at the overall median.
      2. The bottom subplot displays the percentage of total variables that are, on average,
         contained in a block for each instance, with a horizontal line at the overall median percentage.
         
    Both plots use the 'instance' column for labels if available, otherwise 'file_name'.
    
    The x-axis tick labels are shown on both subplots and rotated for clarity.
    
    Parameters:
        df (pd.DataFrame): DataFrame that must include 'avg_block_size' for the top plot,
                           and both 'avg_block_size' and 'variables' (and 'constraints')
                           for the bottom plot.
        output_file (str, optional): If provided, saves the plot to this file;
                                     otherwise, displays the plot interactively.
    """
    # Use 'instance' if available; otherwise, use 'file_name'
    labels = df['instance'] if 'instance' in df.columns else df['file_name']
    x_positions = np.arange(len(labels))
    
    # Create a new figure with two subplots (vertical layout), without sharing the x-axis.
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
    
    # ----- Top Subplot: Average Granularity (Items per Block) -----
    if 'avg_block_size' not in df.columns:
        print("Column 'avg_block_size' not found in DataFrame. Cannot plot granularity.")
        return
    avg_values = df['avg_block_size']
    overall_median = avg_values.median()
    
    axs[0].bar(x_positions, avg_values, color='skyblue')
    axs[0].set_ylabel("Average Items per Block\n(log scale)")
    axs[0].set_title("Average Granularity per Instance")
    axs[0].set_yscale("log")
    axs[0].axhline(y=overall_median, color='red', linestyle='--', linewidth=2,
                   label=f"Overall Median: {overall_median:.2f}")
    axs[0].legend()
    axs[0].set_xticks(x_positions)
    axs[0].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    
    # ----- Bottom Subplot: Average Block Percentage -----
    if 'variables' not in df.columns or 'constraints' not in df.columns:
        print("Required columns ('variables' and 'constraints') not found in DataFrame. Cannot plot block percentage.")
        return
    # Compute percentage = (avg_block_size / (variables * constraints)) * 100
    df = df.copy()
    #df['avg_block_percentage'] = (df['avg_block_size'] / (df['variables'] * df['constraints'])) * 100
    overall_median_percentage = df['avg_block_percentage'].median()
    
    axs[1].bar(x_positions, df['avg_block_percentage'], color='skyblue')
    axs[1].set_xlabel("Instance")
    axs[1].set_ylabel("Average Block Percentage (%)")
    axs[1].set_title("Percentage of Total Variables per Block (Average)")
    axs[1].axhline(y=overall_median_percentage, color='red', linestyle='--', linewidth=2,
                   label=f"Overall Median: {overall_median_percentage:.2f}%")
    axs[1].legend()
    axs[1].set_xticks(x_positions)
    axs[1].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    
    plt.tight_layout()
    # Adjust bottom margin to prevent overlap (if needed)
    plt.subplots_adjust(bottom=0.2)
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
        print(f"Combined granularity plot saved to {output_file}")
    else:
        plt.show()
