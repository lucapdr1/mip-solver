import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_aggregated_comparisons(df: pd.DataFrame, output_file: str) -> None:
    """
    Creates a single image with three subplots (one for each metric)
    where each subplot is a grouped bar chart comparing paired metrics
    for each instance. A logarithmic y-scale is applied for better visualization.
    Each subplot displays its own x-axis labels.
    
    The left bar (permutation) is plotted in a neutral color ("lightblue").
    The right bar (canonical) is colored conditionally:
      - Green if the permutation std is higher than the canonical std.
      - Red if the permutation std is lower than the canonical std.
      - Gray if they are equal.
    
    Parameters:
        df (pd.DataFrame): DataFrame with aggregated metrics.
        output_file (str): Path to save the resulting image.
    """
    # Use the "instance" column if available; otherwise use "file_name"
    labels = df['instance'] if 'instance' in df.columns else df['file_name']
    
    num_instances = len(labels)
    x = np.arange(num_instances)  # label locations
    width = 0.35  # width of the bars

    # Create a figure with three subplots (one row per metric), without sharing the x-axis.
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=False)

    # --- Subplot 1: Permutation Distance Variability (Sample STD) ---
    left_values = df['std_perm_distance_before']
    right_values = df['std_perm_distance_after']
    # Compute colors for the canonical bars
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    axs[0].bar(x - width/2, left_values, width, label='Before Canonicalization', color='lightblue')
    axs[0].bar(x + width/2, right_values, width, label='After Canonicalization', color=right_colors)
    axs[0].set_title('Permutation Distance Variability (Sample STD)')
    axs[0].set_ylabel('Standard Deviation')
    axs[0].legend()
    axs[0].set_yscale("log")  # log scale on y-axis
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=45, ha='right')
    
    # --- Subplot 2: Solve Time Variability ---
    left_values = df['std_all_permutation_solve_time']
    right_values = df['std_all_canonical_solve_time']
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    axs[1].bar(x - width/2, left_values, width, label='Permutation', color='lightblue')
    axs[1].bar(x + width/2, right_values, width, label='Canonical', color=right_colors)
    axs[1].set_title('Solve Time Variability')
    axs[1].set_ylabel('Standard Deviation (Solve Time)')
    axs[1].legend()
    axs[1].set_yscale("log")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=45, ha='right')
    
    # --- Subplot 3: Work Units Variability ---
    left_values = df['std_all_permutation_work_units']
    right_values = df['std_all_canonical_work_units']
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    axs[2].bar(x - width/2, left_values, width, label='Permutation', color='lightblue')
    axs[2].bar(x + width/2, right_values, width, label='Canonical', color=right_colors)
    axs[2].set_title('Work Units Variability')
    axs[2].set_ylabel('Standard Deviation (Work Units)')
    axs[2].legend()
    axs[2].set_yscale("log")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Aggregated comparison plot saved to {output_file}")



def plot_granularity_average(df: pd.DataFrame, output_file: str = None) -> None:
    """
    Plots a bar chart showing the average granularity (average items per block) for each instance.
    The y-axis is set to a logarithmic scale and a horizontal line is drawn at the overall median value.
    
    Parameters:
        df (pd.DataFrame): DataFrame that must include an "avg_block_size" column.
        output_file (str, optional): If provided, saves the plot to this file.
                                     Otherwise, displays the plot interactively.
    """
    # Use 'instance' if available; otherwise, use 'file_name' for labels.
    if 'instance' in df.columns:
        labels = df['instance']
    else:
        labels = df['file_name']

    # Check if the granularity column exists.
    if 'avg_block_size' not in df.columns:
        print("Column 'avg_block_size' not found in DataFrame. Cannot plot granularity.")
        return

    avg_values = df['avg_block_size']
    overall_median = avg_values.median()

    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_values, color='skyblue')
    plt.xlabel("Instance")
    plt.ylabel("Average Items per Block (log scale)")
    plt.title("Average Granularity (Items per Block) per Instance")
    plt.xticks(rotation=45, ha='right')

    # Set the y-axis to a logarithmic scale.
    plt.yscale("log")

    # Add a horizontal line at the overall median value.
    plt.axhline(y=overall_median, color='red', linestyle='--', linewidth=2,
                label=f"Overall Median: {overall_median:.2f}")
    plt.legend()

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        plt.close()
        print(f"Granularity average plot saved to {output_file}")
    else:
        plt.show()
