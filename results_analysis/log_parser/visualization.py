import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams['text.usetex'] = False

def plot_aggregated_comparisons(df: pd.DataFrame, output_file: str) -> None:
    """
    Creates a single image with six subplots:
      - Subplots 1-3: Grouped bar charts for the variability metrics:
           1. Permutation Distance Variability (std before vs. after canonicalization)
           2. Solve Time Variability (permutation vs. canonical)
           3. Work Units Variability (permutation vs. canonical)
         (y-axis uses a logarithmic scale for these metrics)
      - Subplots 4-6: Single bar charts for the reduction percentage metrics:
           4. Permutation Distance Reduction (%)
           5. Solve Time Reduction (%)
           6. Work Units Reduction (%)
         (y-axis is linear since reductions may be negative or positive)
    
    For the grouped bar charts, the canonical bar is colored conditionally:
      - Green if the permutation value is higher than the canonical value (improvement)
      - Red if lower
      - Gray if equal.
    
    In subplots 4-6, a horizontal dashed line is drawn at the average reduction.
    
    For each subplot, the function computes the percentage of green bars and annotates that percentage.
    
    Parameters:
        df (pd.DataFrame): DataFrame with aggregated metrics. Expected columns include:
            'std_perm_distance_before', 'std_perm_distance_after',
            'std_all_permutation_solve_time', 'std_all_canonical_solve_time',
            'std_all_permutation_work_units', 'std_all_canonical_work_units',
            'std_perm_distance_reduction_pct', 'std_solve_time_reduction_pct',
            'std_work_units_reduction_pct'
            plus an 'instance' or 'file_name' column for labeling.
        output_file (str): Path to save the resulting image.
    """
    # Use the "instance" column if available; otherwise use "file_name"
    labels = df['instance'] if 'instance' in df.columns else df['file_name']
    num_instances = len(labels)
    x = np.arange(num_instances)  # label locations
    width = 0.35  # width of the grouped bars
    bar_width = 0.6  # width for single bar charts

    # Create figure with six subplots (vertical layout)
    fig, axs = plt.subplots(6, 1, figsize=(12, 36), sharex=False)

    # --- Subplot 1: Permutation Distance Variability (Sample STD) ---
    left_values = df['std_perm_distance_before'].abs()
    right_values = df['std_perm_distance_after'].abs()
    right_colors = [
        'green' if l > r else 'red' if l < r else 'gray'
        for l, r in zip(left_values, right_values)
    ]
    axs[0].bar(x - width/2, left_values, width, label='Before Canonicalization', color='lightblue')
    axs[0].bar(x + width/2, right_values, width, label='After Canonicalization', color=right_colors)
    axs[0].set_title('Permutation Distance Variability (Sample STD)')
    axs[0].set_ylabel('Standard Deviation')
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for color in right_colors if color == 'green')
    percentage_green = (num_green / len(right_colors)) * 100
    axs[0].text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=axs[0].transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    
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
    axs[1].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for color in right_colors if color == 'green')
    percentage_green = (num_green / len(right_colors)) * 100
    axs[1].text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=axs[1].transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    
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
    axs[2].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for color in right_colors if color == 'green')
    percentage_green = (num_green / len(right_colors)) * 100
    axs[2].text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=axs[2].transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    
    # --- Subplot 4: Permutation Distance Ratio (After/Before) ---
    reduction_values = df['std_perm_distance_reduction_pct']

    # Compute ratio: if reduction is not None and >= 0 then ratio = 1 - (x/100), otherwise set ratio to 1.
    ratio_values = reduction_values.apply(lambda x: 1 - x/100 if (x is not None) else 1)

    # Color coding: now, all positive (or zero) reductions are green, negatives are red.
    ratio_colors = [
        'green' if (x is not None and x >= 0) else 'red'
        for x in reduction_values
    ]

    axs[3].bar(x, ratio_values, bar_width, color=ratio_colors)
    # Compute geometric mean (clip to avoid log(0))
    geom_mean = np.exp(np.mean(np.log(ratio_values.clip(lower=1e-10))))
    axs[3].axhline(geom_mean, color='black', linestyle='--', label=f'Geom Mean: {geom_mean:.3f}')
    axs[3].set_title('Permutation Distance Ratio (After/Before)')
    axs[3].set_ylabel('Ratio (After/Before)')
    axs[3].set_xticks(x)
    axs[3].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')

    # Count how many original reduction values are positive (or zero)
    num_green = sum(1 for x in reduction_values if x is not None and x >= 0)
    percentage_green = (num_green / len(reduction_values)) * 100
    axs[3].text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=axs[3].transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    axs[3].legend()


    # --- Subplot 5: Solve Time Ratio (After/Before) ---
    reduction_values = df['std_solve_time_reduction_pct']
    ratio_values = reduction_values.apply(lambda x: 1 - x/100 if (x is not None) else 1)
    ratio_colors = [
        'green' if (x is not None and x >= 0) else 'red'
        for x in reduction_values
    ]
    axs[4].bar(x, ratio_values, bar_width, color=ratio_colors)
    geom_mean = np.exp(np.mean(np.log(ratio_values.clip(lower=1e-10))))
    axs[4].axhline(geom_mean, color='black', linestyle='--', label=f'Geom Mean: {geom_mean:.3f}')
    axs[4].set_title('Solve Time Ratio (After/Before)')
    axs[4].set_ylabel('Ratio (After/Before)')
    axs[4].set_xticks(x)
    axs[4].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for x in reduction_values if x is not None and x >= 0)
    percentage_green = (num_green / len(reduction_values)) * 100
    axs[4].text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=axs[4].transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    axs[4].legend()


    # --- Subplot 6: Work Units Ratio (After/Before) ---
    reduction_values = df['std_work_units_reduction_pct']
    ratio_values = reduction_values.apply(lambda x: 1 - x/100 if (x is not None) else 1)
    ratio_colors = [
        'green' if (x is not None and x >= 0) else 'red'
        for x in reduction_values
    ]
    axs[5].bar(x, ratio_values, bar_width, color=ratio_colors)
    geom_mean = np.exp(np.mean(np.log(ratio_values.clip(lower=1e-10))))
    axs[5].axhline(geom_mean, color='black', linestyle='--', label=f'Geom Mean: {geom_mean:.3f}')
    axs[5].set_title('Work Units Ratio (After/Before)')
    axs[5].set_ylabel('Ratio (After/Before)')
    axs[5].set_xticks(x)
    axs[5].set_xticklabels([str(label) for label in labels], rotation=45, ha='right')
    num_green = sum(1 for x in reduction_values if x is not None and x >= 0)
    percentage_green = (num_green / len(reduction_values)) * 100
    axs[5].text(0.95, 0.95, f"Green: {percentage_green:.1f}%", transform=axs[5].transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    axs[5].legend()


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
