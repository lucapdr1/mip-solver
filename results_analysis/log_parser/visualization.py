# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_aggregated_comparisons(df: pd.DataFrame, output_file: str) -> None:
    """
    Creates a single image with three subplots (one for each metric)
    where each subplot is a grouped bar chart comparing paired metrics
    for each instance.
    
    Parameters:
        df (pd.DataFrame): DataFrame with aggregated metrics.
        output_file (str): Path to save the resulting image.
    """
    # Use the "instance" column if available; otherwise use "file_name"
    if 'instance' in df.columns:
        labels = df['instance']
    else:
        labels = df['file_name']
    
    num_instances = len(labels)
    x = np.arange(num_instances)  # the label locations
    width = 0.35  # the width of the bars

    # Create a figure with three subplots (one row per metric)
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Subplot 1: Permutation Distance Variability (Sample STD)
    axs[0].bar(x - width/2, df['std_perm_distance_before'], width, label='Before Canonicalization')
    axs[0].bar(x + width/2, df['std_perm_distance_after'], width, label='After Canonicalization')
    axs[0].set_title('Permutation Distance Variability (Sample STD)')
    axs[0].set_ylabel('Standard Deviation')
    axs[0].legend()
    
    # Subplot 2: Solve Time Variability
    axs[1].bar(x - width/2, df['std_all_permutation_solve_time'], width, label='Permutation')
    axs[1].bar(x + width/2, df['std_all_canonical_solve_time'], width, label='Canonical')
    axs[1].set_title('Solve Time Variability')
    axs[1].set_ylabel('Standard Deviation (Solve Time)')
    axs[1].legend()
    
    # Subplot 3: Work Units Variability
    axs[2].bar(x - width/2, df['std_all_permutation_work_units'], width, label='Permutation')
    axs[2].bar(x + width/2, df['std_all_canonical_work_units'], width, label='Canonical')
    axs[2].set_title('Work Units Variability')
    axs[2].set_ylabel('Standard Deviation (Work Units)')
    axs[2].legend()

    # Set the x-axis labels for the bottom subplot (all subplots share the x-axis)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Aggregated comparison plot saved to {output_file}")
