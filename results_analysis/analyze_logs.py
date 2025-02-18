import re
import os
import sys
import numpy as np
import pandas as pd

def parse_log_file(file_path):
    """Parse a single log file and extract key metrics"""
    with open(file_path, 'r') as f:
        content = f.read()

    data = {
        'file_name': os.path.basename(file_path),
        'instance': None,
        'variables': None,
        'constraints': None,
        'objective_sense': None,
        # Initialize all possible fields
        'models_equivalent': [],
        'variable_counts_match': [],
        'constraint_counts_match': [],
        'original_objective_values': [],
        'permuted_objective_values': [],
        'canonical_original_objective_values': [],
        'canonical_permuted_objective_values': [],
        'original_solve_times': [],
        'permuted_solve_times': [],
        'canonical_original_solve_times': [],
        'canonical_permuted_solve_times': [],
        'original_work_units': [],
        'permuted_work_units': [],
        'canonical_original_work_units': [],
        'canonical_permuted_work_units': [],
        'permutation_distances_before': [],
        'permutation_distances_after': [],
        # Aggregated fields
        'avg_original_solve_time': None,
        'std_original_solve_time': None,
        'avg_permuted_solve_time': None,
        'std_permuted_solve_time': None,
        'avg_canonical_original_solve_time': None,
        'std_canonical_original_solve_time': None,
        'avg_canonical_permuted_solve_time': None,
        'std_canonical_permuted_solve_time': None,
        'avg_original_work_units': None,
        'std_original_work_units': None,
        'avg_permuted_work_units': None,
        'std_permuted_work_units': None,
        'perm_distance_before_avg': None,
        'perm_distance_before_std': None,
        'perm_distance_after_avg': None,
        'perm_distance_after_std': None,
        'models_equivalent_pct': None,
        'variable_counts_match_pct': None,
        'constraint_counts_match_pct': None
    }

    # Extract problem characteristics
    instance_match = re.search(r'Successfully loaded problem from S3: .*/(.*?)\.mps', content)
    if instance_match:
        data['instance'] = instance_match.group(1)
    
    prob_details = re.search(
        r'Problem Details:.*?- Variables: (\d+).*?- Constraints: (\d+).*?- Objective Sense: (\w+)',
        content, re.DOTALL
    )
    if prob_details:
        data['variables'] = int(prob_details.group(1))
        data['constraints'] = int(prob_details.group(2))
        data['objective_sense'] = prob_details.group(3)

    # Improved regex pattern with better decimal handling
    iterations = re.finditer(
        r'Running iteration (\d+)/\d+.*?'
        r'- Models equivalent: (True|False).*?'
        r'- Variable counts match: (True|False).*?'
        r'- Constraint counts match: (True|False).*?'
        r'- Original Objective Value: ([-+]?[\d.]+).*?'
        r'- Permuted Objective Value: ([-+]?[\d.]+).*?'
        r'- Canonical from Original Objective Value: ([-+]?[\d.]+).*?'
        r'- Canonical from Permuted Objective Value: ([-+]?[\d.]+).*?'
        r'- Original Solve Time: ([\d.]+) seconds.*?'
        r'- Permuted Solve Time: ([\d.]+) seconds.*?'
        r'- Canonical from Original Solve Time: ([\d.]+) seconds.*?'
        r'- Canonical from Permuted Solve Time: ([\d.]+) seconds.*?'
        r'- Original Work Units: ([\d.]+).*?'
        r'- Permuted Work Units: ([\d.]+).*?'
        r'- Canonical from Original Work Units: ([\d.]+).*?'
        r'- Canonical from Permuted Work Units: ([\d.]+).*?'
        r'- Permutation Distance Before Canonicalization: ([\d.]+).*?'
        r'- Permutation Distance After Canonicalization: ([\d.]+)',
        content, re.DOTALL
    )

    for match in iterations:
        # Extract all 18 captured groups
        data['models_equivalent'].append(match.group(2) == 'True')
        data['variable_counts_match'].append(match.group(3) == 'True')
        data['constraint_counts_match'].append(match.group(4) == 'True')
        
        # Convert all numeric values properly
        data['original_objective_values'].append(float(match.group(5)))
        data['permuted_objective_values'].append(float(match.group(6)))
        data['canonical_original_objective_values'].append(float(match.group(7)))
        data['canonical_permuted_objective_values'].append(float(match.group(8)))
        
        data['original_solve_times'].append(float(match.group(9)))
        data['permuted_solve_times'].append(float(match.group(10)))
        data['canonical_original_solve_times'].append(float(match.group(11)))
        data['canonical_permuted_solve_times'].append(float(match.group(12)))
        
        data['original_work_units'].append(float(match.group(13)))
        data['permuted_work_units'].append(float(match.group(14)))
        data['canonical_original_work_units'].append(float(match.group(15)))
        data['canonical_permuted_work_units'].append(float(match.group(16)))
        
        data['permutation_distances_before'].append(float(match.group(17)))
        data['permutation_distances_after'].append(float(match.group(18)))

    # Calculate aggregated metrics only if we have data
    if data['permuted_solve_times']:
        try:
            # Solve Time Metrics
            data['avg_original_solve_time'] = np.mean(data['original_solve_times'])
            data['std_original_solve_time'] = np.std(data['original_solve_times'])
            data['avg_permuted_solve_time'] = np.mean(data['permuted_solve_times'])
            data['std_permuted_solve_time'] = np.std(data['permuted_solve_times'])
            data['avg_canonical_original_solve_time'] = np.mean(data['canonical_original_solve_times'])
            data['std_canonical_original_solve_time'] = np.std(data['canonical_original_solve_times'])
            data['avg_canonical_permuted_solve_time'] = np.mean(data['canonical_permuted_solve_times'])
            data['std_canonical_permuted_solve_time'] = np.std(data['canonical_permuted_solve_times'])
            
            # Work Unit Metrics
            data['avg_original_work_units'] = np.mean(data['original_work_units'])
            data['std_original_work_units'] = np.std(data['original_work_units'])
            data['avg_permuted_work_units'] = np.mean(data['permuted_work_units'])
            data['std_permuted_work_units'] = np.std(data['permuted_work_units'])
            
            # Permutation Distance Metrics
            data['perm_distance_before_avg'] = np.mean(data['permutation_distances_before'])
            data['perm_distance_before_std'] = np.std(data['permutation_distances_before'])
            data['perm_distance_after_avg'] = np.mean(data['permutation_distances_after'])
            data['perm_distance_after_std'] = np.std(data['permutation_distances_after'])
            
            # Model Consistency Metrics
            data['models_equivalent_pct'] = np.mean(data['models_equivalent']) * 100
            data['variable_counts_match_pct'] = np.mean(data['variable_counts_match']) * 100
            data['constraint_counts_match_pct'] = np.mean(data['constraint_counts_match']) * 100
        except Exception as e:
            print(f"Error calculating metrics for {data['file_name']}: {str(e)}")

    return data

def process_logs_folder(folder_path):
    """Process all log files in a folder"""
    all_data = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.log'):
            file_path = os.path.join(folder_path, file_name)
            try:
                file_data = parse_log_file(file_path)
                all_data.append(file_data)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    
    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_logs.py <log_folder>")
        sys.exit(1)
    
    log_folder = sys.argv[1]
    df = process_logs_folder(log_folder)
    
    # Save results to CSV
    output_file = os.path.join(log_folder, 'analysis_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")

    # Generate summary statistics
    summary = df.describe(include='all')
    print("\nSummary Statistics:")
    print(summary)