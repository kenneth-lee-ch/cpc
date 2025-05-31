import os
import numpy as np
import matplotlib.pyplot as plt
from algorithms.utils.graph_utils import fscore_calculator_skeleton, fscore_calculator_arrowhead, fscore_calculator_tail

def load_and_analyze_results(dir_name):
    # Initialize results dictionary
    results = {
        'n_values': [],
        'cpc_skel_avg': [], 'cpc_skel_std': [],
        'cpc_arrow_avg': [], 'cpc_arrow_std': [],
        'cpc_tail_avg': [], 'cpc_tail_std': [],
        'sat_skel_avg': [], 'sat_skel_std': [],
        'sat_arrow_avg': [], 'sat_arrow_std': [],
        'sat_tail_avg': [], 'sat_tail_std': []
    }
    
    # Get all unique n values from the saved files
    matrices_dir = os.path.join(dir_name, 'matrices')
    n_values = set()
    for filename in os.listdir(matrices_dir):
        if filename.endswith('.npz'):
            n = int(filename.split('_')[0][1:])  # Extract n from filename like 'n10_iter0.npz'
            n_values.add(n)
    
    n_values = sorted(list(n_values))
    
    for n in n_values:
        # Initialize lists for this n
        cpc_skel_scores = []
        cpc_arrow_scores = []
        cpc_tail_scores = []
        sat_skel_scores = []
        sat_arrow_scores = []
        sat_tail_scores = []
        
        # Find all files for this n
        files_for_n = [f for f in os.listdir(matrices_dir) if f.startswith(f'n{n}_')]
        
        for filename in files_for_n:
            data = np.load(os.path.join(matrices_dir, filename))
            ground_truth = data['ground_truth']
            cpc_adj = data['cpc_adj']
            sat_adj = data['sat_adj']
            
            # Calculate scores
            cpc_skel_scores.append(fscore_calculator_skeleton(cpc_adj, ground_truth))
            cpc_arrow_scores.append(fscore_calculator_arrowhead(cpc_adj, ground_truth))
            cpc_tail_scores.append(fscore_calculator_tail(cpc_adj, ground_truth))
            
            sat_skel_scores.append(fscore_calculator_skeleton(sat_adj, ground_truth))
            sat_arrow_scores.append(fscore_calculator_arrowhead(sat_adj, ground_truth))
            sat_tail_scores.append(fscore_calculator_tail(sat_adj, ground_truth))
        
        # Calculate averages and standard errors
        N = len(files_for_n)
        results['n_values'].append(n)
        
        # CPC scores
        results['cpc_skel_avg'].append(np.mean(cpc_skel_scores))
        results['cpc_skel_std'].append(np.std(cpc_skel_scores) / np.sqrt(N))
        results['cpc_arrow_avg'].append(np.mean(cpc_arrow_scores))
        results['cpc_arrow_std'].append(np.std(cpc_arrow_scores) / np.sqrt(N))
        results['cpc_tail_avg'].append(np.mean(cpc_tail_scores))
        results['cpc_tail_std'].append(np.std(cpc_tail_scores) / np.sqrt(N))
        
        # SAT scores
        results['sat_skel_avg'].append(np.mean(sat_skel_scores))
        results['sat_skel_std'].append(np.std(sat_skel_scores) / np.sqrt(N))
        results['sat_arrow_avg'].append(np.mean(sat_arrow_scores))
        results['sat_arrow_std'].append(np.std(sat_arrow_scores) / np.sqrt(N))
        results['sat_tail_avg'].append(np.mean(sat_tail_scores))
        results['sat_tail_std'].append(np.std(sat_tail_scores) / np.sqrt(N))
    
    return results

def plot_results(results):
    # F1-Skeleton plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(results['n_values'], results['cpc_skel_avg'], yerr=results['cpc_skel_std'], 
                 fmt='-o', label='CPC', capsize=5, color='blue')
    plt.errorbar(results['n_values'], results['sat_skel_avg'], yerr=results['sat_skel_std'], 
                 fmt='-s', label='SAT', capsize=5, color='red')
    plt.xlabel('Number of Variables')
    plt.ylabel('F1-Skeleton Score')
    plt.title('F1-Skeleton Score vs Number of Variables')
    plt.legend()
    plt.savefig('f1_skeleton_vs_n.pdf', bbox_inches='tight')
    plt.close()

    # F1-Arrowhead plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(results['n_values'], results['cpc_arrow_avg'], yerr=results['cpc_arrow_std'], 
                 fmt='-o', label='CPC', capsize=5, color='blue')
    plt.errorbar(results['n_values'], results['sat_arrow_avg'], yerr=results['sat_arrow_std'], 
                 fmt='-s', label='SAT', capsize=5, color='red')
    plt.xlabel('Number of Variables')
    plt.ylabel('F1-Arrowhead Score')
    plt.title('F1-Arrowhead Score vs Number of Variables')
    plt.legend()
    plt.savefig('f1_arrowhead_vs_n.pdf', bbox_inches='tight')
    plt.close()

    # F1-Tail plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(results['n_values'], results['cpc_tail_avg'], yerr=results['cpc_tail_std'], 
                 fmt='-o', label='CPC', capsize=5, color='blue')
    plt.errorbar(results['n_values'], results['sat_tail_avg'], yerr=results['sat_tail_std'], 
                 fmt='-s', label='SAT', capsize=5, color='red')
    plt.xlabel('Number of Variables')
    plt.ylabel('F1-Tail Score')
    plt.title('F1-Tail Score vs Number of Variables')
    plt.legend()
    plt.savefig('f1_tail_vs_n.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Specify the directory containing the saved matrices
    dir_name = 'set1'  # Change this to match your directory name
    
    # Load and analyze results
    results = load_and_analyze_results(dir_name)
    
    # Plot the results
    plot_results(results)
    
    print("Analysis complete. Plots have been saved as PDF files.") 