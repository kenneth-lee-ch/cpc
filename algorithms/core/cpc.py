"""
CPC Algorithm Implementation.

This module implements the CPC algorithm.
The algorithm uses conditional independence tests to infer causal relationships between variables.
"""

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.FCI import reorientAllWith, getPossibleDsep
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
import warnings
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from algorithms.core.orientation import FCI_orientations, rule0, kPC_orientations, make_kess_graph
from causallearn.graph.GraphClass import CausalGraph
from itertools import combinations
from causallearn.utils.PCUtils.Helper import append_value
from collections import deque
from algorithms.utils.cit import *
from tqdm.auto import tqdm
import numpy as np
from math import log2
from collections import defaultdict
from sklearn.model_selection import KFold
import itertools
import pandas as pd

def find_conditioning_sets_by_sample_size(
    df: pd.DataFrame,
    max_cond_size: int,
    min_samples: int = 50
) -> List[Tuple[int, ...]]:
    """
    Find conditioning sets that satisfy minimum sample size requirements.
    
    Args:
        df: Input DataFrame
        max_cond_size: Maximum size of conditioning sets to consider
        min_samples: Minimum number of samples required in each group
        
    Returns:
        List of conditioning set indices that satisfy the sample size requirement
    """
    candidate_sets = []
    columns = list(df.columns)
    n = len(columns)
    
    # Consider conditioning sets of sizes 0 to n-2.
    for r in range(1, max_cond_size+1):  
        for subset in itertools.combinations(columns, r):
            if len(columns) - len(subset) < 2:
                continue
            groups = [df] if len(subset) == 0 else [group for _, group in df.groupby(list(subset))]
            if all(len(group) > min_samples for group in groups):
                subset_indices = tuple(columns.index(col) for col in subset)
                candidate_sets.append(subset_indices)
    return candidate_sets


def find_reliable_conditioning_sets_by_contingency(df,  max_cond_size, min_cell_count=5):
    """
    Returns all candidate conditioning sets Z (as tuples of column names)
    such that, for more than half of the pairs (X, Y) among the remaining variables,
    when grouping by Z every cell in the X-Y contingency table has count > min_cell_count.
    """
    reliable_sets = []
    columns = list(df.columns)
    n = len(columns)
    
    for r in range(1,  max_cond_size+1):
        for subset in itertools.combinations(columns, r):
            remaining = [col for col in columns if col not in subset]
            if len(remaining) < 2:
                continue

            reliable_pair_count = 0
            total_pairs = 0

            for X, Y in itertools.combinations(remaining, 2):
                total_pairs += 1
                levels_X = sorted(df[X].unique())
                levels_Y = sorted(df[Y].unique())
                pair_is_reliable = True

                groups = [df] if len(subset) == 0 else [group for _, group in df.groupby(list(subset))]
                for group in groups:
                    table = pd.crosstab(group[X], group[Y])
                    table = table.reindex(index=levels_X, columns=levels_Y, fill_value=0)
                    if not (table > min_cell_count).values.all():
                        pair_is_reliable = False
                        break
                if pair_is_reliable:
                    reliable_pair_count += 1
                    
            if total_pairs > 0 and reliable_pair_count > (total_pairs / 2):
                subset_indices = tuple(columns.index(col) for col in subset)
                reliable_sets.append(subset_indices)
    return reliable_sets


def is_data_continuous(data: np.ndarray) -> bool:
    """
    Check if the input data is continuous.
    
    Args:
    data (np.ndarray): The input data array.
    
    Returns:
    bool: True if the data is likely continuous, False otherwise.
    """
    # Check if data is float type
    if not np.issubdtype(data.dtype, np.floating):
        return False
    
    return True

def all_subsets_except(elements, x, y, max_size=None):
    """
    Generator function that yields all subsets of 'elements', excluding those containing x and y.
    
    Args:
    elements (list): The list of elements to generate subsets from.
    x: Element to exclude from subsets.
    y: Element to exclude from subsets.
    max_size (int, optional): Maximum size of subsets to generate. If None, all sizes are considered.
    
    Yields:
    set: A subset of elements, not containing x or y.
    """
    # Remove x and y from the elements list if they're present
    elements = [e for e in elements if e != x and e != y]
    
    # Determine the range of subset sizes
    max_size = len(elements) if max_size is None else min(max_size, len(elements))
    
    # Generate subsets of each size from 0 to max_size
    for size in range(max_size + 1):
        for subset in combinations(elements, size):
            yield set(subset)

def compute_probabilities(array, col_indices, sample_size=None):
    """
    Compute joint probabilities for the given columns in the NumPy array using sampling if specified.
    """
    if sample_size is not None:
        sampled_indices = np.random.choice(array.shape[0], size=sample_size, replace=False)
        array_sampled = array[sampled_indices, :]
    else:
        array_sampled = array

    # Select the columns specified by col_indices
    selected_columns = array_sampled[:, col_indices]

    # Get unique rows and their counts
    unique_rows, counts = np.unique(selected_columns, axis=0, return_counts=True)
    
    # Compute total count
    total_count = array_sampled.shape[0]
    
    # Compute probabilities
    probabilities = {tuple(row): count / total_count for row, count in zip(unique_rows, counts)}
    
    return probabilities

def mutual_information(array, X, Y, sample_size=None):
    """
    Compute the Mutual Information I(X; Y) for discrete data.
    
    Parameters:
    array - NumPy array containing the data
    X - index of the variable X in the array
    Y - index of the variable Y in the array
    sample_size - optional parameter to specify the sample size for approximation
    
    Returns:
    mi - the estimated mutual information I(X; Y)
    """
    # Compute joint probabilities with optional sampling
    prob_xy = compute_probabilities(array, [X, Y], sample_size)
    prob_x = compute_probabilities(array, [X], sample_size)
    prob_y = compute_probabilities(array, [Y], sample_size)
    
    # Compute mutual information
    mi = 0.0
    for (x, y), p_xy in prob_xy.items():
        p_x = prob_x.get((x,), 1e-10)
        p_y = prob_y.get((y,), 1e-10)
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * log2(p_xy / (p_x * p_y))
    
    return mi

def conditional_mutual_information(array, X, Y, Z=None, sample_size=None):
    """
    Compute the Conditional Mutual Information I(X; Y | Z) for discrete data.
    
    Parameters:
    array - NumPy array containing the data
    X - index of the variable X in the array
    Y - index of the variable Y in the array
    Z - list of indices for the conditioning set Z, can be None or empty
    sample_size - optional parameter to specify the sample size for approximation
    
    Returns:
    cmi - the estimated conditional mutual information I(X; Y | Z)
    """
    if Z is None or len(Z) == 0:
        return mutual_information(array, X, Y, sample_size)
    
    # Compute joint probabilities with optional sampling
    prob_xyz = compute_probabilities(array, [X, Y] + Z, sample_size)
    prob_xz = compute_probabilities(array, [X] + Z, sample_size)
    prob_yz = compute_probabilities(array, [Y] + Z, sample_size)
    prob_z = compute_probabilities(array, Z, sample_size)
    
    # Compute conditional mutual information
    cmi = 0.0
    for (x, y, *z), p_xyz in prob_xyz.items():
        p_xz = prob_xz.get((x, *z), 1e-99)
        p_yz = prob_yz.get((y, *z), 1e-99)
        p_z = prob_z.get(tuple(z), 1e-99)
        
        if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
            cmi += p_xyz * log2(p_xyz * p_z / (p_xz * p_yz))
    
    return cmi



def compute_cmi_variance(data: np.ndarray, num_folds: int, k: int, q: int, idx_to_skip=[]) -> List[frozenset]:
    """
    Compute the variance of conditional mutual information across j folds for different conditioning sets.

    Args:
    data (np.ndarray): The input data array.
    num_folds (int): Number of folds to split the data into.
    k (int): Maximum size of the conditioning set Z.
    q (int): Number of top conditioning sets to return.

    Returns:
    List[frozenset]: Top q conditioning sets, sorted by their CMI variance (ascending).
    """
    n_samples, n_vars = data.shape

    # Generate all possible conditioning sets Z
    all_Z = []
    valid_vars = [i for i in range(n_vars) if i not in idx_to_skip]
    for size in range(1, k + 1):
        all_Z.extend(combinations(valid_vars , size))
    
    # Dictionary to store CMI values for each fold and conditioning set
    cmi_values = defaultdict(list)
    z_variance = defaultdict(list)
    # Create KFold object
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Split data into j folds and compute CMI for each fold
    for train_index, _ in kf.split(data):
        fold_data = data[train_index]

        # Compute CMI for each pair X, Y and conditioning set Z
        for Z in all_Z:
            for X, Y in combinations(set(range(n_vars)) - set(Z), 2):
                cmi = conditional_mutual_information(fold_data, X, Y, list(Z))        
                cmi_values[Z].append(cmi)
            #z_variance[Z].append(np.var(cmi_values[Z]))

    # Compute variance for each conditioning set
    #variances = [(frozenset(Z), np.sum(z_variance[Z])) for Z in all_Z]
    variances = [(frozenset(Z), np.var(cmi_values[Z])) for Z in all_Z]

    # Sort by variance and return top q results (only the frozensets)
    return [set(Z) for Z, _ in sorted(variances, key=lambda x: x[1])[:q]]


def find_all_path_nodes(adj_matrix, start, end):
    def bfs(start_node):
        # Initialize BFS structures
        queue = deque([start_node])
        visited = set([start_node])
        reachability = set([start_node])

        while queue:
            current = queue.popleft()
            # Explore all adjacent nodes
            for neighbor, is_connected in enumerate(adj_matrix[current]):
                if is_connected != 0 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    reachability.add(neighbor)
        return reachability

    # Perform BFS from both start and end nodes
    reach_from_start = bfs(start)
    reach_from_end = bfs(end)

    # The intersection of both reachability sets contains all nodes that are on any path between start and end
    all_path_nodes = reach_from_start.intersection(reach_from_end)
    all_path_nodes = all_path_nodes - set([start])
    all_path_nodes = all_path_nodes - set([end])
    return all_path_nodes

def update_dictionary(d, t, s):
    # Check if the tuple 't' is a key in the dictionary 'd'
    if t not in d:
        # If not, add 't' as a key with an empty list as its value
        d[t] = []
    # Append the set 's' directly to the list associated with the key 't'
    d[t].append(s)  # Appending the set as a whole to the list


def orient(g, nodes, sep_sets, data, alpha, n, independence_test_method, background_knowledge, verbose):
    """
    Orient the graph using the FCI orientation and kPC orientation rules
    """
    node_names = [node.get_name() for node in nodes]
    rule0(graph=g, nodes = nodes, sep_sets=sep_sets, ambiguous_triple = [], knowledge=background_knowledge, verbose=verbose)
    G, edges = FCI_orientations(g, data, sep_sets, nodes, 
                independence_test_method,
                background_knowledge,ambiguous_triple = [],
                alpha = alpha, verbose=verbose)
    # G.graph means adjacency set
    adj=G.graph
    new_adj=kPC_orientations(G,n)
    while (new_adj!=adj).any():
        adj=new_adj
        D , _ = make_kess_graph(new_adj,n, data_name = node_names, nodes=nodes)
        new_adj = kPC_orientations(D,n)
    D, _ =make_kess_graph(new_adj,n, data_names = node_names, nodes=nodes)
    return D,new_adj



def heuristic_search(df, cc_set_selection_method = 'chi-sq', num_folds=10, cond_set_size_upper_bound=1, num_top_sets=10, idx_to_skip=[]):
    """ 
        num_folds: the number of folds to split the data into
        cond_set_size_upper_bound: the upper bound of the size of the conditioning set
        num_top_sets: the number of top conditioning sets to return
        idx_to_skip: the indices of the variables to skip
    """
    if cc_set_selection_method == 'MI':
        # ranking each conditioning set by the variance of the k-fold CMI
        # this assumes discrete data
        data = df.to_numpy()
        Z = compute_cmi_variance(data, num_folds, cond_set_size_upper_bound, num_top_sets, idx_to_skip) 
    elif cc_set_selection_method == 'samplesize>50':
        # sample size > 50
        Z =  find_conditioning_sets_by_sample_size(df, cond_set_size_upper_bound, min_samples=50)
    else:
        # chi-sq
        Z = find_reliable_conditioning_sets_by_contingency(df, cond_set_size_upper_bound, min_cell_count=5)
    Z = [set()] + Z # add back the empty set
    return Z


def CPC(df,tester,I,variables_cannot_be_tested=[], data_names =[],alpha=0.05, 
        path_sep_set_enforced=False,
        traverse_subset_and_skip = False,
        k=1,
        background_knowledge: BackgroundKnowledge | None = None,
        verbose=False, **kwargs):
    data = df.to_numpy()
    n = data.shape[1]
    def is_any_element_in(list1, list2):
        # Convert lists to sets
        set1 = set(list1)
        set2 = set(list2)

        # Check if there is any common element
        return not set1.isdisjoint(set2)  # Returns True if there's at least one common element

    def remove_if_exists(causalg:CausalGraph, x: int, y: int) -> None:
        edge = causalg.G.get_edge(causalg.G.nodes[x], causalg.G.nodes[y])
        if edge is not None:
            causalg.G.remove_edge(edge)

    # start from a complete graph
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    independence_test_method = CIT(data, method=tester, **kwargs)

    ## ------- check parameters ------------
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    ## ------- end check parameters ------------

    # create the node variables
    nodes = []
    idx_to_skip = []
    strname_id_map ={}
    node_to_id_map = {}
    for i in range(data.shape[1]):
        if data_names:
            node = GraphNode(data_names[i])
            strname_id_map[data_names[i]] = i
            if data_names[i] in variables_cannot_be_tested:
                idx_to_skip.append(i)
        else:
            node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)
        node_to_id_map[node] = i 


    # create an empty adjacency sets
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}

    node_names = [node.get_name() for node in nodes]
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(independence_test_method)
    # perform marginal CI tests in a fast adjacency search manner
    var_range = range(no_of_var)
    for x in var_range:
        Neigh_x = cg.neighbors(x)
        for y in Neigh_x:
            # looping through subsets
            if path_sep_set_enforced:
                node_a = cg.G.nodes[x]
                node_b = cg.G.nodes[y]
                
                # check if every element is in the possibleDsep list
                possibleDsep = getPossibleDsep(node_a, node_b, cg.G, -1)
                id_of_possibleDsep = [node_to_id_map[dsepnode] for dsepnode in possibleDsep]
                # get the nodes along any path between x and y 
                path_nodes_id = find_all_path_nodes(cg.G.graph, x, y)
                # get possibleDesep and all nodes along the paths
                possbleDsepIntersectwithPath_nodes_id = path_nodes_id.intersection(id_of_possibleDsep)
                # we need at least one member of S to be in this set
                for sepset in I:
                    if data_names:
                        sepset = set([strname_id_map[s] for s in sepset])
                    if x in sepset or y in sepset:
                        continue
                    isAtLeastOneElementinPossDsep = is_any_element_in(sepset, possbleDsepIntersectwithPath_nodes_id)
                    if isAtLeastOneElementinPossDsep:
                        p = cg.ci_test(x, y, sepset)
                        if p > alpha:
                            remove_if_exists(cg, x, y)
                            remove_if_exists(cg, y, x)
                            append_value(cg.sepset, x, y, tuple(sepset))
                            append_value(cg.sepset, y, x, tuple(sepset))
                            sep_sets[(x, y)] = sepset
                            sep_sets[(y, x)] = sepset
                            break
            else:
                if traverse_subset_and_skip:
                    found = False
                    new_s = [s for s in var_range if x != s and y!= s and s not in idx_to_skip]
                    for r in range(k):
                        for sepset in combinations(new_s, r):                     
                            p = cg.ci_test(x, y, sepset)
                            if p > alpha:
                                remove_if_exists(cg, x, y)
                                remove_if_exists(cg, y, x)
                                append_value(cg.sepset, x, y, tuple(sepset))
                                append_value(cg.sepset, y, x, tuple(sepset))
                                sep_sets[(x, y)] = sepset
                                sep_sets[(y, x)] = sepset
                                found = True
                                break
                        if found:
                            break
                else:
                    for sepset in I:
                        if data_names:
                            sepset = set([strname_id_map[s] for s in sepset])
                        if x in sepset or y in sepset:
                            continue
                        p = cg.ci_test(x, y, sepset)
                        if p > alpha:
                            remove_if_exists(cg, x, y)
                            remove_if_exists(cg, y, x)
                            append_value(cg.sepset, x, y, tuple(sepset))
                            append_value(cg.sepset, y, x, tuple(sepset))
                            sep_sets[(x, y)] = sepset
                            sep_sets[(y, x)] = sepset
                            break


    ########### apply FCI and kPC orientations #################
    g = cg.G 
    # apply orientation rules in the end if it were not applied at each iteration
    D, new_adj = orient(g, nodes, sep_sets, data, alpha, n, independence_test_method, background_knowledge, verbose)
    return D, new_adj
