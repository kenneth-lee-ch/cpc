import numpy as np
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import chisq, fisherz, kci
from causallearn.graph.GraphNode import GraphNode
from algorithms.core.fci import fci_k
import warnings
from algorithms.utils.cit import *
from typing import List, Dict, Tuple, Set
from algorithms.core.skeleton import learnSkeletonUpToK
from algorithms.core.orientation import FCI_orientations, rule0, kPC_orientations, make_kess_graph

def kPC(data,tester,k,n,alpha=0.05,  
        fastAdjSearch=False, 
        printCI = False,
        background_knowledge: BackgroundKnowledge | None = None,
        node_names = [],
        verbose=False, **kwargs):
    
    # start from a complete graph
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    independence_test_method = CIT(data, method=tester, **kwargs)

    ## ------- check parameters ------------
    if (k is None) or type(k) != int:
        raise TypeError("'k' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    ## ------- end check parameters ------------

    # create the node variables
    nodes = []
    for i in range(data.shape[1]):
        if node_names:
            node = GraphNode(node_names[i])
        else:
            node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    # create an empty adjacency sets
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}

   
    if fastAdjSearch:
        # apply fast adjacency search but do not mark any essential edges
        G, edges = fci_k(data, tester, alpha, depth=k, verbose=verbose)
        adj=G.graph
        new_adj=kPC_orientations(G,n)
        while (new_adj!=adj).any():
            adj=new_adj
            D, _ =make_kess_graph(new_adj,n)
            new_adj = kPC_orientations(D,n)
        D, _ =make_kess_graph(new_adj,n)
    else:
        # no fast adj search and no essential edge
        graph , sep_sets = learnSkeletonUpToK(data, nodes, k, sep_sets,graph_given = None, 
                                                     independence_test_method= independence_test_method, alpha=alpha, printCI = printCI)
        rule0(graph=graph, nodes = nodes, sep_sets=sep_sets, ambiguous_triple = [], knowledge=background_knowledge, verbose=verbose)
        G, edges = FCI_orientations(graph, data, sep_sets, nodes, 
                     independence_test_method,
                     background_knowledge, ambiguous_triple = [],
                     alpha = alpha, verbose=verbose)
        adj=G.graph
        new_adj=kPC_orientations(G,n)
        while (new_adj!=adj).any():
            adj=new_adj
            D, _ =make_kess_graph(new_adj,n)
            new_adj = kPC_orientations(D,n)

        D, _ =make_kess_graph(new_adj,n, data_names=node_names)
    return D,new_adj