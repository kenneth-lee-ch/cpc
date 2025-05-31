from __future__ import annotations
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.graph.Edges import Edges
import itertools
from algorithms.utils.cit import *
from copy import deepcopy
from tqdm.auto import tqdm
from causallearn.graph.Node import Node
from numpy import ndarray
from typing import List, Set, Tuple, Dict
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.graph.GraphClass import CausalGraph


def possible_parents(node_x: Node, adjx: List[Node], knowledge: BackgroundKnowledge | None = None) -> List[Node]:
    possible_parents: List[Node] = []

    for node_z in adjx:
        if (knowledge is None) or \
                (not knowledge.is_forbidden(node_z, node_x) and not knowledge.is_required(node_x, node_z)):
            possible_parents.append(node_z)

    return possible_parents


def freeDegree(nodes: List[Node], adjacencies) -> int:
    max_degree = 0
    for node_x in nodes:
        opposites = adjacencies[node_x]
        for node_y in opposites:
            adjx = set(opposites)
            adjx.remove(node_y)

            if len(adjx) > max_degree:
                max_degree = len(adjx)
    return max_degree

def forbiddenEdge(node_x: Node, node_y: Node, knowledge: BackgroundKnowledge | None) -> bool:
    if knowledge is None:
        return False
    elif knowledge.is_forbidden(node_x, node_y) and knowledge.is_forbidden(node_y, node_x):
        print(node_x.get_name() + " --- " + node_y.get_name() +
              " because it was forbidden by background background_knowledge.")
        return True
    return False


def searchAtDepth0(data: ndarray, nodes: List[Node], adjacencies: Dict[Node, Set[Node]],
                   sep_sets: Dict[Tuple[int, int], Set[int]],
                   independence_test_method: CIT | None=None, alpha: float = 0.05,
                   verbose: bool = False, knowledge: BackgroundKnowledge | None = None, pbar=None) -> bool:
    empty = []

    show_progress = pbar is not None
    if show_progress:
        pbar.reset()
    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
            pbar.set_description(f'Depth=0, working on node {i}')
        if verbose and (i + 1) % 100 == 0:
            print(nodes[i + 1].get_name())

        for j in range(i + 1, len(nodes)):
            p_value = independence_test_method(i, j, tuple(empty))
            independent = p_value > alpha
            no_edge_required = True if knowledge is None else \
                ((not knowledge.is_required(nodes[i], nodes[j])) or knowledge.is_required(nodes[j], nodes[i]))
            if independent and no_edge_required:
                sep_sets[(i, j)] = set()

                if verbose:
                    print(nodes[i].get_name() + " _||_ " + nodes[j].get_name() + " | (),  score = " + str(p_value))
            elif not forbiddenEdge(nodes[i], nodes[j], knowledge):
                adjacencies[nodes[i]].add(nodes[j])
                adjacencies[nodes[j]].add(nodes[i])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > 0


def searchAtDepth(data: ndarray, depth: int, nodes: List[Node], adjacencies: Dict[Node, Set[Node]],
                  sep_sets: Dict[Tuple[int, int], Set[int]],
                  independence_test_method: CIT | None = None,
                  alpha: float = 0.05,
                  essential_edges: BackgroundKnowledge | None = None,
                  verbose: bool = False, knowledge: BackgroundKnowledge | None = None, pbar=None) -> bool:
    def edge(adjx: List[Node], i: int, adjacencies_completed_edge: Dict[Node, Set[Node]]) -> bool:
        for j in range(len(adjx)):
            node_y = adjx[j]
            _adjx = list(adjacencies_completed_edge[nodes[i]])
            _adjx.remove(node_y)

            ppx = possible_parents(nodes[i], _adjx, knowledge)

            if len(ppx) >= depth:
                cg = ChoiceGenerator(len(ppx), depth)
                choice = cg.next()
                flag = False
                while choice is not None:
                    cond_set = [nodes.index(ppx[index]) for index in choice]
                    choice = cg.next()

                    Y = nodes.index(adjx[j])
                    p_value = independence_test_method(i, Y, tuple(cond_set))
                    independent = p_value > alpha

                    no_edge_required = True if knowledge is None else \
                            not (knowledge.is_required(nodes[i], adjx[j]) or knowledge.is_required(adjx[j],
                                                                                                  nodes[i]))
                    
                    # if any direction of the edge is essential, we don't want to remove that edge 
                    no_essential_edges = True if essential_edges is None else \
                        not (essential_edges.is_required(nodes[i], adjx[j]) or essential_edges.is_required(adjx[j], nodes[i]))

                    if independent and no_edge_required and no_essential_edges:

                        if adjacencies[nodes[i]].__contains__(adjx[j]):
                            adjacencies[nodes[i]].remove(adjx[j])
                        if adjacencies[adjx[j]].__contains__(nodes[i]):
                            adjacencies[adjx[j]].remove(nodes[i])

                        if cond_set is not None:
                            if sep_sets.keys().__contains__((i, nodes.index(adjx[j]))):
                                sep_set = sep_sets[(i, nodes.index(adjx[j]))]
                                for cond_set_item in cond_set:
                                    sep_set.add(cond_set_item)
                            else:
                                sep_sets[(i, nodes.index(adjx[j]))] = set(cond_set)

                        if verbose:
                            message = "Independence accepted: " + nodes[i].get_name() + " _||_ " + adjx[
                                j].get_name() + " | "
                            for cond_set_index in range(len(cond_set)):
                                message += nodes[cond_set[cond_set_index]].get_name()
                                if cond_set_index != len(cond_set) - 1:
                                    message += ", "
                            message += "\tp = " + str(p_value)
                            print(message)
                        flag = True
                if flag:
                    return False
        return True

    count = 0

    adjacencies_completed = deepcopy(adjacencies)

    show_progress = pbar is not None
    if show_progress:
        pbar.reset()

    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
            pbar.set_description(f'Depth={depth}, working on node {i}')
        if verbose:
            count += 1
            if count % 10 == 0:
                print("count " + str(count) + " of " + str(len(nodes)))
        adjx = list(adjacencies[nodes[i]])
        finish_flag = False
        while not finish_flag:
            finish_flag = edge(adjx, i, adjacencies_completed)
            adjx = list(adjacencies[nodes[i]])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > depth


def searchAtDepth_not_stable(data: ndarray, depth: int, nodes: List[Node], adjacencies: Dict[Node, Set[Node]],
                             sep_sets: Dict[Tuple[int, int], Set[int]],
                             independence_test_method: CIT | None=None, alpha: float = 0.05, essential_edges: BackgroundKnowledge | None = None,
                             verbose: bool = False,
                             knowledge: BackgroundKnowledge | None = None,
                             pbar=None) -> bool:
    def edge(adjx, i, adjacencies_completed_edge):
        for j in range(len(adjx)):
            node_y = adjx[j]
            _adjx = list(adjacencies_completed_edge[nodes[i]])
            _adjx.remove(node_y)
            ppx = possible_parents(nodes[i], _adjx, knowledge)

            if len(ppx) >= depth:
                cg = ChoiceGenerator(len(ppx), depth)
                choice = cg.next()

                while choice is not None:
                    cond_set = [nodes.index(ppx[index]) for index in choice]
                    choice = cg.next()

                    Y = nodes.index(adjx[j])
                    p_value = independence_test_method(i, Y, tuple(cond_set))
                    independent = p_value > alpha

                    no_edge_required = True if knowledge is None else \
                        not (knowledge.is_required(nodes[i], adjx[j]) or knowledge.is_required(adjx[j], nodes[i]))
                    
                    # if any direction of the edge is essential, we don't want to remove that edge 
                    no_essential_edges = True if essential_edges is None else \
                        not (essential_edges.is_required(nodes[i], adjx[j]) or essential_edges.is_required(adjx[j], nodes[i]))


                    if independent and no_edge_required and no_essential_edges:

                        if adjacencies[nodes[i]].__contains__(adjx[j]):
                            adjacencies[nodes[i]].remove(adjx[j])
                        if adjacencies[adjx[j]].__contains__(nodes[i]):
                            adjacencies[adjx[j]].remove(nodes[i])

                        if cond_set is not None:
                            if sep_sets.keys().__contains__((i, nodes.index(adjx[j]))):
                                sep_set = sep_sets[(i, nodes.index(adjx[j]))]
                                for cond_set_item in cond_set:
                                    sep_set.add(cond_set_item)
                            else:
                                sep_sets[(i, nodes.index(adjx[j]))] = set(cond_set)

                        if verbose:
                            message = "Independence accepted: " + nodes[i].get_name() + " _||_ " + adjx[
                                j].get_name() + " | "
                            for cond_set_index in range(len(cond_set)):
                                message += nodes[cond_set[cond_set_index]].get_name()
                                if cond_set_index != len(cond_set) - 1:
                                    message += ", "
                            message += "\tp = " + str(p_value)
                            print(message)
                        return False
        return True

    count = 0

    show_progress = pbar is not None
    if show_progress:
        pbar.reset()

    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
            pbar.set_description(f'Depth={depth}, working on node {i}')
        if verbose:
            count += 1
            if count % 10 == 0:
                print("count " + str(count) + " of " + str(len(nodes)))
        adjx = list(adjacencies[nodes[i]])
        finish_flag = False
        while not finish_flag:
            finish_flag = edge(adjx, i, adjacencies)

            adjx = list(adjacencies[nodes[i]])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > depth


def getPowerSet(s,size):
    for i in itertools.combinations(s, size):
        yield(set(i))



def learnSkeletonViafastAdjSearch_k(data: ndarray, nodes: List[Node], adjacencies = Dict, sep_sets = Dict, essential_edges: BackgroundKnowledge | None = None, 
                  independence_test_method: CIT | None=None, alpha: float = 0.05,
        knowledge: BackgroundKnowledge | None = None, depth: int = -1,
        verbose: bool = False, stable: bool = True,
                    show_progress: bool = True) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]]]:
    """
        running learnSkeletonViafastAdjSearch by one order at a time
    """

    # --------check parameter -----------
    if (depth is not None) and type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (knowledge is not None) and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")

    # --------end check parameter -----------

    # ------- initial variable -----------
    pbar = tqdm(total=len(nodes)) if show_progress else None
    if depth==0:
        _ = searchAtDepth0(data, nodes, adjacencies, sep_sets, independence_test_method, alpha, verbose,
                                knowledge, pbar=pbar)
    else:
        if stable:
            _ = searchAtDepth(data, depth, nodes, adjacencies, sep_sets, independence_test_method, alpha,essential_edges, verbose, knowledge, pbar=pbar)
        else:
            _ = searchAtDepth_not_stable(data, depth, nodes, adjacencies, sep_sets, independence_test_method, alpha, essential_edges,
                                        verbose, knowledge, pbar=pbar)
    if show_progress:
        pbar.close()

    print("Finishing Fast Adjacency Search.")
    return adjacencies, sep_sets




def learnSkeletonUpToK(data, nodes, depth, sep_sets, graph_given = None, 
                  independence_test_method: CIT | None=None, alpha: float = 0.05, 
                    printCI = False):
    def remove_if_exists(causalg:CausalGraph, x: int, y: int) -> None:
        edge = causalg.G.get_edge(causalg.G.nodes[x], causalg.G.nodes[y])
        if edge is not None:
            causalg.G.remove_edge(edge)

    if depth > len(nodes) - 2:
        print("The specified conditioning set size is larger than all possible conditioning set size! ")
    
    # create a complete graph
    if graph_given is None:
        graph = GeneralGraph(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_x = nodes[i]
                node_y = nodes[j]
                graph.add_edge(Edges().undirected_edge(node_x, node_y))
    else:
        graph = graph_given


    ls_var_idx = [i for i in range(len(nodes))]

    node_names = [node.get_name() for node in nodes]
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(independence_test_method)
    var_range = range(no_of_var)
    for x in var_range:
        Neigh_x = cg.neighbors(x)
        for y in Neigh_x:
            # looping through subsets
            new_s = [s for s in var_range if x != s and y!= s]
            found = False
            for r in range(depth+1):
                for sepsets in itertools.combinations(new_s, r):
                    p = cg.ci_test(x, y, sepsets)
                    if printCI:
                        print("x:{}, y:{}, p-val{}".format(node_names[x], node_names[y], p))
                    if p > alpha:
                        remove_if_exists(cg, x, y)
                        remove_if_exists(cg, y, x)
                        append_value(cg.sepset, x, y, tuple(sepsets))
                        append_value(cg.sepset, y, x, tuple(sepsets))
                        sep_sets[(x, y)] = sepsets
                        sep_sets[(y, x)] = sepsets
                        found = True
                        break
                if found:
                    break
    return cg.G, sep_sets



def create_unique_filename(base_filename):
    filename, extension = os.path.splitext(base_filename)
    counter = 1
    # Check if the file exists and increment the counter until an available name is found
    unique_filename = f"{filename}{extension}"
    while os.path.exists(unique_filename):
        unique_filename = f"{filename}_{counter}{extension}"
        counter += 1
    return unique_filename




