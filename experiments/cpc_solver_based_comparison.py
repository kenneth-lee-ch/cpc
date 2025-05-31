import os
import sys
import time

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

import pydot
from IPython.display import Image, display

sys.path.append("")
import unittest

import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.cit import chisq, fisherz, kci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.image as gumimage
from algorithms.core.kpc import kPC
from algorithms.core.cpc import CPC, CPC_modified
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from algorithms.utils.graph_utils import fscore_calculator_skeleton, fscore_calculator_arrowhead, fscore_calculator_tail
from algorithms.utils.graph_utils import visualize_graph
from algorithms.utils.graph_utils import create_CPT
from pyAgrum.lib.bn2graph import BN2dot
import itertools
import random
from typing import List, Dict, Tuple, Set
from algorithms.utils.cit import *
from causallearn.search.ConstraintBased.PC import pc
import csv
from algorithms.core.SAT import SATCausalDiscovery


def randomBNwithSpecificStates(nodes:int,arcs:int, states: List[int], p:float, graphidx, seed):
    table = {}
    high_state_list_name = []
    fixseed = seed + graphidx
    gum.initRandom(fixseed)
    g=gum.BNGenerator()
    tmp=g.generate(nodes,arcs,2)
    bn=gum.BayesNet()
    # Nodes
    v=list(tmp.names())
    random.shuffle(v)
    # h=len(v)//2
    for name in v:
        #np.random.seed(fixseed)
        s = np.random.choice(a=np.array(states), size=1, p=p)
        state_num = s[0]
        print("The number of states assigned is:{}".format(state_num))
        bn.add(name, int(state_num))
        id = bn.ids([name])
        if state_num > 2:
            high_state_list_name.append(id[0])
        table[id[0]] = state_num
    # for name in v[:h]:
    #     bn.add(name,mod1)
    # for name in v[h:]:
    #     bn.add(name,mod2)
    
    # arcs
    bn.beginTopologyTransformation()
    for a,b in tmp.arcs():
        bn.addArc(tmp.variable(a).name(),tmp.variable(b).name())
    bn.endTopologyTransformation()
    bn.generateCPTs()
    # output_dict = {value: key for key, value in table.items()}
    return bn, table, high_state_list_name

def refillCPT_Dirichlet(bn,node_names,domain,option='Dirichlet'):
    no_of_states_dict = {}
    for i in node_names:
        no_of_states_dict[i] = bn.variable(i).domainSize()

    for var_name in node_names:
        create_CPT(bn,var_name,no_of_states_dict,option)

setID = 1 
for w in range(setID, 99999):
    dir_name = 'set'+ str(w) 
    if os.path.exists(dir_name):
        continue
    else:
        os.mkdir(dir_name)
        break

# setID=1
# dir_name = 'set'+str(setID)
# os.mkdir(dir_name)
n = 30
ratio_arc=2

domain=2 # states for each node. 


num_edges = [60]
states = [2, 30]
p_states = [0.7, 0.3]
alpha = 0.05
epsilon = 0.02
N = 10
#bn=gum.randomBN(n=n) # binary by default. also initializes the cpts
tester=chisq
testname = 'chisq'
param_file=dir_name+'/'+'params'
np.savez(param_file,n=n,domain=domain,N=N,density=ratio_arc)
seed = 2020

# for i in range(N):
#     # fix the seed to generate
#     gum.initRandom(i + seed)
#     bn=gum.randomBN(n=n,names=node_names,ratio_arc=ratio_arc,domain_size=domain) # ratio_arc=1.2 default
#     bn.generateCPTs()
#     refillCPT_Dirichlet(bn,node_names,domain)    

#     bn.saveBIF(dir_name+'/'+str(i)+'.bif')
    
def create_name_dict(bn): # consistently recover nodeID-nodeName mapping. Gives col/row numbers in adj matrix
    name_dict = {}
    for i in np.arange(1,n+1):#bn.names()
        #print('X'+str(i),bn.nodeId(bn.variableFromName('X'+str(i))))
        name_dict[bn.nodeId(bn.variableFromName('X'+str(i)))]=i
    return name_dict

def get_ess_adj(bnEs,n):
    ess_adj = np.zeros((n,n))
    for i in bnEs.arcs():
        ess_adj[i[1],i[0]]=1
        ess_adj[i[0],i[1]]=-1
    for i in bnEs.edges():
        ess_adj[i[1],i[0]]=-1
        ess_adj[i[0],i[1]]=-1
    return ess_adj

def combined_metric(my_list,N,proj=None):
    if proj=='total':
        total=[my_list[i][0]+my_list[i][1]+my_list[i][2] for i in range(N)]
    elif proj=='arrow':
        total=[my_list[i][1] for i in range(N)]
    elif proj=='tail':
        total=[my_list[i][2] for i in range(N)]
    elif proj =='skel':
        total=[my_list[i][0] for i in range(N)]
    elif proj =='arr_tail':
        total=[my_list[i][1]+my_list[i][2] for i in range(N)]
    else:
        print('Error!! Enter projection type for F1 scores!')
    #total=[my_list[i][1] for i in range(N)]
    return total

def PP_graph(adj):
    # remove bidirected edges since they are known to not be present in any DAG
    n = np.shape(adj)[0]
    for i in range(n):
        for j in range(n):
            if adj[i,j]==1 and adj[j,i]==1:
                adj[i,j]=0
                adj[j,i]=0
    return adj

def get_adj(bn,n):
    adj_ = np.zeros((n,n))
    for i in bn.arcs():
        adj_[i[1],i[0]]=1
        adj_[i[0],i[1]]=-1
    return adj_

showgraphs=False

def plot_cdf(my_list,N,proj,line_style,color):
    total=combined_metric(my_list,N,proj)
    new_total=[]
    for i in total:
        if np.isnan(i):
            new_total.append(0)
        else:
            new_total.append(i)
    total=new_total
    H,X1=np.histogram(total,bins=100,density=True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    plt.plot(X1[1:], F1,line_style,color=color,linewidth=2)

num_nodes = [10, 20, 30, 40, 50]
num_edges = [20, 40, 60, 80, 100]
#num_nodes = [10,]
#num_edges = [20]
NO_SAMPLES = 1000
N= 10
for counter, n in enumerate(num_nodes):
    node_names=['X'+str(i) for i in np.arange(1,n+1)]
    ls_var_idx = [i for i in range(n)]


    # setID is preserved from prev cell
    # param_file=dir_name+'/'+'params'
    # params=np.load(param_file+'.npz')
    # n=int(params['n'])
    # domain=int(params['domain'])
    # N=int(params['N'])
    # density=int(params['density'])


    density=ratio_arc

    proj_types=['total','arrow','tail','skel','arr_tail']

    k_range=[0,1,2]

    max_degree_list = []
    cpc_list =[]
    cpc_greaterthan50_list = []
    cpc_MI_list = []
    cpc_MV_list = []
    ges_list = []
    grasp_list = []
    notear_list = []
    sat_list = []

    kpc_dict = {}
    kpc_ess_no_fas_dict = {}
    kpc_fas_dict = {}
    for k in k_range:
        kpc_dict[k]=[]

    num_edge = num_edges[counter]

    # Create directory for saving matrices if it doesn't exist
    matrices_dir = os.path.join(dir_name, 'matrices')
    if not os.path.exists(matrices_dir):
        os.makedirs(matrices_dir)

    for i in range(N):  
        print('Working on instance %d'%i)
        # bn = gum.loadBN(dir_name+'/'+str(i)+'.bif')
        iseed = i + seed
        random.seed(iseed)  
        
        print("Begin constructing the random DAG...")
        bn, table, high_state_list = randomBNwithSpecificStates(n, num_edge, states =states, p=p_states, graphidx=i, seed=seed)
        print("Finish constructing the random DAG.")

        if showgraphs:
            gr = BN2dot(bn)
            gr.write(dir_name+'/'+'test_instance_'+str(i)+'True.pdf', format='pdf')
        # new_ls_name = [c for c in ls_var_idx if c not in high_state_list]
        # non-data-driven way to select the candidate sets
        # if new_ls_name and num_order1_sets <= len(new_ls_name):
        #     orderone_list_for_I = random.sample(list(itertools.combinations(new_ls_name, 1)), num_order1_sets)
        #     # add heuristic to find the conditioning set, add lambda to alpha and only pick the variables that is beyond alpha+-lambda
        #     I = [set()] + [set(a) for a in orderone_list_for_I]
        # else:
        #     I = [set()]
        # ordertwo_list_for_I = random.sample(list(itertools.combinations(ls_var_idx, 2)), num_order2_sets)
        # I = [set()] + [set(a) for a in orderone_list_for_I] + [set(w) for w in ordertwo_list_for_I]

     
        gum.initRandom(iseed)
        g=gum.BNDatabaseGenerator(bn)
        g.drawSamples(NO_SAMPLES)
        df=g.to_pandas()
        data=df.to_numpy()
        data = data.astype(float)

        bnEs=gum.EssentialGraph(bn)
        ess_adj=get_ess_adj(bnEs,n) 

        j = 5
        num_top_sets = 4
        
        print("CPC started...")
        start_time = time.time()
        D,_ = CPC_modified(data,df, tester, 1, j, num_top_sets, n,[], [], alpha=alpha, cc_set_selection_method='MV')
        cpc_time = time.time() - start_time
        adj = D.graph
        print("f1arrowhead-score of CPC:")
        print(fscore_calculator_arrowhead(adj,ess_adj))
        cpc_MV_list.append((fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj)))

        # add solver-based causal discovery algortihm
        print("SAT started...")
        data_sat = df.astype(int)
        start_time = time.time()
        discovery = SATCausalDiscovery(significance_level=0.05, max_cond_size=1)
        adjacency_matrix = discovery.discover(data_sat)
        sat_time = time.time() - start_time
        sat_list.append((fscore_calculator_skeleton(adjacency_matrix,ess_adj),fscore_calculator_arrowhead(adjacency_matrix,ess_adj),fscore_calculator_tail(adjacency_matrix,ess_adj)))
        print("f1arrowhead-score of SAT:")
        print(fscore_calculator_arrowhead(adjacency_matrix,ess_adj))

        # Store timing information
        if 'cpc_times' not in locals():
            cpc_times = []
            sat_times = []
        cpc_times.append(cpc_time)
        sat_times.append(sat_time)

        # Save matrices for this iteration
        matrix_file = os.path.join(matrices_dir, f'n{n}_iter{i}.npz')
        np.savez(matrix_file,
                ground_truth=ess_adj,
                cpc_adj=adj,
                sat_adj=adjacency_matrix,
                n=n,
                iteration=i)

    # Calculate averages and standard errors
    cpc_avg = np.mean(cpc_MV_list, axis=0)
    sat_avg = np.mean(sat_list, axis=0)
    cpc_std = np.std(cpc_MV_list, axis=0) / np.sqrt(N)
    sat_std = np.std(sat_list, axis=0) / np.sqrt(N)
    
    # Store results for each n
    if 'results' not in locals():
        results = {
            'n_values': [],
            'cpc_skel_avg': [], 'cpc_skel_std': [],
            'cpc_arrow_avg': [], 'cpc_arrow_std': [],
            'cpc_tail_avg': [], 'cpc_tail_std': [],
            'sat_skel_avg': [], 'sat_skel_std': [],
            'sat_arrow_avg': [], 'sat_arrow_std': [],
            'sat_tail_avg': [], 'sat_tail_std': [],
            'cpc_time_avg': [], 'cpc_time_std': [],
            'sat_time_avg': [], 'sat_time_std': []
        }
    
    results['n_values'].append(n)
    results['cpc_skel_avg'].append(cpc_avg[0])
    results['cpc_skel_std'].append(cpc_std[0])
    results['cpc_arrow_avg'].append(cpc_avg[1])
    results['cpc_arrow_std'].append(cpc_std[1])
    results['cpc_tail_avg'].append(cpc_avg[2])
    results['cpc_tail_std'].append(cpc_std[2])
    
    results['sat_skel_avg'].append(sat_avg[0])
    results['sat_skel_std'].append(sat_std[0])
    results['sat_arrow_avg'].append(sat_avg[1])
    results['sat_arrow_std'].append(sat_std[1])
    results['sat_tail_avg'].append(sat_avg[2])
    results['sat_tail_std'].append(sat_std[2])
    
    results['cpc_time_avg'].append(np.mean(cpc_times))
    results['cpc_time_std'].append(np.std(cpc_times) / np.sqrt(N))
    results['sat_time_avg'].append(np.mean(sat_times))
    results['sat_time_std'].append(np.std(sat_times) / np.sqrt(N))

# Create the plots
import matplotlib.pyplot as plt

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

# Time plot
plt.figure(figsize=(10, 6))
plt.errorbar(results['n_values'], results['cpc_time_avg'], yerr=results['cpc_time_std'], 
             fmt='-o', label='CPC', capsize=5, color='blue')
plt.errorbar(results['n_values'], results['sat_time_avg'], yerr=results['sat_time_std'], 
             fmt='-s', label='SAT', capsize=5, color='red')
plt.xlabel('Number of Variables')
plt.ylabel('Time (seconds)')
plt.title('Computation Time vs Number of Variables')
plt.legend()
plt.savefig('time_vs_n.pdf', bbox_inches='tight')
plt.close()