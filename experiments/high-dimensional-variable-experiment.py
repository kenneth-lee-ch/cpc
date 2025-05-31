import os
import sys
import time

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


sys.path.append("")
import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.cit import chisq
import pyAgrum as gum
from algorithms.core.kpc import kPC
from algorithms.core.cpc import  CPC_modified
from algorithms.utils.graph_utils import fscore_calculator_skeleton, fscore_calculator_arrowhead, fscore_calculator_tail, pr_calculator_skeleton, pr_calculator_arrowhead, pr_calculator_tail
from algorithms.utils.graph_utils import visualize_graph
from algorithms.utils.graph_utils import create_CPT
from pyAgrum.lib.bn2graph import BN2dot
import random
from typing import List, Dict, Tuple, Set
from algorithms.utils.cit import *
import csv


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
n = 20
ratio_arc=2

domain=2 # states for each node. 
node_names=['X'+str(i) for i in np.arange(1,n+1)]
sample_sizes = [2000]
num_order1_sets = 15 
num_order2_sets = 20
num_edges = [60] # set 12 is 100 edges  and set 10 is 60 edges
states = [2, 30]
p_states = [0.7, 0.3]
alpha = 0.05
epsilon = 0.02

#bn=gum.randomBN(n=n) # binary by default. also initializes the cpts
N=100
tester=chisq
testname = 'chisq'
param_file=dir_name+'/'+'params'
np.savez(param_file,n=n,domain=domain,N=N,density=ratio_arc)
seed = 2020 # 2020 original
ls_var_idx = [i for i in range(n)]

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


for NO_SAMPLES in sample_sizes:

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
    cpc_MV_p_list = []
    cpc_MV_r_list = []
    ges_list = []
    ges_p_list = []
    ges_r_list = []

    grasp_list = []
    grasp_p_list = []
    grasp_r_list = []
    notear_list = []
    sat_list = []

    kpc_dict = {}
    kpc_p_dict = {}
    kpc_r_dict = {}
    kpc_ess_no_fas_dict = {}
    kpc_fas_dict = {}
    for k in k_range:
        kpc_dict[k]=[]
        kpc_p_dict[k]=[]
        kpc_r_dict[k]=[]

    for i in range(N):  
        print('Working on instance %d'%i)
        # bn = gum.loadBN(dir_name+'/'+str(i)+'.bif')
        iseed = i + seed
        iseed = iseed % 41
        random.seed(iseed)
        if i == 73:
            iseed = iseed + 100  
        num_edge = random.sample(num_edges, 1)[0]
        print(num_edge)
        print("Begin constructing the random DAG...")
        bn, table, high_state_list = randomBNwithSpecificStates(n, num_edge, states =states, p=p_states, graphidx=i, seed=seed)
        print("Finish constructing the random DAG.")

        if showgraphs:
            gr = BN2dot(bn)
            gr.write(dir_name+'/'+'test_instance_'+str(i)+'True.pdf', format='pdf')
        new_ls_name = [c for c in ls_var_idx if c not in high_state_list]
        # non-data-driven way to select the candidate sets
        # if new_ls_name and num_order1_sets <= len(new_ls_name):
        #     orderone_list_for_I = random.sample(list(itertools.combinations(new_ls_name, 1)), num_order1_sets)
        #     # add heuristic to find the conditioning set, add lambda to alpha and only pick the variables that is beyond alpha+-lambda
        #     I = [set()] + [set(a) for a in orderone_list_for_I]
        # else:
        #     I = [set()]
        # ordertwo_list_for_I = random.sample(list(itertools.combinations(ls_var_idx, 2)), num_order2_sets)
        # I = [set()] + [set(a) for a in orderone_list_for_I] + [set(w) for w in ordertwo_list_for_I]

        for t in range(3): # using 3 datasets to average out finite-sample effects
            t_seed = iseed+t
            gum.initRandom(t_seed)
            g=gum.BNDatabaseGenerator(bn)
            g.drawSamples(NO_SAMPLES)
            df=g.to_pandas()
            data=df.to_numpy()
            data = data.astype(float)

            bnEs=gum.EssentialGraph(bn)

            ### COMPARE AGAINST ESS OR DAG
            ess_adj=get_ess_adj(bnEs,n) 
            #true_adj = get_adj(bn,n)
            #ess_adj = true_adj# use ground truth graph
            print("GES started...")
            record = ges(data, score_func='local_score_BDeu')
            G=record['G']
            

            # if t==0 and showgraphs:
            #    visualize_graph(G,name=dir_name+ '/' + 'test_instance_' + str(i)+'ges')
            adj=G.graph

            ges_list.append((fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj)))

            sk_precision, sk_recall = pr_calculator_skeleton(adj,ess_adj)
            ar_precision, ar_recall = pr_calculator_arrowhead(adj,ess_adj)
            ta_precision, ta_recall = pr_calculator_tail(adj,ess_adj)

            ges_p_list.append((sk_precision,ar_precision,ta_precision))
            ges_r_list.append((sk_recall,ar_recall,ta_recall))


            print("GRasP started...")

            G = grasp(data , score_func='local_score_BDeu')
            adj=G.graph

            grasp_list.append((fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj)))
            sk_precision, sk_recall = pr_calculator_skeleton(adj,ess_adj)
            ar_precision, ar_recall = pr_calculator_arrowhead(adj,ess_adj)
            ta_precision, ta_recall = pr_calculator_tail(adj,ess_adj)

            grasp_p_list.append((sk_precision,ar_precision,ta_precision))
            grasp_r_list.append((sk_recall,ar_recall,ta_recall))


            j = 5
            num_top_sets = 4

           
            print("CPC started...")
            D,_ = CPC_modified(data,df, tester, 1, j, num_top_sets, n,[], [], alpha=alpha, cc_set_selection_method='chi-sq')
            adj = D.graph
            print("f1arrowhead-score of CPC:")

            print(fscore_calculator_arrowhead(adj,ess_adj))
            cpc_MV_list.append((fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj)))
            sk_precision, sk_recall = pr_calculator_skeleton(adj,ess_adj)
            ar_precision, ar_recall = pr_calculator_arrowhead(adj,ess_adj)
            ta_precision, ta_recall = pr_calculator_tail(adj,ess_adj)

            cpc_MV_p_list.append((sk_precision,ar_precision,ta_precision))
            cpc_MV_r_list.append((sk_recall,ar_recall,ta_recall))


            print("kPC started...")

            for k in k_range:
                D,_ = kPC(data,tester,k,n,alpha=alpha, fastAdjSearch=False)
                if t==0 and showgraphs:
                    visualize_graph(D, name=dir_name+ '/' + 'test_instance_' + str(i)+'k_' + str(k)) 
                adj = D.graph
                # OPTIONAL: Post-processing and removing <-> edges. 
                # With finite-sample noise, this mostly hurts performance due to extra bidirected
                # edges recovered because of incorrect marginal independences
                #adj = PP_graph(adj)

                #print(fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj))


                kpc_dict[k].append((fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj)))
                sk_precision, sk_recall = pr_calculator_skeleton(adj,ess_adj)
                ar_precision, ar_recall = pr_calculator_arrowhead(adj,ess_adj)
                ta_precision, ta_recall = pr_calculator_tail(adj,ess_adj)

                kpc_p_dict[k].append((sk_precision,ar_precision,ta_precision))
                kpc_r_dict[k].append((sk_recall,ar_recall,ta_recall))
    





    
    def r_title_name(proj):
        if proj=='total':
            title='$Recall^{sk}$+$F_1^{ar}$+$F_1^{ta}$'
        elif proj=='arrow':
            title='$Recall^{ar}$'
        elif proj=='tail':
            title='$Recall^{ta}$'
        elif proj =='skel':
            title='$Recall^{sk}$'
        elif proj =='arr_tail':
            title='$Recall^{ar}$+$F_1^{ta}$'
        else:
            print('Error!! Enter projection type for F1 scores!')
        #total=[my_list[i][1] for i in range(N)]
        return title
    
    def p_title_name(proj):
        if proj=='total':
            title='$Precision^{sk}$+$F_1^{ar}$+$F_1^{ta}$'
        elif proj=='arrow':
            title='$Precision^{ar}$'
        elif proj=='tail':
            title='$Precision^{ta}$'
        elif proj =='skel':
            title='$Precision^{sk}$'
        elif proj =='arr_tail':
            title='$Precision^{ar}$+$F_1^{ta}$'
        else:
            print('Error!! Enter projection type for F1 scores!')
        #total=[my_list[i][1] for i in range(N)]
        return title

    #### F1 scores  #########
    def title_name(proj):
        if proj=='total':
            title='$F_1^{sk}$+$F_1^{ar}$+$F_1^{ta}$'
        elif proj=='arrow':
            title='$F_1^{ar}$'
        elif proj=='tail':
            title='$F_1^{ta}$'
        elif proj =='skel':
            title='$F_1^{sk}$'
        elif proj =='arr_tail':
            title='$F_1^{ar}$+$F_1^{ta}$'
        else:
            print('Error!! Enter projection type for F1 scores!')
        #total=[my_list[i][1] for i in range(N)]
        return title
    

    import matplotlib.pyplot as plt
    SMALL_SIZE = 20
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 26

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    for proj in proj_types:
        plt.figure()


        colors=['maroon','olive','pink']
        # colors2= ['blue', 'red', 'purple']
        # colors3= ['peru', 'yellow', 'pink']
        linetypes=['--','-.',':']
        for k in k_range:
            my_list = kpc_dict[k]
            with open(dir_name+'/'+'kpc_f1_'+proj+'k_%d_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(k, n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list)
            # saving data
            line_style=linetypes[k]
            color=colors[k]
            plot_cdf(my_list,N,proj,line_style,color)


        my_list = ges_list
        with open(dir_name+'/'+'ges_f1_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='--'
        color='purple'
        plot_cdf(my_list,N,proj,line_style,color)

        my_list = grasp_list
        with open(dir_name+'/'+'grasp_f1_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='--'
        color='orange'
        plot_cdf(my_list,N,proj,line_style,color)

        # my_list = cpc_list
        # with open(dir_name+'/'+'cpc_f1_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(my_list)
        
        # line_style='-'
        # color='blue'
        # plot_cdf(my_list,N,proj,line_style,color)
        


        my_list = cpc_MV_list

        with open(dir_name+'/'+'cpcMV_f1_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='-'
        color='blue'
        plot_cdf(my_list,N,proj,line_style,color)






        # my_list = notear_list
        # line_style='--'
        # color='cyan'
        # plot_cdf(my_list,N,proj,line_style,color)


        title=title_name(proj)
        plt.title('CDF of '+title+', N=%d'%NO_SAMPLES)
        legend_text = ['kPC,k='+str(i) for i in k_range] + ["GES"] + ['GRaSP'] + ['CPC'] 
        #legend_text = ['CC-kPC'] +  ['kPC,k='+str(i) for i in k_range] 
        #plt.legend(['PC','kPC,k=1', 'kPC,k=2'])
        # plt.legend([i for i in legend_text],loc='lower right')
        plt.xticks(fontsize=18, fontweight='bold')  # Custom labels with rotation
        plt.yticks(fontsize=18, fontweight='bold')
        lgd = plt.legend([i for i in legend_text],loc='center left', bbox_to_anchor=(1, 0.5),prop={'weight': 'bold'})
        ax = plt.gca()
        plt.xlabel('F1-score', fontsize=18, fontweight='bold')
        plt.ylabel('CDF', fontsize=18, fontweight='bold')
        #ax.set_ylim([0, 1])
        #plt.savefig(dir_name+'/'+'cdf_combined_n_%d_k_%d_dom_%d_den_%d_samples_%d.pdf'%(n,k,domain,density,NO_SAMPLES),transparent=True)
        plt.savefig(dir_name+'/'+'cdf_'+proj+'_n_%d_k_%d_dom_%d_den_%d_samples_%d_test_%s.pdf'%(n,k,domain,density,NO_SAMPLES, str(testname)),bbox_extra_artists=(lgd,), transparent=True, bbox_inches='tight')
        plt.close()


    #### precision #########
    for proj in proj_types:
        plt.figure()


        colors=['maroon','olive','pink']
        # colors2= ['blue', 'red', 'purple']
        # colors3= ['peru', 'yellow', 'pink']
        linetypes=['--','-.',':']
        for k in k_range:
            my_list = kpc_p_dict[k]
            with open(dir_name+'/'+'kpc_precision_'+proj+'k_%d_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(k, n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list)
            # saving data
            line_style=linetypes[k]
            color=colors[k]
            plot_cdf(my_list,N,proj,line_style,color)


        my_list = ges_p_list
        with open(dir_name+'/'+'ges_precision_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='--'
        color='purple'
        plot_cdf(my_list,N,proj,line_style,color)

        my_list = grasp_p_list
        with open(dir_name+'/'+'grasp_precision_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='--'
        color='orange'
        plot_cdf(my_list,N,proj,line_style,color)

        # my_list = cpc_list
        # with open(dir_name+'/'+'cpc_f1_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(my_list)
        
        # line_style='-'
        # color='blue'
        # plot_cdf(my_list,N,proj,line_style,color)
        


        my_list = cpc_MV_p_list

        with open(dir_name+'/'+'cpcMV_precision_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='-'
        color='blue'
        plot_cdf(my_list,N,proj,line_style,color)






        # my_list = notear_list
        # line_style='--'
        # color='cyan'
        # plot_cdf(my_list,N,proj,line_style,color)


        title=p_title_name(proj)
        plt.title('CDF of '+title+', N=%d'%NO_SAMPLES)
        legend_text = ['kPC,k='+str(i) for i in k_range] + ["GES"] + ['GRaSP'] + ['CPC'] 
        #legend_text = ['CC-kPC'] +  ['kPC,k='+str(i) for i in k_range] 
        #plt.legend(['PC','kPC,k=1', 'kPC,k=2'])
        # plt.legend([i for i in legend_text],loc='lower right')
        plt.xticks(fontsize=18, fontweight='bold')  # Custom labels with rotation
        plt.yticks(fontsize=18, fontweight='bold')
        lgd = plt.legend([i for i in legend_text],loc='center left', bbox_to_anchor=(1, 0.5),prop={'weight': 'bold'})
        ax = plt.gca()
        plt.xlabel('Precision', fontsize=18, fontweight='bold')
        plt.ylabel('CDF', fontsize=18, fontweight='bold')
        #ax.set_ylim([0, 1])
        #plt.savefig(dir_name+'/'+'cdf_combined_n_%d_k_%d_dom_%d_den_%d_samples_%d.pdf'%(n,k,domain,density,NO_SAMPLES),transparent=True)
        plt.savefig(dir_name+'/'+'cdf_precision_'+proj+'_n_%d_k_%d_dom_%d_den_%d_samples_%d_test_%s.pdf'%(n,k,domain,density,NO_SAMPLES, str(testname)),bbox_extra_artists=(lgd,), transparent=True, bbox_inches='tight')
        plt.close()



    ##### recall ###########
    for proj in proj_types:
        plt.figure()


        colors=['maroon','olive','pink']
        # colors2= ['blue', 'red', 'purple']
        # colors3= ['peru', 'yellow', 'pink']
        linetypes=['--','-.',':']
        for k in k_range:
            my_list = kpc_r_dict[k]
            with open(dir_name+'/'+'kpc_recall_'+proj+'k_%d_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(k, n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list)
            # saving data
            line_style=linetypes[k]
            color=colors[k]
            plot_cdf(my_list,N,proj,line_style,color)


        my_list = ges_r_list
        with open(dir_name+'/'+'ges_recall_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='--'
        color='purple'
        plot_cdf(my_list,N,proj,line_style,color)

        my_list = grasp_r_list
        with open(dir_name+'/'+'grasp_recall_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='--'
        color='orange'
        plot_cdf(my_list,N,proj,line_style,color)

        # my_list = cpc_list
        # with open(dir_name+'/'+'cpc_f1_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(my_list)
        
        # line_style='-'
        # color='blue'
        # plot_cdf(my_list,N,proj,line_style,color)
        


        my_list = cpc_MV_r_list

        with open(dir_name+'/'+'cpcMV_recall_'+proj+'_n_%d_dom_%d_den_%d_samples_%d_test_%s.csv'%(n,domain,density,NO_SAMPLES, str(testname)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(my_list)
        
        line_style='-'
        color='blue'
        plot_cdf(my_list,N,proj,line_style,color)






        # my_list = notear_list
        # line_style='--'
        # color='cyan'
        # plot_cdf(my_list,N,proj,line_style,color)


        title=r_title_name(proj)
        plt.title('CDF of '+title+', N=%d'%NO_SAMPLES)
        legend_text = ['kPC,k='+str(i) for i in k_range] + ["GES"] + ['GRaSP'] + ['CPC'] 
        #legend_text = ['CC-kPC'] +  ['kPC,k='+str(i) for i in k_range] 
        #plt.legend(['PC','kPC,k=1', 'kPC,k=2'])
        # plt.legend([i for i in legend_text],loc='lower right')
        plt.xticks(fontsize=18, fontweight='bold')  # Custom labels with rotation
        plt.yticks(fontsize=18, fontweight='bold')
        lgd = plt.legend([i for i in legend_text],loc='center left', bbox_to_anchor=(1, 0.5),prop={'weight': 'bold'})
        ax = plt.gca()
        plt.xlabel('Recall', fontsize=18, fontweight='bold')
        plt.ylabel('CDF', fontsize=18, fontweight='bold')
        #ax.set_ylim([0, 1])
        #plt.savefig(dir_name+'/'+'cdf_combined_n_%d_k_%d_dom_%d_den_%d_samples_%d.pdf'%(n,k,domain,density,NO_SAMPLES),transparent=True)
        plt.savefig(dir_name+'/'+'cdf_recall_'+proj+'_n_%d_k_%d_dom_%d_den_%d_samples_%d_test_%s.pdf'%(n,k,domain,density,NO_SAMPLES, str(testname)),bbox_extra_artists=(lgd,), transparent=True, bbox_inches='tight')
        plt.close()