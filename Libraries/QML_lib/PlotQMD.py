import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import ticker
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.ticker import Formatter
#from QMD import  *
#from QML import *
import DataBase
import Evo as evo

#### Hinton Disagram ####

def latex_name_ising(name):
    terms=name.split('PP')
    rotations = ['xTi', 'yTi', 'zTi']
    hartree_fock = ['xTx', 'yTy', 'zTz']
    transverse = ['xTy', 'xTz', 'yTz']
    
    
    present_r = []
    present_hf = []
    present_t = []
    
    for t in terms:
        if t in rotations:
            present_r.append(t[0])
        elif t in hartree_fock:
            present_hf.append(t[0])
        elif t in transverse:
            string = t[0]+t[-1]
            present_t.append(string)
        else:
            print("Term",t,"doesn't belong to rotations, Hartree-Fock or transverse.")
            print("Given name:", name)
    present_r.sort()
    present_hf.sort()
    present_t.sort()

    r_terms = ','.join(present_r)
    hf_terms = ','.join(present_hf)
    t_terms = ','.join(present_t)
    
    
    latex_term = ''
    if len(present_r) > 0:
        latex_term+='R_{'+r_terms+'}'
    if len(present_hf) > 0:
        latex_term+='HF_{'+hf_terms+'}'
    if len(present_t) > 0:
        latex_term+='T_{'+t_terms+'}'
    
    final_term = 'r$'+latex_term+'$'
    
    return latex_term

def ExpectationValuesTrueSim(qmd, model_ids=None, champ=True, max_time=3, t_interval=0.01, save_to_file=None):
    import random
    if model_ids is None and champ == True:
        model_ids = [qmd.ChampID]
    elif model_ids is not None and champ == True:

        if type(model_ids) is not list:
            model_ids = [model_ids]
        if qmd.ChampID not in model_ids:
            model_ids.append(qmd.ChampID)

    probe_id = random.choice(range(qmd.NumProbes))
    times = np.arange(0, max_time, t_interval)
    true_colour ='b'
    champion_colour = 'r'
    sim_colours = ['g', 'c', 'm', 'y', 'k']

    plt.clf()
    plt.xlabel('Time')
    plt.ylabel('Expectation Value')

    true = qmd.TrueOpName
    true_op = DataBase.operator(true)
    true_params = qmd.TrueParamsList
    true_ops = true_op.constituents_operators
    true_ham = np.tensordot(true_params, true_ops, axes=1)
    true_dim = true_op.num_qubits
    true_probe = qmd.ProbeDict[(probe_id,true_dim)]
    true_expec_values = [evo.expectation_value(ham=true_ham, t=t, state=true_probe) for t in times]
    plt.scatter(times, true_expec_values, label='True Expectation Value', marker='x', color = true_colour)

    
    ChampionsByBranch = {v:k for k,v in qmd.BranchChampions.items()}
    for i in range(len(model_ids)):
        mod_id = model_ids[i]
        sim = qmd.ModelNameIDs[mod_id]
        sim_op  = DataBase.operator(sim)
        mod=qmd.reducedModelInstanceFromID(mod_id)
        sim_params = list(mod.FinalParams[:,0])
        sim_ops = sim_op.constituents_operators
        sim_ham = np.tensordot(sim_params, sim_ops, axes=1)
        sim_dim = sim_op.num_qubits
        sim_probe = qmd.ProbeDict[(probe_id,sim_dim)]
        colour_id = int(i%len(sim_colours))
        sim_col = sim_colours[colour_id]
        sim_expec_values = [evo.expectation_value(ham=sim_ham, t=t, state=sim_probe) for t in times]

        if mod_id == qmd.ChampID:
            models_branch = ChampionsByBranch[mod_id]
#            sim_label = 'Champion Model (Branch ' +str(models_branch)+')'
            sim_label = 'Champion Model'
            sim_col = champion_colour
        elif mod_id in list(qmd.BranchChampions.values()):
            models_branch = ChampionsByBranch[mod_id]
            sim_label = 'Branch '+str(models_branch)+' Champion'
        else:
            sim_label = 'Model '+str(mod_id)

        plt.plot(times, sim_expec_values, label=sim_label, color=sim_col)

    ax = plt.subplot(111)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    qty = 0.1
    ax.set_position([box.x0, box.y0 + box.height * qty,
                     box.width, box.height * (1.0-qty)])

    handles, labels = ax.get_legend_handles_labels()
    label_list = list(labels)
    handle_list = list(handles)

    new_labels=[]
    new_handles=[]

    special_labels=[]
    special_handles=[]

    special_terms = ['True Expectation Value', 'Champion Model']

    for i in range(len(label_list)):
        if label_list[i] in special_terms:
            special_labels.append(label_list[i])
            special_handles.append(handle_list[i])
        else:
            new_labels.append(label_list[i])
            new_handles.append(handle_list[i])


    special_handles = tuple(special_handles)
    special_labels = tuple(special_labels)

    extra_lgd = True
    if len(new_handles) == 0:
        print("No models other than champ/true")
        extra_lgd=False
        
    new_handles = tuple(new_handles)
    new_labels = tuple(new_labels)

    if extra_lgd:
        lgd_spec=ax.legend(special_handles, special_labels, loc='upper center', bbox_to_anchor=(1, 1),fancybox=True, shadow=True, ncol=1)
        lgd_new=ax.legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(1.15, 0.75),fancybox=True, shadow=True, ncol=1)
        plt.gca().add_artist(lgd_spec)
    else:
        lgd_spec=ax.legend(special_handles, special_labels, loc='upper center', bbox_to_anchor=(1, 1),fancybox=True, shadow=True, ncol=1)
        
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')

    
    
def BayF_IndexDictToMatrix(ModelNames, AllBayesFactors, StartBayesFactors=None):
    
    size = len(ModelNames)
    Bayf_matrix = np.zeros([size,size])
    
    for i in range(size):
        for j in range(size):
            try: 
                Bayf_matrix[i,j] = AllBayesFactors[i][j][-1]
            except:
                Bayf_matrix[i,j] = 1
    
            # elif j<i and (StartBayesFactors is not None):
                # try: 
                    # Bayf_matrix[i,j] = StartBayesFactors[i][j]
                # except:
                    # Bayf_matrix[i,j] = 1
    
    return Bayf_matrix
    

class SquareCollection(collections.RegularPolyCollection):
    """Return a collection of squares."""

    def __init__(self, **kwargs):
        super(SquareCollection, self).__init__(4, rotation=np.pi/4., **kwargs)

    def get_transform(self):
        """Return transform scaling circle areas to data space."""
        ax = self.axes
        pts2pixels = 72.0 / ax.figure.dpi
        scale_x = pts2pixels * ax.bbox.width / ax.viewLim.width
        scale_y = pts2pixels * ax.bbox.height / ax.viewLim.height
        return transforms.Affine2D().scale(scale_x, scale_y)
        
        
        
class IndexLocator(ticker.Locator):

    def __init__(self, max_ticks=21):
        self.max_ticks = max_ticks

    def __call__(self):
        """Return the locations of the ticks."""
        dmin, dmax = self.axis.get_data_interval()
        if dmax < self.max_ticks:
            step = 1
        else:
            step = np.ceil(dmax / self.max_ticks)
        return self.raise_if_exceeds(np.arange(0, dmax, step))


        
def hinton(inarray, max_value=None, use_default_ticks=True, skip_diagonal = True, skip_which = None, grid = True, white_half = 0., where_labels = 'bottomleft'):
    """Plot Hinton diagram for visualizing the values of a 2D array.

    Plot representation of an array with positive and negative values
    represented by white and black squares, respectively. The size of each
    square represents the magnitude of each value.

    AAG modified 04/2018

    Parameters
    ----------
    inarray : array
        Array to plot.
    max_value : float
        Any *absolute* value larger than `max_value` will be represented by a
        unit square.
    use_default_ticks: boolean
        Disable tick-generation and generate them outside this function.
    skip_diagonal: boolean
        remove plotting of values on the diagonal
    skip_which: None, upper, lower
        whether to plot both upper and lower triangular matrix or just one of them
    grid: Boolean
        to remove the grid from the plot
    white_half : float
        adjust the size of the white "coverage" of the "skip_which" part of the diagram
    where_labels: "bottomleft", "topright"
        move the xy labels and ticks to the corresponding position
    """

    ax = plt.gca()
    ax.set_facecolor('silver')
    # make sure we're working with a numpy array, not a numpy matrix
    inarray = np.asarray(inarray)
    height, width = inarray.shape
    if max_value is None:
        finite_inarray = inarray[np.where(inarray>-np.inf)]
        max_value = 2**np.ceil(np.log(np.max(np.abs(finite_inarray)))/np.log(2))
    values = np.clip(inarray/max_value, -1, 1)
    rows, cols = np.mgrid[:height, :width]

    pos = np.where( np.logical_and(values > 0 , np.abs(values) < np.inf)  )
    neg = np.where( np.logical_and(values < 0 , np.abs(values) < np.inf) )

    # if skip_diagonal:
        # for mylist in [pos,neg]:
            # diags = np.array([ elem[0] == elem[1] for elem in mylist ])
            # diags = np.where(diags == True)
            # print(diags)
            # for elem in diags:
                # del(mylist[elem])
                # del(mylist[elem])    
    
    for idx, color in zip([pos, neg], ['white', 'black']):
        if len(idx[0]) > 0:
            xy = list(zip(cols[idx], rows[idx]))

            circle_areas = np.pi / 2 * np.abs(values[idx])
            if skip_diagonal:
                diags = np.array([ elem[0] == elem[1] for elem in xy ])
                diags = np.where(diags == True)
                
                for delme in diags[0][::-1]:
                    circle_areas[delme] = 0
            
            if skip_which is not None:
                if skip_which is 'upper':
                    lows = np.array([ elem[0] > elem[1] for elem in xy ])
                if skip_which is 'lower':
                    lows = np.array([ elem[0] < elem[1] for elem in xy ])
                lows = np.where(lows == True)
                
                for delme in lows[0][::-1]:
                    circle_areas[delme] = 0 
            
            squares = SquareCollection(sizes=circle_areas,
                                       offsets=xy, transOffset=ax.transData,
                                       facecolor=color, edgecolor=color)
            ax.add_collection(squares, autolim=True)
            
    if white_half > 0:
        for i in range(width):
            for j in range(i):
                
                xy = [(i,j)] if skip_which is 'upper' else [(j,i)]

                squares = SquareCollection(sizes=[white_half],
                                       offsets=xy, transOffset=ax.transData,
                                       facecolor='white', edgecolor='white')
                ax.add_collection(squares, autolim=True)
                

    ax.axis('scaled')
    # set data limits instead of using xlim, ylim.
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(height-0.5, -0.5)
    
    if grid: ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    if use_default_ticks:
        ax.xaxis.set_major_locator(IndexLocator())
        ax.yaxis.set_major_locator(IndexLocator())
        
    if where_labels is 'topright':
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        
        
        
def format_fn(tick_val, tick_pos, labels):
    
    if int(tick_val) in range(len(labels)):
        return labels[int(tick_val)]
    else:
        return ''
        
        
class QMDFuncFormatter(Formatter):
    """
    Use a user-defined function for formatting.

    The function should take in two inputs (a tick value ``x`` and a
    position ``pos``), and return a string containing the corresponding
    tick label.
    """
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __call__(self, x, pos=None):
        """
        Return the value of the user defined function.

        `x` and `pos` are passed through as-is.
        """
        return self.func(x, pos, self.args)   
        
def plotHinton(model_names, bayes_factors, save_to_file=None):
    hinton_mtx=BayF_IndexDictToMatrix(model_names, bayes_factors)
    log_hinton_mtx = np.log10(hinton_mtx)
    labels = [latex_name_ising(name) for name in model_names.values()]


    fig, ax = plt.subplots(figsize=(7,7))

    hinton(log_hinton_mtx, use_default_ticks=True, skip_diagonal=True, where_labels='topright', skip_which='upper')
    ax.xaxis.set_major_formatter(QMDFuncFormatter(format_fn, labels))
    ax.yaxis.set_major_formatter(QMDFuncFormatter(format_fn, labels))
    plt.xticks(rotation=90)

    # savefigs(expdire, "EXP_CompareModels_BFhinton"+mytimestamp+".pdf")

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')
    plt.show()
    
    
    
    
###### Tree diagram #####

def available_position_list(max_this_branch, max_any_branch):   
    # Used to get a list of positions to place nodes centrally 
    N = 2*max_any_branch - 1
    all_nums = list(range(N))
    evens = [a for a in all_nums if a%2==0]
    odds = [a for a in all_nums if a%2!=0]    
    
    diff = max_any_branch-max_this_branch 
    if diff%2==0:
        all_positions = evens
        even_odd = 'even'
    else:
        all_positions = odds
        even_odd = 'odd'

    if diff > 1:
        if even_odd=='even':
            to_cut = int(diff/2)
            available_positions = all_positions[to_cut:-to_cut]
        else:
            to_cut = int((diff)/2)
            available_positions = all_positions[to_cut:-to_cut]
    else:
        available_positions = all_positions
        
    return available_positions

def adjacent_branch_test(qmd, mod1, mod2):
    mod_a = qmd.reducedModelInstanceFromID(mod1).Name
    mod_b = qmd.reducedModelInstanceFromID(mod2).Name
    br_a = qmd.pullField(name=mod_a, field='branchID')
    br_b = qmd.pullField(name=mod_b, field='branchID')
       
    diff = br_a - br_b
    if diff in [-1, 0, 1]:
        return True
    else:
        return False
def plotTreeDiagram(qmd, modlist=None, save_to_file=None, only_adjacent_branches=True):
    plt.clf()
    plt.figure(figsize=(16,11))
    G=nx.Graph()
    losing_node_colour = 'r'
    branch_champ_node_colour = 'b'
    overall_champ_node_colour = 'g'
    positions = {}
    labels = {}
    edges_weights = {}
    branch_x_filled = {}
    branch_mod_count = {}
    node_colours = {}
    
    
    max_branch_id = qmd.HighestBranchID
    max_mod_id = qmd.HighestModelID
    if modlist is None:
        modlist = range(max_mod_id)
    for i in range(max_branch_id+1):
        branch_x_filled[i] = 0
        branch_mod_count[i] =  0 

    for i in modlist:
        G.add_node(i)
        node_colours[i] = losing_node_colour
        mod = qmd.reducedModelInstanceFromID(i)
        name = mod.Name
        branch=qmd.pullField(name=name, field='branchID')
        branch_mod_count[branch] += 1
        latex_term = mod.LatexTerm
        labels[i] = latex_term

    most_models_per_branch = max(branch_mod_count.values())
    
    for i in modlist:
        mod = qmd.reducedModelInstanceFromID(i)
        name = mod.Name
        branch=qmd.pullField(name=name, field='branchID')
        num_models_this_branch = branch_mod_count[branch]
        pos_list = available_position_list(num_models_this_branch, most_models_per_branch)
        branch_filled_so_far = branch_x_filled[branch]
        branch_x_filled[branch]+=1
        
        x_pos = pos_list[branch_filled_so_far]
        y_pos = branch
        positions[i] = (x_pos, y_pos)

    # set node colour based on whether that model won a branch 
    for b in list(qmd.BranchChampions.values()):
        node_colours[b] = branch_champ_node_colour
    node_colours[qmd.ChampID] = overall_champ_node_colour
    
    edges = []
    for a in modlist:
        for b in modlist:
            if adjacent_branch_test(qmd, a, b) or only_adjacent_branches==False:
                if a!=b:
                    unique_pair = DataBase.unique_model_pair_identifier(a,b)
                    if unique_pair not in edges and unique_pair in qmd.BayesFactorsComputed:
                        edges.append(unique_pair)
    edge_tuples = []
    weights = []
    for pair in edges:
        mod_ids = pair.split(",")
        pair_tuple=tuple([int(s) for s in mod_ids])
        pair_bf = qmd.AllBayesFactors[float(mod_ids[0])][float(mod_ids[1])][-1]
        weights.append(pair_bf)
        edge_tuples.append(pair_tuple)

    edge_tuples = tuple(edge_tuples)
    weights = np.log10(weights)
    weights = tuple(weights)
    n_colours = tuple(node_colours.values())
    
    nx.draw_networkx(
        G, 
        labels=labels, 
        pos=positions, 
        width=5,
        k=2, #node spacing
        arrows=True,
        arrowstyle='->',
        node_size=5000,
        linewidth=5,
        node_shape='8',
        node_color=n_colours,
        edgelist=edge_tuples,
        edge_color=weights, 
        edge_cmap=plt.cm.Spectral,
    )  
    
    edges_for_cmap = nx.draw_networkx_edges(G,pos=positions,edgelist=edge_tuples, edge_color=weights,width=4,edge_cmap=plt.cm.Spectral)
       
    plt.tight_layout()
    plt.gca().invert_yaxis() # so branch 0 on top
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel('Branch')
    plt.colorbar(edges_for_cmap)
    plt.title('Tree Diagram for QMD')

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')

        
        
       
def BayesFactorsCSV(qmd, save_to_file, names_ids='latex'):

    import csv
    fields = ['ID', 'Name']
    if names_ids=='latex':
        names = [DataBase.latex_name_ising(qmd.ModelNameIDs[i]) for i in range(qmd.HighestModelID)]
    elif names_ids=='nonlatex':
        names = [qmd.ModelNameIDs[i] for i in range(qmd.HighestModelID)]
    elif names_ids=='ids':
        names=range(qmd.HighestModelID)
    else:
        print("BayesFactorsCSV names_ids must be latex, nonlatex, or ids.")

    fields.extend(names)

    with open(save_to_file, 'w') as csvfile:

        fieldnames = fields
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(qmd.HighestModelID):
            model_bf = {}
            for j in qmd.AllBayesFactors[i].keys():
                if names_ids=='latex':
                    other_model_name = DataBase.latex_name_ising(qmd.ModelNameIDs[j])
                elif names_ids=='nonlatex':
                    other_model_name = qmd.ModelNameIDs[j]
                elif names_ids=='ids':
                    other_model_name = j
                model_bf[other_model_name] = qmd.AllBayesFactors[i][j][-1]

            if names_ids=='latex':
                model_bf['Name'] = DataBase.latex_name_ising(qmd.ModelNameIDs[i])
            else:
                model_bf['Name'] = qmd.ModelNameIDs[i]
            model_bf['ID'] = i
            writer.writerow(model_bf)
     
        
