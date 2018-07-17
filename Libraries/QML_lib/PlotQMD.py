import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import ticker
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.ticker import Formatter
from matplotlib import colors as mcolors


import random
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap
from matplotlib.patches import FancyArrowPatch, Circle, ArrowStyle
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

#from QMD import  *
#from QML import *
import DataBase
import Evo as evo

#### Hinton Diagram ####

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
            print("Term",t,"doesn't belong to rotations, \
                Hartree-Fock or transverse."
            )
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

def ExpectationValuesTrueSim(qmd, model_ids=None, champ=True, 
    max_time=3, t_interval=0.01, experimental_measurements_dict=None,
    use_experimental_data=False, upper_x_lim=None,
    save_to_file=None
):
    import random
    if model_ids is None and champ == True:
        model_ids = [qmd.ChampID]
    elif model_ids is not None and champ == True:

        if type(model_ids) is not list:
            model_ids = [model_ids]
        if qmd.ChampID not in model_ids:
            model_ids.append(qmd.ChampID)

    if qmd.ChampID in model_ids and champ==False:
        model_ids.remove(qmd.ChampID)
    
    probe_id = random.choice(range(qmd.NumProbes))
    # names colours from
    # https://matplotlib.org/2.0.0/examples/color/named_colors.html
    true_colour =  colors.cnames['lightsalmon'] #'b'
    true_colour =  'r' #'b'
    
    champion_colour =  colors.cnames['cornflowerblue'] #'r'
    sim_colours = ['g', 'c', 'm', 'y', 'k']

    plt.clf()
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Expectation Value')

    
    if (experimental_measurements_dict is not None) and (use_experimental_data==True):
        times = sorted(list(experimental_measurements_dict.keys()))
        true_expec_values = [
            experimental_measurements_dict[t] for t in times
        ]
    else:
        times = np.arange(0, max_time, t_interval)
        true = qmd.TrueOpName
        true_op = DataBase.operator(true)
        true_params = qmd.TrueParamsList
        true_ops = true_op.constituents_operators
        true_ham = np.tensordot(true_params, true_ops, axes=1)
        true_dim = true_op.num_qubits
        true_probe = qmd.ProbeDict[(probe_id,true_dim)]
        true_expec_values = [evo.expectation_value(ham=true_ham, t=t, 
            state=true_probe) for t in times
        ]
    plt.scatter(times, true_expec_values, label='True Expectation Value',
        marker='o', s=8, color = true_colour
    )

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
        sim_expec_values = [evo.expectation_value(ham=sim_ham, t=t,
            state=sim_probe) for t in times
        ]

        if mod_id == qmd.ChampID:
            models_branch = ChampionsByBranch[mod_id]
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
    
    if upper_x_lim is not None:
        plt.xlim(0,upper_x_lim)
    else:
        plt.xlim(0, max(times))
    
    if extra_lgd:
        lgd_spec=ax.legend(special_handles, special_labels, 
            loc='upper center', bbox_to_anchor=(1, 1),fancybox=True,
            shadow=True, ncol=1
        )
        lgd_new=ax.legend(new_handles, new_labels, 
            loc='upper center', bbox_to_anchor=(1.15, 0.75),
            fancybox=True, shadow=True, ncol=1
        )
        plt.gca().add_artist(lgd_spec)
    else:
        lgd_spec=ax.legend(special_handles, special_labels,
            loc='upper center', bbox_to_anchor=(1, 1),fancybox=True, 
            shadow=True, ncol=1
        )
        
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')

    
def ExpectationValuesQHL_TrueModel(qmd, 
    max_time=3, t_interval=0.01, experimental_measurements_dict=None,
    use_experimental_data=False, upper_x_lim=None,
    save_to_file=None
):
    model_ids = [qmd.TrueOpModelID]
    probe_id = random.choice(range(qmd.NumProbes))
    # names colours from
    # https://matplotlib.org/2.0.0/examples/color/named_colors.html
    true_colour =  colors.cnames['lightsalmon'] #'b'
    true_colour =  'r' #'b'
    
    champion_colour =  colors.cnames['cornflowerblue'] #'r'
    sim_colours = ['g', 'c', 'm', 'y', 'k']

    plt.clf()
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Expectation Value')

    
    if (experimental_measurements_dict is not None) and (use_experimental_data==True):
        times = sorted(list(experimental_measurements_dict.keys()))
        true_expec_values = [
            experimental_measurements_dict[t] for t in times
        ]
    else:
        times = np.arange(0, max_time, t_interval)
        true = qmd.TrueOpName
        true_op = DataBase.operator(true)
        true_params = qmd.TrueParamsList
        true_ops = true_op.constituents_operators
        true_ham = np.tensordot(true_params, true_ops, axes=1)
        true_dim = true_op.num_qubits
        true_probe = qmd.ProbeDict[(probe_id,true_dim)]
        true_expec_values = [evo.expectation_value(ham=true_ham, t=t, 
            state=true_probe) for t in times
        ]
    plt.scatter(times, true_expec_values, label='True Expectation Value',
        marker='o', s=8, color = true_colour
    )

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
        sim_expec_values = [evo.expectation_value(ham=sim_ham, t=t,
            state=sim_probe) for t in times
        ]

        if mod_id == qmd.TrueOpModelID:
            sim_label = 'Simulated Model'
            sim_col = champion_colour
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

    special_terms = ['True Expectation Value', 'Champion Model', 'Simulated Model']

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
    
    if upper_x_lim is not None:
        plt.xlim(0,upper_x_lim)
    else:
        plt.xlim(0, max(times))
    
    if extra_lgd:
        lgd_spec=ax.legend(special_handles, special_labels, 
            loc='upper center', bbox_to_anchor=(1, 1),fancybox=True,
            shadow=True, ncol=1
        )
        lgd_new=ax.legend(new_handles, new_labels, 
            loc='upper center', bbox_to_anchor=(1.15, 0.75),
            fancybox=True, shadow=True, ncol=1
        )
        plt.gca().add_artist(lgd_spec)
    else:
        lgd_spec=ax.legend(special_handles, special_labels,
            loc='upper center', bbox_to_anchor=(1, 1),fancybox=True, 
            shadow=True, ncol=1
        )
        
    plt.title(str("QHL test for " + 
        str(DataBase.latex_name_ising(qmd.TrueOpName)))
    )        
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


        
def hinton(inarray, max_value=None, use_default_ticks=True, 
    skip_diagonal = True, skip_which = None, grid = True, white_half = 0.,
    where_labels = 'bottomleft'
):
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
        whether to plot both upper and lower triangular 
        matrix or just one of them
    grid: Boolean
        to remove the grid from the plot
    white_half : float
        adjust the size of the white "coverage" of the "skip_which"
        part of the diagram
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

    
# static coloring property definitions
losing_node_colour = 'r'
branch_champ_node_colour = 'b'
overall_champ_node_colour = 'g'    

    
def qmdclassTOnxobj(qmd, modlist=None, directed=True,
    only_adjacent_branches=True
):
    
    if directed:
        G=nx.DiGraph()
    else:
        G=nx.Graph()
        
    positions = {}
    branch_x_filled = {}
    branch_mod_count = {}

    
    max_branch_id = qmd.HighestBranchID
    max_mod_id = qmd.HighestModelID
    if modlist is None:
        modlist = range(max_mod_id)
    for i in range(max_branch_id+1):
        branch_x_filled[i] = 0
        branch_mod_count[i] =  0 

    for i in modlist:
        mod = qmd.reducedModelInstanceFromID(i)
        name = mod.Name
        branch=qmd.pullField(name=name, field='branchID')
        branch_mod_count[branch] += 1
        latex_term = mod.LatexTerm
        
        G.add_node(i)
        G.node[i]['label'] = latex_term
        G.node[i]['status'] = 0.2
        G.node[i]['info'] = 'Non-winner'

    # Set x-coordinate for each node based on how many nodes 
    # are on that branch (y-coordinate)
    most_models_per_branch = max(branch_mod_count.values())
    for i in modlist:
        mod = qmd.reducedModelInstanceFromID(i)
        name = mod.Name
        branch=qmd.pullField(name=name, field='branchID')
        num_models_this_branch = branch_mod_count[branch]
        pos_list = available_position_list(num_models_this_branch,
            most_models_per_branch
        )
        branch_filled_so_far = branch_x_filled[branch]
        branch_x_filled[branch]+=1
        
        x_pos = pos_list[branch_filled_so_far]
        y_pos = branch
        positions[i] = (x_pos, y_pos)
        G.node[i]['pos'] = (x_pos, y_pos)

    # set node colour based on whether that model won a branch 
    for b in list(qmd.BranchChampions.values()):
        if b in modlist:
            G.node[b]['status'] = 0.45
            G.node[b]['info'] = 'Branch Champion'
            
    G.node[qmd.ChampID]['status'] = 0.9
    G.node[qmd.ChampID]['info'] = 'Overall Champion'

    edges = []
    for a in modlist:
        for b in modlist:
            is_adj = adjacent_branch_test(qmd, a, b)
            if is_adj or not only_adjacent_branches:
                if a!=b:
                    unique_pair = DataBase.unique_model_pair_identifier(a,b)
                    if ((unique_pair not in edges)
                        and (unique_pair in qmd.BayesFactorsComputed)
                    ):
                        edges.append(unique_pair)
                        vs = [int(stringa) for stringa 
                            in unique_pair.split(',')
                        ]
                        
                        thisweight = np.log10(
                            qmd.AllBayesFactors[float(vs[0])][float(vs[1])][-1]
                        )
                        
                        if thisweight < 0:
                            # flip negative valued edges and move 
                            # them to positive
                            thisweight = - thisweight 
                            flipped = True
                            G.add_edge(vs[1], vs[0], 
                                weight=thisweight, flipped=flipped, 
                                adj=is_adj
                            )
                        else:
                            flipped = False
                            G.add_edge(vs[0], vs[1], 
                                weight=thisweight, flipped=flipped, 
                                adj=is_adj
                            )
    return G
    
    
def plotQMDTree(qmd, save_to_file=None, 
                only_adjacent_branches=True, id_labels=False,
                modlist=None):

    G = qmdclassTOnxobj(qmd, 
        only_adjacent_branches=only_adjacent_branches, 
        modlist=modlist)
    
    plotTreeDiagram(G, n_cmap = plt.cm.pink_r, 
        e_cmap = plt.cm.Blues, arrow_size = 8.0,
        nonadj_alpha = 0.1, e_alphas = [], 
        label_padding = 0.4, pathstyle="curve",
        id_labels=id_labels, save_to_file=save_to_file)
    
   

def plotTreeDiagram(G, n_cmap, e_cmap, 
                    e_alphas = [], nonadj_alpha=0.1, 
                    label_padding = 0.4, 
                    arrow_size = 0.02, widthscale=1.0,
                    pathstyle = "straight", id_labels = False,
                    save_to_file=None
):
    plt.clf()
    plt.figure(figsize=(6,11))   
    
    directed  = nx.is_directed(G)

    if int(nx.__version__[0])>=2: 
        list_of_edges = list(G.edges(data=True))
    
    
    edge_tuples = tuple( G.edges() )
    
    positions = dict( zip( G.nodes(), tuple(  [prop['pos'] for
        (n,prop) in G.nodes(data=True)]  ) )
    )
    n_colours = tuple( [ n_cmap(prop['status']) for (n,prop) 
        in G.nodes(data=True)]   
    )
    
    label_positions = []   
    if id_labels is True:
        labels = dict( zip( G.nodes(), tuple(  [n for (n,prop) in 
            G.nodes(data=True)]  ) )
        )
        for key in positions.keys():
            label_positions.append( tuple( np.array(positions[key]) -
            np.array([0., 0.]) ) 
        )
    else:
        labels = dict( zip( G.nodes(), tuple(  [prop['label'] for 
            (n,prop) in G.nodes(data=True)]  ) )
        )  
        for key in positions.keys():
            label_positions.append( tuple( np.array(positions[key])- 
                np.array([0., label_padding]) ) 
        )
    
    label_positions = dict(zip( positions.keys(), tuple(label_positions) ))
     
    
    if len(e_alphas) == 0: 
        for idx in range(len(edge_tuples)):
            e_alphas.append(  
                0.8 if list_of_edges[idx][2]["adj"] 
                else nonadj_alpha 
            )
    
    weights = tuple( [prop['weight'] for (u,v,prop) in list_of_edges] )

    nx.draw_networkx_nodes(
        G, with_labels = False, # labels=labels, 
        pos=positions, 
        k=1.5, #node spacing
        width=None, 
        node_size=700, #node_shape='8',
        node_color = n_colours
    )  
    
    edges_for_cmap = draw_networkx_arrows(G, edgelist=edge_tuples,
        pos=positions, arrows=True, 
        arrowstyle='->', width = arrow_size, widthscale=widthscale,
        pathstyle=pathstyle, alphas = e_alphas, edge_color= weights,
        edge_cmap=e_cmap, edge_vmin=0.8, edge_vmax=0.85
    )
    
    nx.draw_networkx_labels(G, label_positions, labels)
    plt.tight_layout()
    
    plt.gca().invert_yaxis() # so branch 0 on top
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel('Branch')
    
    xmin = min( np.array(list(label_positions.values()))[:,0] )
    xmax = max( np.array(list(label_positions.values()))[:,0] )
    plt.xlim(xmin -0.8, xmax +0.8)
    
    plt.colorbar(edges_for_cmap, orientation="horizontal", 
        pad= 0, label=r'$\log_{10}$ Bayes factor'
    ) 

    nodes=list(G.nodes)
    distinct_status=[]
    labels=[]
    handles=[]
    for n in nodes:
        stat = G.nodes[n]['status'] # only for status not yet represented in legend
        if stat not in distinct_status:
            distinct_status.append(stat)
            # node colour encodes either number wins or branch champion
            info=str(G.nodes[n]['info']) 
            col = tuple( n_cmap(G.nodes[n]['status']) )

            handles.append(mpatches.Patch(color=col))
            labels.append(info)
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))


    lgd_handles=[]
    
    if 'Branch Champion' in labels:
        legend_title='Champion Type'
    else:
        legend_title='# QMD wins'
    plt.legend(handles, labels, title=legend_title)

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def colour_dicts_from_win_count(winning_count, min_colour_value=0.1):
    max_wins=max(list(winning_count.values()))
    min_wins=min(list(winning_count.values()))
    min_col = min_colour_value
    max_col = 0.9

    win_count_vals = list(winning_count.values())
    win_count_vals.append(0)
    distinct_win_vals=sorted(list(set(win_count_vals)))
    num_colours = len(distinct_win_vals)
    col_space = np.linspace(min_col, max_col, num_colours)
    colour_by_win_count = {}
    colour_by_node_name = {}
    all_models=list(
        ising_terms_full_list(return_branch_dict='latex_terms').keys()
    )

    for k in range(num_colours):
        colour_by_win_count[k] = col_space[k]

    for k in all_models:
        try:
            num_wins = winning_count[k]
        except:
            num_wins = 0
        idx = distinct_win_vals.index(num_wins)
        colour_by_node_name[k] = colour_by_win_count[idx]
    return colour_by_node_name, colour_by_win_count


def cumulativeQMDTreePlot(
        cumulative_csv, 
        wins_per_mod, 
        avg='means',
        only_adjacent_branches=True,
        directed=True,
        save_to_file=None
    ):
    
    import networkx as nx
    import copy
    means, medians, counts = multiQMDBayes(cumulative_csv)
    if avg=='means':
        bayes_factors = means #medians
    elif avg=='medians':
        bayes_factors = medians

    modlist = ising_terms_full_list()
    term_branches = (
        ising_terms_full_list(return_branch_dict='latex_terms')       
    ) 

    pair_freqs={}
    for c in list(counts.keys()):
        for k in list(counts[c].keys()):
            this_edge=(c,k)
            pair_freqs[this_edge]=counts[c][k]


    if directed:
        G=nx.DiGraph()
    else:
        G=nx.Graph()

    positions = {}
    branch_x_filled = {}
    branch_mod_count = {}

#    max_branch_id = qmd.HighestBranchID # TODO: get this number without access to QMD class instance
 #   max_mod_id = qmd.HighestModelID
    max_branch_id = 9 # TODO: this is hardcoded - is there an alternative?
    max_mod_id = 17
    
    

    for i in range(max_branch_id+1):
        branch_x_filled[i] = 0
        branch_mod_count[i] =  0 

    colour_by_node_name, colour_by_count = (
        colour_dicts_from_win_count(wins_per_mod, 0.4)
    )
    min_colour = min(list(colour_by_node_name.values()))
    
    for m in modlist:
        branch = term_branches[m]
        branch_mod_count[branch]+=1
        G.add_node(m)
        G.nodes[m]['label']=str(m)
        try:
            G.nodes[m]['status']=colour_by_node_name[m]
            G.nodes[m]['wins']=wins_per_mod[m]
            G.nodes[m]['info']=wins_per_mod[m]
            
            
        except:
            G.nodes[m]['status']=min_colour
            G.nodes[m]['info']=0
            
            
    max_num_mods_any_branch=max(list(branch_mod_count.values()))
    for m in modlist:
        branch = term_branches[m]
        num_mods_this_branch = branch_mod_count[branch]
        pos_list = available_position_list(num_mods_this_branch,
            max_num_mods_any_branch
        )
        branch_filled_so_far = branch_x_filled[branch]
        branch_x_filled[branch]+=1

        x_pos = pos_list[branch_filled_so_far]
        y_pos = branch
        positions[i] = (x_pos, y_pos)
        G.node[m]['pos'] = (x_pos, y_pos)

    edges = []
    edge_frequencies=[]
    max_frequency = max(list(pair_freqs.values()))
    for a in modlist:
        remaining_modlist = modlist[modlist.index(a)+1:]
        for b in remaining_modlist:
            is_adj = global_adjacent_branch_test(a, b, term_branches)
            if is_adj or not only_adjacent_branches:
                if a!=b:
                    pairing = (a,b)
                    frequency = pair_freqs[pairing]/max_frequency
                    edges.append(pairing)
                    edge_frequencies.append(frequency)
                    vs = [a,b]

                    try:
                        thisweight = np.log10(bayes_factors[a][b])
                    except:
                        thisweight=0 #TODO is this right?

                    if thisweight < 0:  
                        # flip negative valued edges and move them to positive
                        thisweight = - thisweight 
                        flipped = True
                        G.add_edge(b, a, weight=thisweight, flipped=flipped,
                            adj = is_adj, freq=frequency
                        )
                    else:
                        flipped = False
                        G.add_edge(a, b, weight=thisweight, flipped=flipped,
                            adj=is_adj, freq=frequency
                        )
                        
    max_freq = max(edge_frequencies)
    
    freq_scale = 10/max_freq
    edge_f = [i*freq_scale for i in edge_frequencies]
    
    plotTreeDiagram(G,n_cmap = plt.cm.pink_r, e_cmap = plt.cm.Blues, 
                       nonadj_alpha = 0.1, e_alphas = [] , widthscale=10.5, 
                       label_padding = 0.4, pathstyle="curve",
                       arrow_size=None
       )   

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')
    return G, edges, edge_f   
    
def draw_networkx_arrows(G, pos,
                        edgelist=None,
                        nodedim = 0.,
#                        width=0.02,
                        width=1.0,
                        widthscale = 1.0,
                        edge_color='k',
                        style='solid',
                        alphas=1.,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None,
                        ax=None,
                        label=[None],
                        pathstyle='straight',
                        **kwds):

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = G.edges()
        
    if width is None:
        try:
            widthlist = np.array(list(  [(widthscale*prop['freq']) for (u,v,prop) in G.edges(data=True)]  ))
            widthlist = widthscale*widthlist/np.max(widthlist)
            # widthlist = [(a+widthscale*0.1) for a in widthlist] ## this was giving colour to non-existent edges
        except:
#            widthlist = widthscale*0.02
            widthlist = widthscale
            
    else:
        widthlist = width

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None
        
    if len(alphas)<len(edgelist):
        alphas = np.repeat(alphas, len(edgelist))

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    
    if not cb.iterable(widthlist):
        lw = (widthlist,)
    else:
        lw = widthlist

    if not cb.is_string_like(edge_color) \
           and cb.iterable(edge_color) \
           and len(edge_color) == len(edge_pos):
        if np.alltrue([cb.is_string_like(c)
                         for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c)
                                 for c in edge_color])
        elif np.alltrue([not cb.is_string_like(c)
                           for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if np.alltrue([cb.iterable(c) and len(c) in (3, 4)
                             for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of \
                either color names or numbers'
            )
    else:
        if cb.is_string_like(edge_color) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color), )
        else:
            raise ValueError('edge_color must be a single color or \
            list of exactly m colors where m is the number or edges'
        )

    edge_collection = collections.LineCollection(edge_pos,
        colors=edge_colors, linewidths=lw
    )
    edge_collection.set_zorder(1)  # edges go behind nodes

    # ax.add_collection(edge_collection)

    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(np.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()
    

    for n in G:
        c=Circle(pos[n],radius=0.02,alpha=0.5)
        ax.add_patch(c)
        G.node[n]['patch']=c
        x,y=pos[n]
    seen={}

    
    if G.is_directed():
        seen = {}
        
        for idx in range(len(edgelist)):
        
            if not cb.iterable(widthlist):
                lw = widthlist
            else:
                lw = widthlist[idx]
            
            if pathstyle is "straight":
                (src, dst) = edge_pos[idx]
                x1, y1 = src
                x2, y2 = dst
                delta = 0.2
                theta = np.arctan((y2-y1)/(x2-x1))
                # print(theta)
                if x1==x2:
                    dx = x2-x1
                    dy = y2-y1 - np.sign(y2-y1)*delta
                elif y1==y2:
                    dx = x2-x1 - np.sign(x2-x1)*delta
                    dy = y2-y1 
                else:
                    dx = x2-x1 - np.sign(x2-x1)*np.abs(np.cos(theta)*delta)   # x offset
                    dy = y2-y1 - np.sign(y2-y1)*np.abs(np.sin(theta)*delta)   # y offset 
                
                thislabel = None if len(label)<len(edgelist) else label[idx]

                
                
                ax.arrow(
                    x1,y1, dx,dy,
                    facecolor=edge_cmap(edge_color[idx]), alpha = alphas[idx],
                    linewidth = 0, antialiased = True,
                    width= lw, head_width = 5*lw,
                    overhang = -5*0.02/lw, length_includes_head=True, 
                    label=thislabel, zorder=1
                    )
                    
            elif pathstyle is "curve":
                
                (u,v) = edgelist[idx]
                n1=G.node[u]['patch']
                n2=G.node[v]['patch']
                rad=0.1
                
                if (u,v) in seen:
                    rad=seen.get((u,v))
                    rad=(rad+np.sign(rad)*0.1)*-1
                alpha=0.5
                
                kwargs = {'head_width': 5*lw, 
                          #'overhang':-5*0.02/lw,  
                          #'length_includes_head': True
                          }
                          
                          
                arrow_style = ArrowStyle("->", head_length=1.9, head_width=1.9) # Can be accepted by fancy arrow patch to alter arrows

#                arrow_style = ArrowStyle("-[", widthB=2.0, lengthB=1.0) # Can be accepted by fancy arrow patch to alter arrows

                e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
#                                    arrowstyle='-|>',
#                                    capstyle='projecting',
                                    arrowstyle=arrow_style,
                                    connectionstyle='arc3,rad=%s'%rad,
                                    mutation_scale=10.0,
                                    lw=lw,   #AROUND 10 TO BE FEASIBLE
                                    alpha=alphas[idx],
                                    color=edge_cmap(edge_color[idx]),
                            #        **kwargs
                                    )
                seen[(u,v)]=rad
                ax.add_patch(e)
           

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx-minx
    h = maxy-miny
    padx,  pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return edge_collection


### Parameter Estimate Plot ###


def parameterEstimates(qmd, modelID, save_to_file=None):

    mod = qmd.reducedModelInstanceFromID(modelID)
    name = mod.Name
        
    if name not in list(qmd.ModelNameIDs.values()):
        print("True model ", name, " not in studied models", 
            list(qmd.ModelNameIDs.values())
        )
        return False
    terms = DataBase.get_constituent_names_from_name(name)
    num_terms = len(terms) 

    term_positions={}
    param_estimate_by_term = {}

    for t in range(num_terms):
        term_positions[terms[t]] = t 
        term = terms[t]
        param_position = term_positions[term]
        param_estimates = mod.TrackEval[:,param_position]
        param_estimate_by_term[term] = param_estimates    

    colours = ['b','r','g','orange', 'pink', 'grey']
    # TODO use color map as list
    num_epochs = qmd.NumExperiments
    fig = plt.figure()
    ax = plt.subplot(111)

    i=0
    for term in list(param_estimate_by_term.keys()):
        colour = colours[i%len(colours)]
        i+=1
        try:
            y_true = qmd.TrueParamDict[term]
            true_term_latex = DataBase.latex_name_ising(term)
            ax.axhline(y_true, label=str(true_term_latex+ ' True'), color=colour)
        except:
            pass
        y = param_estimate_by_term[term]
        x = range(1,1+len(param_estimate_by_term[term]))
        latex_term = DataBase.latex_name_ising(term)
        ax.scatter(x,y, s=max(1,50/num_epochs), 
            label=str(latex_term + ' Estimate'), color=colour
        )

    plt.xlabel('Epoch')
    plt.ylabel('Parameter Estimate')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title(str("Parameter estimation for model " +  
        DataBase.latex_name_ising(name))
    )

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


### Radar Plot ###

def plotRadar(qmd, modlist, save_to_file=None, plot_title=None):
    from matplotlib import cm as colmap
#    from viz_library_undev import radar_factory # TODO IS THIS THE RIGHT FUNCTION?
    
    labels = [DataBase.latex_name_ising(qmd.ModelNameIDs[l]) for l in modlist]
    size = len(modlist)
    theta = custom_radar_factory(size, frame='polygon') 
    
    fig, ax = plt.subplots(figsize=(12,6), subplot_kw=dict(projection='radar'))
    
#    cmap = colmap.get_cmap('viridis')
    cmap = colmap.get_cmap('RdYlBu')
    colors = [cmap(col) for col in np.linspace(0.1,1,size)]

    required_bayes = {}
    scale = []

    for i in modlist:
        required_bayes[i] = {}
        for j in modlist:
            if i is not j:
                try:
                    val = qmd.AllBayesFactors[i][j][-1]
                    scale.append(np.log10(val))
                except:
                    val = 1.0
                required_bayes[i][j] = val


    [scale_min, scale_max] = [min(scale), max(scale)]
    many_circles = 4
    low_ini = scale_min
    shift_ini = 1
    shift = 6
    ax.set_rgrids(list(shift_ini + 
        np.linspace(low_ini+0.05,0.05,many_circles)), 
        labels = list(np.round(np.linspace(low_ini+0.05,0.05,many_circles), 
        2)), angle=180
    )

    for i in modlist:
        dplot = []
        for j in modlist:
            if i is not j:
                try:
                    bayes_factor = qmd.AllBayesFactors[i][j][-1]
                except: 
                    bayes_factor = 1.0

                log_bayes_factor = np.log10(bayes_factor)
                dplot.append(shift+log_bayes_factor)
            else:
                dplot.append(shift+0.0)
        ax.plot(theta, np.array(dplot), color=colors[int(i%len(colors))],
            linestyle = '--', alpha = 1.
        )
        ax.fill(theta, np.array(dplot), 
            facecolor=colors[int(i%len(colors))], alpha=0.25
        )

    ax.plot(theta, np.repeat(shift, len(labels)), color='black', 
        linestyle = '-', label='BayesFactor=1'
    )
    
    ax.set_varlabels(labels, fontsize=15)
    try:
        ax.tick_params(pad=50)
    except:
        pass
        
    legend = ax.legend(labels, loc=(1.5, .35),
        labelspacing=0.1, fontsize=14)

    if plot_title is not None:
        plt.title(str(plot_title))

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')



    
class IndexLocator(ticker.Locator):

    def __init__(self, max_ticks=10):
        self.max_ticks = max_ticks

    def __call__(self):
        """Return the locations of the ticks."""
        dmin, dmax = self.axis.get_data_interval()
        if dmax < self.max_ticks:
            step = 1
        else:
            step = np.ceil(dmax / self.max_ticks)
        return self.raise_if_exceeds(np.arange(0, dmax, step))
        
        
        
        
        
def custom_radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize = None, frac=1.0):
            self.set_thetagrids(np.degrees(theta), labels, 
                fontsize = fontsize, frac=frac
            )

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts
    


#### Cumulative Bayes CSV and InterQMD Tree plotting ####

def multiQMDBayes(all_bayes_csv):
    import csv, pandas
    cumulative_bayes = pandas.DataFrame.from_csv(all_bayes_csv)
    names=list(cumulative_bayes.keys())

    count_bayes={}
    mod_names= ising_terms_full_list()

    for mod in mod_names:
        count_bayes[mod] = {}
        model_results=cumulative_bayes[mod]
        for comp_mod in mod_names:
            try:
                num_bayes=model_results[comp_mod].count()
            except:
                num_bayes=0
            count_bayes[mod][comp_mod] = num_bayes



    piv = pandas.pivot_table(cumulative_bayes, 
        index='ModelName', values=names, aggfunc=[np.mean, np.median]
    )
    means=piv['mean']
    medians=piv['median']

    b=means.apply(lambda x: x.dropna().to_dict(), axis=1)
    means_dict = b.to_dict()

    c=medians.apply(lambda x: x.dropna().to_dict(), axis=1)
    medians_dict = c.to_dict()        
    
    return means_dict, medians_dict, count_bayes


def updateAllBayesCSV(qmd, all_bayes_csv):
    import os,csv
    
    data = get_bayes_latex_dict(qmd)
    names = list(data.keys())
    fields = ['ModelName']
    fields += names
    all_models= ['ModelName']
    all_models += ising_terms_full_list()
    
    if os.path.isfile(all_bayes_csv) is False:
        with open(all_bayes_csv, 'a+') as bayes_csv:
            writer = csv.DictWriter(bayes_csv, fieldnames=all_models)
            writer.writeheader()
    
    with open(all_bayes_csv, 'a') as bayes_csv:
        writer = csv.DictWriter(bayes_csv, fieldnames=all_models)

        for f in names:
            single_model_dict = data[f]
            single_model_dict['ModelName']=f
            writer.writerow(single_model_dict)


def get_bayes_latex_dict(qmd):
    latex_dict = {}
    for i in list(qmd.AllBayesFactors.keys()):
        mod_a = DataBase.latex_name_ising(qmd.ModelNameIDs[i])
        latex_dict[mod_a] = {}
        for j in list(qmd.AllBayesFactors[i].keys()):
            mod_b = DataBase.latex_name_ising(qmd.ModelNameIDs[j])
            latex_dict[mod_a][mod_b]= qmd.AllBayesFactors[i][j][-1]
    return latex_dict





def ising_terms_full_list(return_branch_dict=None):
    pauli_terms = ['x','y','z']

    branches= {}
    branch_by_term_dict = {}
    for i in range(9):
        branches[i] = []
    
    rotation_terms = []
    hf_terms = []
    transverse_terms = []

    for t in pauli_terms:
        rotation_terms.append(t+'Ti')
        hf_terms.append(t+'T'+t)
        for k in pauli_terms:
            if k>t:
                transverse_terms.append(t+'T'+k)

    ising_terms = []            
    add = 'PP'

    for r in rotation_terms:
        ising_terms.append(r)
        branches[0].append(r)
        branch_by_term_dict[r] = 0
        
    for r in rotation_terms:
        new_terms=[]
        for i in rotation_terms:
            if r<i:
                branches[1].append(r+add+i)
                branch_by_term_dict[r+add+i] = 1
                new_terms.append(r+add+i)
        ising_terms.extend(new_terms)

    full_rotation = add.join(rotation_terms)
    ising_terms.append(full_rotation)
    branches[2].append(full_rotation)
    branch_by_term_dict[full_rotation] = 2
    

    for t in hf_terms:
        new_term = full_rotation+add+t
        branches[3].append(new_term)
        branch_by_term_dict[new_term] = 3
        ising_terms.append(new_term)

    for t in hf_terms:
        for k in hf_terms:
            if t<k:
                dual_hf_term= full_rotation+add+t+add+k
                branches[4].append(dual_hf_term)
                branch_by_term_dict[dual_hf_term] = 4
                ising_terms.append(dual_hf_term)

    for t in hf_terms:
        for l in hf_terms:
            for k in hf_terms:
                if t<k<l:
                    triple_hf = full_rotation + add + t + add + k + add + l
                    branches[5].append(triple_hf)
                    ising_terms.append(triple_hf)
                    branch_by_term_dict[triple_hf] = 5



    for t in transverse_terms:
        transverse_term= triple_hf+add+t
        branches[6].append(transverse_term)
        branch_by_term_dict[transverse_term] = 6
        ising_terms.append(transverse_term)


    for t in transverse_terms:
        for k in transverse_terms:
            if t<k:
                dual_transverse_term= triple_hf+add+t+add+k
                branches[7].append(dual_transverse_term)
                branch_by_term_dict[dual_transverse_term] = 7
                ising_terms.append(dual_transverse_term)

    for t in transverse_terms:
        for l in transverse_terms:
            for k in transverse_terms:
                if t<k<l:
                    triple_hf_term= triple_hf+add+t+add+k+add+l
                    branch_by_term_dict[triple_hf_term] = 8
                    branches[8].append(triple_hf_term)
                    ising_terms.append(triple_hf_term)

    
    latex_terms = [DataBase.latex_name_ising(i) for i in ising_terms]
    
    if return_branch_dict=='branches':
        return branches
    elif return_branch_dict=='terms':
        return branch_by_term_dict

    elif return_branch_dict == 'latex_terms':
        for k in list(branch_by_term_dict.keys()):
            branch_by_term_dict[DataBase.latex_name_ising(k)]=(
                branch_by_term_dict.pop(k)
            )
        return branch_by_term_dict
        
    else:
        return latex_terms

def global_adjacent_branch_test(a,b, term_branches):
    branch_a = term_branches[a]
    branch_b = term_branches[b]
    
    if (branch_a==branch_b) or (branch_a==branch_b+1) or (branch_a==branch_b-1):
        return True
    else:
        return False

    
#### Manipulate QMD output
def BayesFactorsCSV(qmd, save_to_file, names_ids='latex'):

    import csv
    fields = ['ID', 'Name']
    if names_ids=='latex':
        names = [DataBase.latex_name_ising(qmd.ModelNameIDs[i]) for i in 
            range(qmd.HighestModelID)
        ]
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
                    other_model_name = (
                        DataBase.latex_name_ising(qmd.ModelNameIDs[j])
                    )
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
     
        
