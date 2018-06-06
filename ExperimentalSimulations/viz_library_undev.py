import numpy as np
import os, sys

import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import transforms
from matplotlib import ticker

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap
from matplotlib.patches import FancyArrowPatch, Circle

import networkx as nx


sys.path.append(os.path.join("..", "Libraries","QML_lib"))
import DataBase 

####################################################
########## Bayes Factors manipulation ########
####################################################


def BayesFactorfromLogL(LogLikelihood1, LogLikelihood2):
    BayesFactorValue = np.expm1(LogLikelihood1-LogLikelihood2)+1
    return(BayesFactorValue)
    
    
    

def BayF_DictToMatrix(ModelNames, AllBayesFactors, StartBayesFactors=None):
    
    size = len(ModelNames)
    Bayf_matrix = np.zeros([size,size])
    
    for i in range(size):
        for j in range(size):
            if j > i:
                Bayf_matrix[i,j] = AllBayesFactors[
                    ModelNames[i]+" VS "+ ModelNames[j]]
    
            elif j<i and (StartBayesFactors is not None):
                Bayf_matrix[i,j] = StartBayesFactors[
                    ModelNames[i]+" VS "+ ModelNames[j]]
            
    
    return Bayf_matrix
    
    
def KlogL_to_BF(ModelNames, KLogTotLikelihoods):
        
        lst=np.arange(len(ModelNames))
        outlst = np.empty(0)
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i is not j:
                    outlst = np.append(outlst, np.array([lst[i],lst[j]]) )
        
        BayesFactorNames = []
        BayesFactorDictionary=outlst
        BayesFactorsList = []

        for i in range(int(len(outlst)/2)):
            #first two numbers are telling us which models in the ModelsList we are comparing the third number is the Bayes factorvalue for the two models under consideration
            #print('Iteration'+str(i)+' gives '+str(int(outlst[2*i]))+' and '+str(int(outlst[2*i+1])))
            BayesFactorNames.append("")
            BayesFactorNames[-1]= ModelNames[int(outlst[2*i])]+" VS "+ModelNames[int(outlst[2*i+1])]
            
            BayesFactorsList.append(BayesFactorfromLogL(KLogTotLikelihoods[int(outlst[2*i])], KLogTotLikelihoods[int(outlst[2*i+1])]) )
        
        return {key:value for key, value in zip(BayesFactorNames,BayesFactorsList)}
        




####################################################
########## I/O definitions ########
####################################################        
        
         
def savefigs(dire, thisfile):
    """
    dire: is the directory where the file will be stored, non terminated by slashes
        e.g. "C:\\mydirectory"
    thisfile: is the name of the file with extension, without slashes
        e.g. "myfile.pdf"
    """
    if not os.path.exists(dire):
        os.makedirs(dire)
    outpath = os.path.normpath(dire+"/"+thisfile)
    plt.savefig(outpath, bbox_inches='tight')
    print(outpath)
    


####################################################
########## Hinton Diagrams ########
####################################################


    
    
# TODO: Add yutils.mpl._coll to mpltools and use that for square collection.
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


def hinton(inarray, max_value=None, use_default_ticks=True, skip_diagonal = True, grid = True):
    """Plot Hinton diagram for visualizing the values of a 2D array.

    Plot representation of an array with positive and negative values
    represented by white and black squares, respectively. The size of each
    square represents the magnitude of each value.

    Unlike the hinton demo in the matplotlib gallery [1]_, this implementation
    uses a RegularPolyCollection to draw squares, which is much more efficient
    than drawing individual Rectangles.

    .. note::
        This function inverts the y-axis to match the origin for arrays.

    .. [1] http://matplotlib.sourceforge.net/examples/api/hinton_demo.html
plt.sh
    Parameters
    ----------
    inarray : array
        Array to plot.
    max_value : float
        Any *absolute* value larger than `max_value` will be represented by a
        unit square.
    use_default_ticks: boolean
        Disable tick-generation and generate them outside this function.
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
            
            squares = SquareCollection(sizes=circle_areas,
                                       offsets=xy, transOffset=ax.transData,
                                       facecolor=color, edgecolor=color)
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


        
        
####################################################
########## Radar Plotting ########
####################################################



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
        
        
        
        
        
def radar_factory(num_vars, frame='circle'):
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
            self.set_thetagrids(np.degrees(theta), labels, fontsize = fontsize, frac=frac)

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
    
    

####################################################
########## Graph maniuplation ########
####################################################   

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

    
def qmdclassTOnxobj(qmd, modlist=None, directed=True, only_adjacent_branches=True):
    
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
        latex_term = mod.LatexTerm[1:]
        
        G.add_node(i)
        G.node[i]['label'] = latex_term
        G.node[i]['status'] = 0.2

    # Set x-coordinate for each node based on how many nodes are on that branch (y-coordinate)
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
        G.node[i]['pos'] = (x_pos, y_pos)

    # set node colour based on whether that model won a branch 
    """
    for b in list(qmd.BranchChampions.values()):
        if b in modlist:
            G.node[b]['status'] = 0.4
        G.node[qmd.ChampID]['status'] = 0.6
    """
    edges = []
    for a in modlist:
        for b in modlist:
            is_adj = adjacent_branch_test(qmd, a, b)
            if is_adj or not only_adjacent_branches:
                if a!=b:
                    unique_pair = DataBase.unique_model_pair_identifier(a,b)
                    if unique_pair not in edges and unique_pair in qmd.BayesFactorsComputed:
                        edges.append(unique_pair)
                        vs = [int(stringa) for stringa in unique_pair.split(',')]
                        
                        thisweight = np.log10(qmd.AllBayesFactors[float(vs[0])][float(vs[1])][-1])
                        
                        if thisweight < 0:
                            thisweight = - thisweight # flip negative valued edges and move them to positive
                            flipped = True
                            G.add_edge(vs[1], vs[0], weight=thisweight, flipped = flipped, adj = is_adj)
                        else:
                            flipped = False
                            G.add_edge(vs[0], vs[1], weight=thisweight, flipped = flipped, adj = is_adj)
    
    return G
    
    
def plotQMDTree(qmd, save_to_file=None, only_adjacent_branches=True, id_labels=False, modlist=None):
    G = qmdclassTOnxobj(qmd, only_adjacent_branches=only_adjacent_branches, modlist=modlist)
    plotTreeDiagram(G, n_cmap = plt.cm.pink_r, e_cmap = plt.cm.viridis_r, arrow_size = 0.1,
                    nonadj_alpha = 0.1, e_alphas = [], 
                    label_padding = 0.4, pathstyle="curve", id_labels=id_labels, save_to_file=save_to_file)
    
    
    
def plotTreeDiagram(G, n_cmap, e_cmap, 
                    e_alphas = [], nonadj_alpha=0.1, label_padding = 0.4, 
                    arrow_size = 0.02, pathstyle = "straight", id_labels = False, save_to_file=None):
    plt.clf()
    plt.figure(figsize=(6,11))   
    
    directed  = nx.is_directed(G)

    if int(nx.__version__[0])>=2: 
        list_of_edges = list(G.edges(data=True))
    
    
    edge_tuples = tuple( G.edges() )
    
    positions = dict( zip( G.nodes(), tuple(  [prop['pos'] for (n,prop) in G.nodes(data=True)]  ) ))
    # n_colours = tuple(  [prop['color'] for (n,prop) in G.nodes(data=True)]  ) 
    n_colours = tuple( [ n_cmap(prop['status']) for (n,prop) in G.nodes(data=True)]   )
    
    
    label_positions = []   
    if id_labels is True:
        labels = dict( zip( G.nodes(), tuple(  [n for (n,prop) in G.nodes(data=True)]  ) ))
        for key in positions.keys():
            label_positions.append( tuple( np.array(positions[key])- np.array([0., 0.]) ) )
    else:
        labels = dict( zip( G.nodes(), tuple(  [prop['label'] for (n,prop) in G.nodes(data=True)]  ) ))  
        for key in positions.keys():
            label_positions.append( tuple( np.array(positions[key])- np.array([0., label_padding]) ) )
    
    label_positions = dict(zip( positions.keys(), tuple(label_positions) ))
     
    
    

    if len(e_alphas) == 0: 
        for idx in range(len(edge_tuples)):
            e_alphas.append(  0.8 if list_of_edges[idx][2]["adj"] else nonadj_alpha )
    
    weights = tuple( [prop['weight'] for (u,v,prop) in list_of_edges] )

    nx.draw_networkx_nodes(
        G, with_labels = False, # labels=labels, 
        pos=positions, 
        k=1.5, #node spacing
        node_size=700, #node_shape='8',
        node_color = n_colours
    )  
    
    # edges_for_cmap = nx.draw_networkx_edges(G, width = 3,  pos=positions, arrows=True, arrowstyle='->', edgelist=edge_tuples, edge_color= weights, edge_cmap=plt.cm.Spectral)
    edges_for_cmap = draw_networkx_arrows(G, edgelist=edge_tuples, pos=positions, arrows=True, 
        arrowstyle='->', width = arrow_size,  pathstyle=pathstyle,
        alphas = e_alphas, edge_color= weights, edge_cmap=e_cmap)
    
    nx.draw_networkx_labels(G, label_positions, labels)
    plt.tight_layout()
    
    plt.gca().invert_yaxis() # so branch 0 on top
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel('Branch')
    
    xmin = min( np.array(list(label_positions.values()))[:,0] )
    xmax = max( np.array(list(label_positions.values()))[:,0] )
    plt.xlim(xmin -0.8, xmax +0.8)
    
    plt.colorbar(edges_for_cmap, orientation="horizontal", pad= 0, label=r'$\log_{10}$ Bayes factor') # DONE - negative weights are unaccetpable for directed graphs, as they simply mean a flipped edge

    # plt.colorbar(n_colours)

    plt.title('Tree Diagram for QMD')
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')
    
    
def draw_networkx_arrows(G, pos,
                        edgelist=None,
                        nodedim = 0.,
                        width=0.02,
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

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None
        
    if len(alphas)<len(edgelist):
        alphas = np.repeat(alphas, len(edgelist))

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    
    

    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

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
            raise ValueError('edge_color must consist of either color names or numbers')
    else:
        if cb.is_string_like(edge_color) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color), )
        else:
            raise ValueError('edge_color must be a single color or list of exactly m colors where m is the number or edges')

    edge_collection = collections.LineCollection(edge_pos,
                                     colors=edge_colors,
                                     linewidths=lw
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
                    width= width, head_width = 5*width, overhang = -5*0.02/width, length_includes_head=True, 
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

                e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                                    arrowstyle='-|>',
                                    connectionstyle='arc3,rad=%s'%rad,
                                    mutation_scale=10.0,
                                    lw=10,
                                    alpha=alphas[idx],
                                    color=edge_cmap(edge_color[idx]))
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
   
    
####################################################
########## Utilities for plotting ########
####################################################    
    
def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)
        
        
        

def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts
    
    
    
    
    
def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    # Copied from https://github.com/joferkington/oost_paper_code in
    # accordance with its license agreement.
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    :param cov: The 2x2 covariance matrix to base the ellipse on.
    :param pos: The location of the center of the ellipse. Expects a 2-element
        sequence of ``[x0, y0]``.
    :param nstd: The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.

    :return: A matplotlib ellipse artist.
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]


    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    plt.add_artist(ellip)
    return ellip
    
    
    
