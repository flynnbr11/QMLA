import sys,os
sys.path.append(os.path.join("..", "QMD","Libraries","QML_lib"))
import PlotQMD as ptq
import pandas
import numpy as np
from PlotQMD import *



from matplotlib.patches import _Style, _pprint_styles, ArrowStyle




def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
    
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

    cumulative_bayes['index'] = cumulative_bayes.index #REMOVE: included to fix different Pandas version

    piv = pandas.pivot_table(cumulative_bayes, 
        index='index', values=names, aggfunc=[np.mean, np.median] #REVERT: included to fix different Pandas version
    )
    
    
    means=piv['mean']
    medians=piv['median']

    b=means.apply(lambda x: x.dropna().to_dict(), axis=1)
    means_dict = b.to_dict()

    c=medians.apply(lambda x: x.dropna().to_dict(), axis=1)
    medians_dict = c.to_dict()        
    
    return means_dict, medians_dict, count_bayes    
    
    
    
    
    
    
    
    
    
def draw_networkx_arrows(
    G, 
    pos,
    edgelist=None,
    nodedim = 0.,
    width=0.02,    #                        width=0.02, 1.0
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
    **kwds
):
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

    edge_collection = collections.LineCollection(
        edge_pos,
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
    


    max_bayes_value = max(edge_collection.get_clim())
    edge_collection.set_clim(0.5, max_bayes_value)
    # for i in range(len(edgelist)):
    #     print(edgelist[i], ":", edge_color[i])

    for n in G:
        c=Circle(pos[n],radius=0.02,alpha=0.5)
        ax.add_patch(c)
        G.node[n]['patch']=c
        x,y=pos[n]
    seen={}

    # Rescale all weights between 0,1 so cmap can find the appropriate RGB value.
    offset = 0.7
    norm_edge_color = edge_color/max_bayes_value

    # print("all color cmap values", norm_edge_color)

    if G.is_directed():
        seen = {}
        for idx in range(len(edgelist)):
            if not cb.iterable(widthlist):
                lw = widthlist
            else:
                lw = widthlist[idx]
            
            arrow_colour =  edge_cmap(norm_edge_color[idx])

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
                    facecolor=arrow_colour, 
                    alpha = alphas[idx],
                    linewidth = 0, 
                    antialiased = True,
                    width = lw, 
                    head_width = 5*lw,
                    overhang = -5*0.02/lw,
                    length_includes_head=True, 
                    label=thislabel, zorder=1
                )
                    
            elif pathstyle is "curve":
                
                (u,v) = edgelist[idx]
                # (u,v,prop) = prop['weight'] for  in list_of_edges
                # flipped = G.edge[(u,v)]
                
                winner = G.edges[(u,v)]['winner']
                loser = G.edges[(u,v)]['loser']
                n1=G.node[loser]['patch']
                n2=G.node[winner]['patch']

                # n1=G.node[u]['patch']
                # n2=G.node[v]['patch']

                rad=0.1

                if (u,v) in seen:
                    rad=seen.get((u,v))
                    rad=(rad+np.sign(rad)*0.1)*-1
                alpha=0.5
                
                kwargs = {
                    # 'head_width': 5*lw, 
                    'facecolor': arrow_colour[0:3]+(alphas[idx],),
                    'edgecolor': (0,0,0,0.)
                      #'overhang':-5*0.02/lw,  
                      #'length_includes_head': True,
                      # capstyle='projecting',
                }
                          
                # Can be accepted by fancy arrow patch to alter arrows
                arrow_style = ArrowStyle.Wedge(tail_width = lw,
                    shrink_factor = 0.3)

                e = FancyArrowPatch(
                    n1.center,
                    n2.center,
                    patchA=n1,
                    patchB=n2,
                    
                    arrowstyle=arrow_style,
                    connectionstyle='arc3,rad=%s'%rad,
                    mutation_scale=10.0,
                    lw=lw,   #AROUND 10 TO BE FEASIBLE
                   **kwargs
                )
                seen[(u,v)]=rad
                ax.add_patch(e)
           
    # print("rad", rad)
    # print("Node coordinates", n1, n2)
    # print("arrowcolor", arrow_colour)
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



def plotTreeDiagram(
    G, 
    n_cmap, e_cmap, 
    e_alphas = [], nonadj_alpha=0.1, 
    label_padding = 0.4, 
    arrow_size = 0.02, widthscale=1.0,
    entropy=None, inf_gain=None,
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
    
    edges_for_cmap = draw_networkx_arrows(
        G, 
        edgelist=edge_tuples,
        pos=positions, arrows=True, 
        arrowstyle='->', width = arrow_size, widthscale=widthscale,
        pathstyle=pathstyle, alphas = e_alphas, edge_color= weights,
        edge_cmap=e_cmap, 
        edge_vmin=None, #0.8, 
        edge_vmax=None, #0.85
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
    plt.legend(handles, labels, title=legend_title, mode="expand", ncol=6, loc='lower center')

    plot_title = ''
    if entropy is not None:
        plot_title += str( 
            '\t$\mathcal{S}$=' 
            + str(round(entropy, 2))
        )
    if inf_gain is not None:
        plot_title += str(
            '\t $\mathcal{IG}$=' 
            + str(round(inf_gain, 2))
        )
    if entropy is not None or inf_gain is not None: 
        plt.title(plot_title)

    if save_to_file is not None:
        print("Saving plot in: ", save_to_file)
        plt.savefig(save_to_file, bbox_inches='tight')
    else:
        plt.show()





def cumulativeQMDTreePlot(
        cumulative_csv, 
        wins_per_mod, 
        avg='means',
        only_adjacent_branches=True,
        directed=True,
        pathstyle="curve",
        entropy=None, 
        inf_gain=None,
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
                    
                    
                    frequency = pair_freqs[pairing]/(max_frequency)  
                    
                    
                    edges.append(pairing)
                    edge_frequencies.append(frequency)

                    vs = [a,b]

                    try:
                        thisweight = np.log10(bayes_factors[a][b])
                    except:
                        thisweight=0 #TODO is this right?

                    if thisweight < 0:  
                        # flip negative valued edges and move them to positive
                        thisweight = -thisweight
                        flipped = True
                        G.add_edge(
                            a, b, 
                            weight=thisweight, 
                            winner=b,
                            loser=a,
                            flipped=flipped,
                            adj = is_adj, freq=frequency
                        )
                    else:
                        flipped = False
                        G.add_edge(
                            b, a,
                            winner=a,
                            loser=b,
                            weight=thisweight, 
                            flipped=flipped,
                            adj=is_adj, freq=frequency
                        )
                        
    max_freq = max(edge_frequencies)
    print(max_freq)
    
    freq_scale = 10/(max_freq )  
    
    
    edge_f = [i*freq_scale for i in edge_frequencies]

    arr = np.linspace(0, 50, 100).reshape((10, 10))
    cmap = plt.get_cmap('viridis')
    new_cmap = truncate_colormap(cmap, 0.35, 1.0)

    plotTreeDiagram(G,
        n_cmap = plt.cm.pink_r, 
        e_cmap = new_cmap,
#        e_cmap = plt.cm.Blues, 
        #e_cmap = plt.cm.Paired,     
        #e_cmap = plt.cm.rainbow,     
        nonadj_alpha = 0.1, e_alphas = [] , widthscale=3, #10.5, #widthscale 1
        label_padding = 0.4, pathstyle=pathstyle,
        arrow_size=None,
        entropy=entropy, inf_gain=inf_gain,
        save_to_file = save_to_file
    )   

    return G, edges, edge_f   
    








	
	
    
    
    
	
	
def plot_tree_multi_QMD(
        results_csv, 
        all_bayes_csv, 
        avg_type='medians',
        pathstyle = 'curve',
        entropy=None, 
        inf_gain=None, 
        save_to_file=None
    ):
#    res_csv="/home/bf16951/Dropbox/QML_share_stateofart/QMD/ExperimentalSimulations/Results/multtestdir/param_sweep.csv"
    qmd_res = pandas.DataFrame.from_csv(results_csv, index_col='LatexName')
    mods = list(qmd_res.index)
    winning_count = {}
    for mod in mods:
        winning_count[mod]=mods.count(mod)

    cumulativeQMDTreePlot(
        cumulative_csv=all_bayes_csv, 
        wins_per_mod=winning_count, 
        only_adjacent_branches=True, pathstyle=pathstyle, 
        avg=avg_type, entropy=entropy, inf_gain=inf_gain,
        save_to_file=save_to_file
    )        
	
	
	
	
	
	
	
	

    
    
    
    
    
    
def draw_network(G,pos,ax, node_size, node_color,edge_color,edge_size,edge_alpha,rad=0.1,edge_style="curve"):

    for n in G:
        if len(node_color) > 1:
            c=Rectangle(pos[n],width=node_size, height=node_size, color=node_color[n])
        else:
            c=Rectangle(pos[n],width=node_size, height=node_size, color=node_color[0])
        G.node[n]['patch']=c
        x,y=pos[n]
    seen={}
    
    for (u,v,d) in G.edges(data=True):
        n1=G.node[u]['patch']
        n2=G.node[v]['patch']
        
        if (u,v) in seen:
            rad=seen.get((u,v))
            rad=(rad+np.sign(rad)*0.1)*-1

        nodeA_pos = (n1.xy[0] + node_size/2, n1.xy[1] + node_size/2)
        nodeB_pos = (n2.xy[0] + node_size/2, n2.xy[1] + node_size/2)
        
        if len(edge_alpha) > 1:
            this_alpha = edge_alpha[(u,v)]
            this_color = edge_color[(u,v)][0:3]
        else:
            this_alpha = edge_alpha[0]
            this_color = edge_color[0:3]

        
        if edge_style == "curve":
            arrowstyle = ArrowStyle.Wedge(tail_width = edge_size,  shrink_factor = 0.3)
            kwargs = {
                    'facecolor': this_color+(this_alpha,),
                    'edgecolor': this_color+(0,),
                }
            e = FancyArrowPatch(nodeA_pos, nodeB_pos,patchA=n1,patchB=n2,
                            arrowstyle=arrowstyle,
                            connectionstyle='arc3,rad=%s'%rad,
                            mutation_scale=10.0,
                            lw=2,
                           **kwargs)
            
        else:
            arrowstyle = "->"
            kwargs = {
                    'edgecolor': this_color+(this_alpha,),
                    
                }
            e = FancyArrowPatch(nodeA_pos, nodeB_pos,patchA=n1,patchB=n2,
                            arrowstyle=arrowstyle,
                            connectionstyle='arc3,rad=%s'%rad,
                            mutation_scale=3.0,
                            lw=edge_size,
                           **kwargs)
        
        seen[(u,v)]=rad
        ax.add_patch(e)
        
        
    for n in G:
        ax.add_patch(G.node[n]['patch'])    
        
    return e  