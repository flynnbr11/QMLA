import sys
import os
import numpy as np
import pickle
import itertools
import copy
import scipy
import math
import pandas as pd
import seaborn as sns
import random
import sklearn
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.pyplot import GridSpec
try:
    from lfig import LatexFigure, get_latex_rc_params
except:
    from qmla.shared_functionality.latex_figure import LatexFigure, get_latex_rc_params

import qmla

def plot_qmla_branches(
    q, 
    show_fscore_cmap=False,
    return_graphs=False
):
    trees = list(q.trees.values())
    q.log_print(["Plotting QMLA branch graphs. Trees:", trees])
    plt.rcParams.update(get_latex_rc_params(font_scale=1.5,))

    for tree in trees:
        # tree = trees[0] # TODO loop over trees
        q.log_print(["Working on tree ", tree.exploration_strategy])
        graphs = {}

        branches = [
            tree.branches[b] for b in 
            sorted(tree.branches.keys())
        ]
        if len(branches) > 1:
            branches = branches[:-1]

        num_branches = len(branches)
        ncols = int(np.ceil(np.sqrt(num_branches))) 
        if num_branches == 1:
            ncols = 1
            nrows = 1
        elif num_branches == 2:
            ncols = 1
            nrows = 2
        else:
            nrows = int( np.ceil(num_branches / ncols))  
        
        # Generate plot
        widths = [1]*ncols
        widths.append(0.1)
        if show_fscore_cmap:
           widths.append(0.1)
        total_ncols = len(widths)
        
        size_scaler = min(2, 4 / num_branches)
        label_fontsize = 15*size_scaler

        lf = LatexFigure(
            auto_label=False, 
            gridspec_layout = (nrows, total_ncols), 
            gridspec_params = {
                'width_ratios' : widths,
                'wspace' : 0.25
            } 
        )

        # colour maps
        f_score_cmap = q.exploration_class.f_score_cmap
        bf_cmap = q.exploration_class.bf_cmap
        min_bf = q.bayes_factors_df.log10_bayes_factor.min()
        max_bf = q.bayes_factors_df.log10_bayes_factor.max()
        norm = plt.Normalize(vmin = min_bf, vmax = max_bf)
        bf_cmapper = plt.cm.ScalarMappable(norm=norm, cmap=bf_cmap)
        bf_cmapper.set_array([]) # still no idea what this is for??

        # BF cmap
        cbar_ax = lf.new_axis(
            force_position=(0, total_ncols-1-int(show_fscore_cmap)),
            span = ('all', 1)
        )
        lf.fig.colorbar(bf_cmapper, cax = cbar_ax)        
        cbar_ax.set_title(
            r"$\\log_{10}(BF)$", 
            loc='center'
        )

        # F score cmap
        if show_fscore_cmap:
            ax = lf.new_axis(
                force_position=[0, total_ncols-1],
                span = ('all', 1)
            )

            sm = plt.cm.ScalarMappable(
                cmap = f_score_cmap, 
                norm=plt.Normalize(vmin=0, vmax=1)
            )
            sm.set_array([])
            lf.fig.colorbar(sm, cax=ax, orientation='vertical')
            ax.set_ylabel(
                r"F-score", 
            )

        # Plot graphs

        min_seen_bf = 0
        max_seen_bf = 0
        for branch in branches:
            ax = lf.new_axis()
            
            models = branch.models.keys()
            pairs = branch.pairs_to_compare

            graph = nx.Graph()
            for m in models:
                graph.add_node(
                    m, 
                    model_id=int(m), 
                    model_name=branch.models[m],
                    f_score = q.model_f_scores[m],
                    colour = f_score_cmap(q.model_f_scores[m]),
                    longest_path_to_any_target = 0
                )

            for edge in pairs:
                e = (min(edge), max(edge)) # edges go A->B low -> high
                bf = q.all_bayes_factors[e[0]][e[1]][-1]
                lbf = np.log10( bf ) 
                colour = bf_cmapper.to_rgba(lbf)
                graph.add_edge(
                    e[0], e[1],
                    log_bf = lbf,
                    colour = colour
                )    

            pos = nx.kamada_kawai_layout(graph) # could use spring_layout or other
            labels = nx.get_node_attributes(graph, 'model_id')
            node_colours = [ graph.nodes[m]['colour'] for m in models]
            edge_colours = [ graph.edges[e]['colour'] for e in graph.edges ]

            nx.draw_networkx_nodes(
                graph, 
                pos, 
                alpha = 0.7,
                node_color = node_colours,
                node_size = 400*size_scaler,
                ax = ax
            )
            nx.draw_networkx_labels(graph,pos,labels,font_size=label_fontsize, ax = ax)

            nx.draw_networkx_edges(
                graph, 
                pos, 
                edgelist = graph.edges,
                edge_color = edge_colours,
                ax = ax,
                alpha = 1,
                width=4
            )           
            
            # summarise this graph in a text box
            node_edges = { n : len(graph.edges(n)) for n in graph.nodes}
            lowest_connectivity = min( node_edges.values() )
            highest_connectivity = max( node_edges.values() )

            for n in graph:
                for target in graph:
                    if n != target:
                        try:
                            shortest_path = nx.shortest_path_length(
                                graph, 
                                source = n, 
                                target = target
                            )
                        except:
                            # can't connect to some other node
                            shortest_path = -1

                        this_node_current_longest_path = graph.nodes[n]['longest_path_to_any_target']
                        if shortest_path  > this_node_current_longest_path:
                            graph.nodes[n]['longest_path_to_any_target'] = shortest_path

            distance_between_nodes = [ graph.nodes[n]['longest_path_to_any_target'] for n in graph.nodes]                

            lowest_distance_between_nodes = min(distance_between_nodes)
            highest_distance_between_nodes = max(distance_between_nodes)

            summary = "Branch {} \n{} models \n {} edges\n ${} \leq  C \leq{}$\n $d \leq {}$".format(
                branch.branch_id,
                len(graph.nodes),
                len(graph.edges), 
                lowest_connectivity, highest_connectivity, 
                highest_distance_between_nodes
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            
            ax.text(-0.05, 0.95, 
                    summary, 
                    transform=ax.transAxes, 
                    fontsize = label_fontsize*0.5, 
                    bbox=props,
                    ha = 'center', 
                    va  ='center'
            )
            ax.axis('off')
           
            graphs[branch.branch_id] =  graph

        # Save figure
        save_file = 'graphs_of_branches_{}'.format(tree.exploration_strategy)
        q.log_print(["Storing ", save_file])
        lf.save(os.path.join(q.qmla_controls.plots_directory, save_file), file_format = q.qmla_controls.figure_format)
