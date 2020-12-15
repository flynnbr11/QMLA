import sys
import os
import numpy as np
import pickle
import itertools
import copy
import scipy
import time
import math
import pandas as pd
import seaborn as sns
import random
import sklearn
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import GridSpec

import qmla.shared_functionality.elo_graphs

def generate_random_regular_graph(
    model_list, 
    degree_rate=None,
):
    num_models = len(model_list)
    if degree_rate is None: 
        # set degree based on num models - must be high enough that there is a path between every pair
        degree_rate = 0.5
        # if num_models < 10:
        #     degree_rate = 0.5
        # elif num_models < 20:
        #     degree_rate = 0.4
        # else: 
        #     degree_rate = 1/3

    g = nx.random_regular_graph(
        n = len(model_list), 
        d = int(degree_rate*len(model_list))
    )
    
    model_list_iter = iter(model_list)
    node_labels = {
        n : next(model_list_iter)
        for n in list(g.nodes)
    }
    nx.relabel_nodes(g,node_labels,False)

    for n in g.nodes:
        g.nodes[n]['longest_path_to_any_target'] = 0

    node_edges = { n : len(g.edges(n)) for n in model_list}

    lowest_connectivity = min( node_edges.values() )
    highest_connectivity = max( node_edges.values() )

    for n in g:
        for target in g:
            if n != target:
                shortest_path = nx.shortest_path_length(
                    g, 
                    source = n, 
                    target = target
                )

                this_node_current_longest_path = g.nodes[n]['longest_path_to_any_target']
                if shortest_path  > this_node_current_longest_path:
                    g.nodes[n]['longest_path_to_any_target'] = shortest_path

    distance_between_nodes = [ g.nodes[n]['longest_path_to_any_target'] for n in g]                

    lowest_distance_between_nodes = min(distance_between_nodes)
    highest_distance_between_nodes = max(distance_between_nodes)
    
    connections = list(g.edges)
    summary = "{} connections, lowest_conn={}, highest_conn={}, max_distance={}".format(
        len(connections), 
        lowest_connectivity, 
        highest_connectivity,
        highest_distance_between_nodes
    )

    result = {
        'connections' : connections, 
        'num_connections': len(connections),
        'lowest_distance' : lowest_distance_between_nodes, 
        'highest_distance' : highest_distance_between_nodes, 
        'lowest_connectivity' : lowest_connectivity, 
        'highest_connectivity' : highest_connectivity,
        'graph' : g,
        'node_edges' : node_edges, 
        'summary' : summary, 
    }

    return connections, g


def generate_graph(
    model_list, 
    num_connections
):
    r""" 
    Generate a graph with random connections. 
    
    Analyse several aspects of that graph and store them. 
    
    :return dict graph_result: 
        aspects of the analysis pertinent to whether the graph can be useful
    """

    all_pairs = list(itertools.combinations(model_list, 2))
    sampled_pairs = random.sample(
        all_pairs, int(num_connections)
    )

    g = nx.Graph()
    for n in model_list:
        g.add_node(n)
        g.nodes[n]['longest_path_to_any_target'] = 0
    for s in sampled_pairs:
        g.add_edge(min(s),max(s)) # edge min -> max

    node_edges = { n : len(g.edges(n)) for n in model_list}

    lowest_connectivity = min( node_edges.values() )
    highest_connectivity = max( node_edges.values() )

    for n in g:
        for target in g:
            if n != target:
                shortest_path = nx.shortest_path_length(
                    g, 
                    source = n, 
                    target = target
                )

                this_node_current_longest_path = g.nodes[n]['longest_path_to_any_target']
                if shortest_path  > this_node_current_longest_path:
                    g.nodes[n]['longest_path_to_any_target'] = shortest_path

    distance_between_nodes = [ g.nodes[n]['longest_path_to_any_target'] for n in g]                

    lowest_distance_between_nodes = min(distance_between_nodes)
    highest_distance_between_nodes = max(distance_between_nodes)

    result = {
        'connections' : sampled_pairs, 
        'num_connections': len(sampled_pairs),
        'lowest_distance' : lowest_distance_between_nodes, 
        'highest_distance' : highest_distance_between_nodes, 
        'lowest_connectivity' : lowest_connectivity, 
        'highest_connectivity' : highest_connectivity,
        'graph' : g
    }
    
    return result
    


def check_graph_meet_criteria(
    graph_result,
    config,
):
    r"""
    Compare the result of a graph with some configuration. 
    The configuration defines the conditions which the graph must meet in order to be useful. 
    In particular, 
    - the maximum distance allowed between any pair of nodes, 
    - the minimum connectivity allowed of any node
    - the maximum connectivty allowed of any node
    
    :return: True if a useful graph is detected; False otherwise. 
    """
    num_connections = config[0]
    max_distance = config[1]
    min_connectivity = config[2]
    max_connectivity = config[3]

    if (
        graph_result['highest_distance'] <= max_distance # highest distance between any pair of nodes
        and graph_result['lowest_connectivity'] >= min_connectivity # lowest number of connections of any node
        and graph_result['highest_connectivity'] <= max_connectivity # highest number of connections of any node
    ):
        print("Hurray!")
        return True
    else:
        return False



def generate_configurations(model_list, num_iterations_per_graph=10, yield_result=True):
    r"""
    Construct a set of configurations to attempt to generate useful graphs from. 
    """

    # number of connections - minimise expense 
    num_models = len(model_list)
    worst_case = scipy.special.comb(num_models, 2)
    max_num_connections = math.ceil(0.75 * worst_case) # if above 3/4 fully connected, just use fully connected
    if worst_case == max_num_connections:
        allowable_num_connections = [int(worst_case)]
    elif num_models > 15:
        # realistically will use at least 4*N, though it would be great to get below
        allowable_num_connections = list(range(4*num_models, max_num_connections, 1))
    else:
        allowable_num_connections = list(range(num_models, max_num_connections, 1))

    # max distance allowed between any pair of models
    # do not allow d<2 as this will force a fully connected graph
    # do not allow very distantly-connected graphs
    # as we want to make sure models have competed in close range
    # max_distances = list( range(2, math.ceil(num_models/2)) )
    max_distances = list( range(2, 5) )
    if len(max_distances) == 0:
        max_distances = [num_models-1]
        
    # minimal connectivity for any node in graph
    # eg with 10 nodes, such that all nodes are connected to at least 4 others
    min_connections = list(range(
        math.ceil(num_models/2), # start with a high threshold -> fewer comparisons to do
        math.ceil(num_models/10), # don't allow too few connections  
        -1
    ))
    if len(min_connections) == 0 :
        min_connections = [1]
    
    all_configs = []   
    for s in allowable_num_connections:        
        for d in max_distances:
            for min_conn in min_connections:
                max_connections = list(
                    range( min_conn+1, math.ceil(1.5*min_conn) )
                )
                if len(max_connections)==0:
                    max_connections = [min_conn]
                for max_conn in max_connections:

                    config = [s, d, min_conn, max_conn]
                    
                    for i in range(num_iterations_per_graph):
                        yield config


def attempt_minimal_graph(model_list, num_iterations_per_graph = 10):
    r"""
    Try to construct a minimally complex graph that satisfies user requirements. 
    
    Graphs are generated at random with increasing connectivity. 
    These are then checked against a series of increasingly leanient criteria
    until either a graph satisfies the criteria, or terminate after a fixed number of iterations. 
    The first graph which satisfies the criteria will do so at the minimal-complexity criteria, 
    such that the resultant graph is optimal.
    
    :param list model_list: list of ints corresponding to models
    :return graph_result: (if graph found) data about the optimal graph.   
    """
    t_init = time.time()
    availabe_configs = generate_configurations(
        model_list = model_list,
        num_iterations_per_graph = num_iterations_per_graph
    )
    print("Getting configurations to loop through took {} sec".format(time.time() - t_init))

    minimal_graph_found = False
    counter = 0
    fail_counter = 0
    t_init = time.time()
    try:
        while not minimal_graph_found:
            config = next(availabe_configs)
            graph_generated=False
            try:
                graph_result = generate_graph(
                    model_list,
                    num_connections = config[0]
                )
                minimal_graph_found = check_graph_meet_criteria(
                    graph_result = graph_result,
                    config = config,
                )
            except:
                fail_counter += 1
            counter += 1
    except:
        pass
    print("Checking configurations took {} sec".format(time.time() - t_init))
    
    if minimal_graph_found:
        print("Found a useful graph after {} attempts.".format(counter))
        print("It has {} connections. Models are separated by distance up to {}.".format(
            graph_result['num_connections'], graph_result['highest_distance']
        ))
        print("Models have between {} and {} connections.".format(
            graph_result['lowest_connectivity'],
            graph_result['highest_connectivity']
        ))
        return graph_result
    else:
        print(
            "No such graph found after {} attempts. [{} failed graph generations]".format(counter, fail_counter)
        )
        
        
def find_efficient_comparison_pairs(model_names):
    r"""
    Try to find a graph which connects the given model_names efficiently
    to minimise the number of pairwise comparisons needed. 
    """
    
    # generate numerical IDs for each model
    model_id_names = {
        model_names.index(m) : m
        for  m in model_names
    }

    n_nodes = len(model_names)
    path_elements = qmla.shared_functionality.elo_graphs.__file__.split('/')
    path_elements = path_elements[:-1]
    path_elements.append("optimal_graph_{}_nodes.p".format(n_nodes))
    graph_pickle_file = os.path.abspath('/'.join(path_elements)) #  os.path.abspath(os.path.join(*path_elements))

    print("Graph pickle file:", graph_pickle_file)
    try:
        graph_data = pickle.load(open(graph_pickle_file, 'rb'))
        graph_retrieved = True
        print("Retrieved graph of size ", n_nodes)
    except:
        print("Failed to retrieve graph")
        graph_retrieved = False

    model_id_list = list(model_id_names.keys())
    
    if not graph_retrieved:
    
        try_get_graph = attempt_minimal_graph(
            model_list = model_id_list, 
            num_iterations_per_graph = 50
        )
    
        if not try_get_graph:
            print("A graph was not found; fully connecting model list.")
            model_pairs = list(itertools.combinations(model_names, 2))
            graph=None
            return model_pairs, graph

        # nx.draw(graph)
        graph_data = try_get_graph
        pickle.dump(
            graph_data,
            open(graph_pickle_file, 'wb')
        )
        print("Storing to {}: {}".format(graph_pickle_file, graph_data))
        
    graph = graph_data['graph']
    connections = graph_data['connections']
    model_pairs = [ 
        ( model_id_names[m1], model_id_names[m2] ) for m1, m2 in connections
    ]
    
    print("model_pairs: ", model_pairs)
    return model_pairs, graph