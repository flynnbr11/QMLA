import numpy as np
import itertools as itr

import os as os
import sys as sys 
import pandas as pd
import warnings
import copy
import time as time
#import Evo as evo
import DataBase 
import warnings
import ModelNames



class topology_grid():
    def __init__(
        self,
        dimension = 2,
        num_sites = 1,
        maximum_connection_distance = 1, # to count as connected, i.e. 1 is nearest neighbours
        linear_connections_only = True, 
        all_sites_connected = False
    ):
        self.dimension = dimension
        self.maximum_connection_distance = maximum_connection_distance
        self.only_linearly_connections = linear_connections_only
        self.connect_all_sites = all_sites_connected
        
        self.occupation = {
            'rows' : {
                1 : [1]
            },
            'cols' : {
                1 : [1]
            }
        }
        self.coordinates = {
            1 : [1,1]
        }
        self.nearest_neighbours = {
            1 : []
        }
        self.site_connections = {
            1 : []
        }
        
        self.new_connections = []
            
            
        for i in range(1, num_sites):
            self.add_site()
        
    def add_site(self):
        if self.dimension == 1:
            new_site_idx = self.add_site_1d_grid()
        elif self.dimension == 2:
            new_site_idx = self.add_site_2d_grid()
        
        this_site_new_connections = []
        new_coordinate = self.coordinates[new_site_idx ]
        self.nearest_neighbours[new_site_idx] = []            
        self.site_connections[new_site_idx] = []
        other_sites = list(set(self.site_indices) - set([new_site_idx]))
        for i in other_sites:
            other_coords = self.coordinates[i] 
            nearest_neighbour = self.check_nearest_neighbour_sites(
                site_1 = new_coordinate,
                site_2 = other_coords
            )
            
            if nearest_neighbour is True:
                if i not in self.nearest_neighbours[new_site_idx]:
                    try:
                        self.nearest_neighbours[new_site_idx].append(i)
                    except:
                        self.nearest_neighbours[new_site_idx] = [i]
                if new_site_idx not in self.nearest_neighbours[i]:
                    try:
                        self.nearest_neighbours[i].append(new_site_idx)
                    except: 
                        self.nearest_neighbours[i] = [new_site_idx]

            connected_sites, shared_axis = self.check_sites_connection(
                site_1_idx = i,
                site_2_idx = new_site_idx
            )
            
            if (
                self.connect_all_sites == True
                or 
                (
                    connected_sites == True
                    and
                    (
                        shared_axis == True 
                        or 
                        self.only_linearly_connections==False
                    )
                )
            ):
                conn = tuple(sorted([i, new_site_idx]))
                this_site_new_connections.append(conn)
                if i not in self.site_connections[new_site_idx]:
                    try:
                        self.site_connections[new_site_idx].append(i)
                    except:
                        self.site_connections[new_site_idx] = [i]
                if new_site_idx not in self.site_connections[i]:
                    try:
                        self.site_connections[i].append(new_site_idx)
                    except: 
                        self.site_connections[i] = [new_site_idx]
                        
        self.new_connections.append(this_site_new_connections)
                        
        
    @property
    def site_indices(self):
        return list(self.coordinates.keys())
    
    def num_sites(self):
        return len(list(self.coordinates.keys()))

        
    def check_nearest_neighbours_from_indices(self, idx_1, idx_2):
        site_1 = self.coordinates[idx_1]
        site_2 = self.coordinates[idx_2]
        print("Site 1:", site_1)
        print("Site 2:", site_2)
        return self.check_nearest_neighbour_sites(site_1, site_2)
        
    def check_nearest_neighbour_sites(self, site_1, site_2):
        # simply checks whether sites are adjacent (or comptues distance)
        # assumes Cartesian coordinates
        if len(site_1) != len(site_2):
            print(
                "Site distance calculation: both sites must have same number of dimensions.",
                "Given:", site_1, site_2
            )
            raise NameError('Unequal site dimensions.')

        dim = len(site_1)
        dist = 0 
        for d in range(dim):
            dist += np.abs(site_1[d] - site_2[d])

        if dist == 1:
            return True
        else:
            return False

    def get_distance_between_sites(self, site_1_idx, site_2_idx):
        site_1 = self.coordinates[site_1_idx]
        site_2 = self.coordinates[site_2_idx]

        if len(site_1) != len(site_2):
            print(
                "Site distance calculation: both sites must have same number of dimensions.",
                "Given:", site_1, site_2
            )
            raise NameError('Unequal site dimensions.')

        dim = len(site_1)
        dist = 0
        shared_axis = False
        for d in range(dim):
            dist += np.abs(site_1[d] - site_2[d])**2
            if site_1[d] == site_2[d]:
                shared_axis = True
        dist = np.sqrt(dist)
        return dist, shared_axis
        
    def check_sites_connection(self, site_1_idx, site_2_idx):
#         site_1 = self.coordinates[site_1_idx]
#         site_2 = self.coordinates[site_2_idx]
        dist, shared_axis = self.get_distance_between_sites(site_1_idx, site_2_idx)
        
        if dist <= self.maximum_connection_distance:
            connected = True
        else:
            connected = False
            
        return connected, shared_axis
        
    def get_connected_site_list(self):
        
        coordinates = self.coordinates
        site_indices = list(coordinates.keys())
        connected_sites = []

        for i in range(len(site_indices)):
            idx_1 = site_indices[i]
            for j in range(i+1, len(site_indices)):
                idx_2 = site_indices[j]
                connected, shared_axis = self.check_sites_connection(
                    site_1_idx = idx_1,
                    site_2_idx = idx_2
                )
                if (
                    self.connect_all_sites == True
                    or 
                    (
                        connected == True
                        and
                        (
                            shared_axis == True 
                            or 
                            self.only_linearly_connections==False
                        )
                    )
                ):
                    connected_sites.append( (idx_1, idx_2) )
                    
        return connected_sites
        
    def get_nearest_neighbour_list(self):
        
        coordinates = self.coordinates
        site_indices = list(coordinates.keys())
        nearest_neighbours = []

        for i in range(len(site_indices)):
            idx_1 = site_indices[i]
            for j in range(i, len(site_indices)):
                idx_2 = site_indices[j]
                nn = self.check_nearest_neighbour_sites(
                    site_1 = coordinates[idx_1],
                    site_2 = coordinates[idx_2],
                )
                if nn is True:
                    nearest_neighbours.append( (idx_1, idx_2) )
        return nearest_neighbours


    def add_site_1d_grid(self):
        max_site_idx = max(list(self.coordinates.keys()))
        new_site_idx = max_site_idx + 1
        self.nearest_neighbours[new_site_idx] = []
        new_coordinate = [new_site_idx, 1] # in 1d site ID is same as position            
        self.coordinates[new_site_idx] = new_coordinate
        
        return new_site_idx
        
        
    def add_site_2d_grid(self):
        # grows in a manner which minimises area of the topology
        rows = self.occupation['rows']
        cols = self.occupation['cols']
        
        row_values = rows.keys()
        col_values = cols.keys() 
        min_span_row = None
        min_span_col = None        

        for row_idx in rows:
            span = max(rows[row_idx]) - min(rows[row_idx])
            if (
                min_span_row is None 
                or
                span < min_span_row
            ):
                min_span_row = span
                min_span_row_idx = row_idx

        for col_idx in cols:
            span = max(cols[col_idx]) - min(cols[col_idx])
            if (
                min_span_col is None 
                or
                span < min_span_col
            ):
                min_span_col = span
                min_span_col_idx = col_idx

        if min_span_col < min_span_row:
            # growing downward in y-axis
            new_row = max(cols[min_span_col_idx]) + 1
            new_col = min_span_col_idx
        else:
            # growing rightward in x-axis
            new_col = max(rows[min_span_row_idx]) + 1
            new_row = min_span_row_idx

        new_coordinate = [new_row, new_col]

        try:
            self.occupation['rows'][new_row].append(new_col)
        except:
            self.occupation['rows'][new_row] = [new_col]

        try:
            self.occupation['cols'][new_col].append(new_row)
        except:
            self.occupation['cols'][new_col] = [new_row]


        max_site_idx = max(list(self.coordinates.keys()))
        new_site_idx = max_site_idx + 1
        self.coordinates[new_site_idx] = new_coordinate
        
        return new_site_idx


    def add_sites_until_closed_topology(self):
        # Add sites in such a way that all sites have at least two nearest neighbours
        ## Assumption to minimise energy -- not always necessary
        all_sites_greater_than_2_nearest_neighbours = False
        while all_sites_greater_than_2_nearest_neighbours == False:
            new_site_idx = self.add_new_coordinate_2d_lattice()
            nn_lists = list(self.nearest_neighbours.values())
            num_nearest_neighbours = np.array([len(a) for a in nn_lists])
            all_sites_greater_than_2_nearest_neighbours = np.all(
                num_nearest_neighbours >= 2
            )            
            
            
            
    def draw_topology(self):
        import networkx as nx
        plt.clf()
        Graph = nx.Graph()
        
        print("site indices:", self.site_indices)
        for c in self.site_indices:
            Graph.add_node(c)
            Graph.nodes[c]['neighbours'] = self.nearest_neighbours[c]
            Graph.nodes[c]['position'] = tuple(self.coordinates[c])
            Graph.nodes[c]['label'] = str(c)
            
        
        # Get positions and labels
        positions = dict( 
            zip( 
                Graph.nodes(), 
                tuple(  [prop['position'] for (n,prop) in Graph.nodes(data=True)]  ) 
            )
        )
        label_positions = []   
        label_padding = 0.0
        labels = dict( 
            zip( 
                Graph.nodes(), 
                tuple(  [prop['label'] for (n,prop) in Graph.nodes(data=True)]  ) 
            )
        )  
        for key in positions.keys():
            label_positions.append( 
                tuple( 
                    np.array(positions[key]) - np.array([0., label_padding]) 
                ) 
        )
    
        label_positions = dict(
            zip( 
                positions.keys(), 
                tuple(label_positions) 
            )
        )

        # which nodes to connect (nearest neighbours)
        edges = []
        for c in self.site_indices:
            neighbours = self.nearest_neighbours[c]
            for n in neighbours:
                edge = tuple(sorted([c,n]))
                if edge not in edges:
                    edges.append(edge)

        
        plt.gca().invert_yaxis() # so branch 0 on top
        plt.title("Topology of system")
        nx.draw_networkx_nodes(
            Graph, 
            with_labels = True, # labels=labels, 
            pos=positions, 
            node_size=600,
            node_color='blue',
            alpha=0.2
        )
        nx.draw_networkx_labels(
            Graph, 
            label_positions, 
            labels,
            font_color='black',
            font_weight='bold'
        )
        
        nx.draw_networkx_edges(
            Graph, 
            pos = self.coordinates,
            edgelist = edges,
            edge_color = 'grey',
            alpha = 0.8,
            style='dashed',
            label='Nearest neighbours'
        )
        
        self.Graph = Graph
