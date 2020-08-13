import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.growth_rules import connected_lattice, growth_rule
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla import construct_models
from qmla.shared_functionality import topology_predefined

class LatticeSet(
    # connected_lattice.ConnectedLattice
    growth_rule.GrowthRule
):

    def __init__(
        self,
        growth_generation_rule,
        true_model=None,
        **kwargs
    ):
        if true_model is not None: 
            self.true_model = true_model
            print("[LatticeSet] got true model {}".format(self.true_model))
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model=true_model,
            **kwargs
        )
        self.tree_completed_initially = True # fixed lattice set considered
        self.base_terms = ['x', 'z']
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.lattice_set_grouped_pauli
        self.initial_models = None # so that QMLA will call generate_models first
        # self.true_model = self.model_from_lattice(self.available_lattices[0])
        self.max_time_to_consider = 45
        self.transverse_field = None
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = self.fraction_own_experiments_for_bf

        self.available_lattices_by_name = {
            # Ising chains
            'chain_2' : topology_predefined._2_site_chain,
            'chain_3' : topology_predefined._3_site_chain,
            'chain_4' : topology_predefined._4_site_chain,
            # 'chain_5' : topology_predefined._5_site_chain,
            # 'chain_6' : topology_predefined._6_site_chain,

            # # fully connected
            # 'fully_connected_3' : topology_predefined._3_site_lattice_fully_connected, 
            # 'fully_connected_4' : topology_predefined._4_site_lattice_fully_connected, 
            # 'fully_connected_5' : topology_predefined._5_site_lattice_fully_connected, 

            # # other lattices
            # 'grid_4' : topology_predefined._4_site_square,
            # 'grid_6' : topology_predefined._6_site_grid
        }
        self.available_lattices = list(self.available_lattices_by_name.values())
        self.lattice_names = list(sorted(self.available_lattices_by_name.keys()))
        # self.true_lattice = topology_predefined._4_site_square
        # randomly select a true model from the available lattices
        lattice_idx = self.qmla_id % len(self.available_lattices)
        self.true_lattice_name = self.lattice_names[ lattice_idx ]
        self.true_lattice = self.available_lattices_by_name[self.true_lattice_name]
        self.log_print(["QMLA {} using lattice {}: {}".format(self.qmla_id, self.true_lattice_name, self.true_lattice)])
        self.true_model = self.model_from_lattice(self.true_lattice)

        self.max_num_models_by_shape = {
            1 : 2, 
            2 : 2, 
            3 : 2, 
            4 : 2, 
            5 : 2, 
            6 : 2, 
            'other' : 0
        }
        self.num_processes_to_parallelise_over = len(self.available_lattices)


    def model_from_lattice(
        self, 
        lattice
    ):
        connected_sites = lattice.get_connected_site_list()
        conn_list = [list(str(c) for c in conn) for conn in connected_sites]
        conn_string = '_'.join(['J'.join(c) for c in conn_list])
        lattice_dimension = lattice.num_sites()

        individual_operators = [
            'pauliLikewise_l{}_{}_d{}'.format(
                op, 
                conn_string, 
                lattice_dimension
            )
            for op in self.base_terms
        ]
        complete_model = '+'.join(individual_operators)

        if self.transverse_field is not None:
            transverse_string = (
                '_'.join(list(str(s) for s in range(1, lattice_dimension+1)))
            )
            transverse_term = 'pauliLikewise_l{}_{}_d{}'.format(
                self.transverse_field, 
                transverse_string, 
                lattice_dimension
            )
            complete_model += '+{}'.format(transverse_term)
        return construct_models.alph(complete_model)


    def generate_models(
        self, 
        model_list, 
        **kwargs 
    ):
        model_set = [
            self.model_from_lattice(lattice)
            for lattice in self.available_lattices
        ]
        self.log_print([
            "Generate models returning ", model_set
        ])

        return model_set


    def growth_rule_specific_plots(
        self,
        save_directory,
        qmla_id=0, 
    ):
        save_file = os.path.join(
            save_directory, 
            'true_lattice.png'
        )
        self.log_print([
            "GR plots for fixed lattice. Save file:", save_file
        ])
        self.true_lattice.draw_topology(
            save_to_file = save_file
        )

    def growth_rule_finalise(
        self
    ):        
        self.storage.lattice_record =  pd.DataFrame(
            columns=['true_lattice', 'model_type',],
            data =np.array([[self.true_lattice_name, self.growth_generation_rule]])
        )
        

