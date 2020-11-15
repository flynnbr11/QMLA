import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla import construct_models
from qmla.shared_functionality import topology_predefined

class LatticeSet(
    exploration_strategy.ExplorationStrategy
):

    def __init__(
        self,
        exploration_rules,
        true_model=None,
        **kwargs
    ):
        if true_model is not None: 
            self.true_model = true_model
            print("[LatticeSet] got true model {}".format(self.true_model))
        super().__init__(
            exploration_rules=exploration_rules,
            true_model=true_model,
            **kwargs
        )

        self._shared_true_parameters = False
        self.tree_completed_initially = True # fixed lattice set considered
        # self.base_terms = ['x', 'z']
        # self.transverse_field = None
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.lattice_set_grouped_pauli
        self.initial_models = None # so that QMLA will call generate_models first
        # self.true_model = self.model_from_lattice(self.available_lattices[0])
        self.max_time_to_consider = 45
        self.check_champion_reducibility = False
        self.fraction_own_experiments_for_bf = 0.25
        self.fraction_opponents_experiments_for_bf = self.fraction_own_experiments_for_bf
        self.fraction_particles_for_bf = 0.25

        self.lattice_names = [
            '_2_site_chain', 
            '_3_site_chain', 
            '_4_site_chain', 
            '_5_site_chain', 
            '_6_site_chain', 
            '_3_site_lattice_fully_connected', 
            '_4_site_lattice_fully_connected',
            '_5_site_lattice_fully_connected',
            '_4_site_square',
            '_6_site_grid'
        ]
        # self.rerun_lattices = [
        #     '_5_site_lattice_fully_connected',
        #     '_6_site_grid'
        # ]
        # self.lattice_names = list(sorted(self.available_lattices_by_name.keys()))
        self.available_lattices_by_name = {
            k : topology_predefined.__getattribute__(k)
            for k in self.lattice_names
        }
        self.available_lattices = [
            topology_predefined.__getattribute__(k)
            for k in self.lattice_names
        ]
        # self.true_lattice = topology_predefined._4_site_square
        # randomly select a true model from the available lattices
        if self._shared_true_parameters:
            lattice_idx = -1
        else:
            lattice_idx = self.qmla_id % len(self.available_lattices)  
            # Rerunning subset with more resources
            # lattice_idx = self.qmla_id % len(self.rerun_lattices)  
        self.true_lattice_name = self.lattice_names[ lattice_idx ]
        # self.true_lattice_name = self.rerun_lattices[ lattice_idx ]
        self.true_lattice = self.available_lattices_by_name[self.true_lattice_name]
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


    def exploration_strategy_specific_plots(
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

    def exploration_strategy_finalise(
        self
    ):        
        self.storage.lattice_record =  pd.DataFrame(
            columns=['true_lattice', 'model_type',],
            data =np.array([[self.true_lattice_name, self.exploration_rules]])
        )
        

