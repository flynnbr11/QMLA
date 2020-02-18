import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, connected_lattice_probabilistic
from qmla import experiment_design_heuristics
from qmla import topology
# from qmla import model_generation
from qmla import model_naming
from qmla import probe_set_generation
from qmla import database_framework


class HeisenbergXYZProbabilistic(
    # connected_lattice.ConnectedLattice
    connected_lattice_probabilistic.ConnectedLatticeProbabilistic
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        self.lattice_dimension = 2
        self.initial_num_sites = 2
        self.lattice_connectivity_max_distance = 1
        self.lattice_connectivity_linear_only = True
        self.lattice_full_connectivity = False
        self.model_heuristic_function = experiment_design_heuristics.MultiParticleGuessHeuristic
        self.max_num_sites = 4

        # self.true_model_partially_connected = 'pauliSet_xJx_1J2_d4PPPPpauliSet_yJy_1J2_d4PPPPpauliSet_xJx_2J3_d4PPPPpauliSet_yJy_3J4_d4PPPPpauliSet_zJz_3J4_d4PPPPpauliSet_yJy_1J4_d4'
        # self.true_model_partially_connected ='pauliSet_xJx_1J2_d4+pauliSet_yJy_1J2_d4+pauliSet_xJx_2J3_d4+pauliSet_yJy_3J4_d4+pauliSet_zJz_3J4_d4+pauliSet_yJy_1J4_d4'
        # self.true_model_fully_connected_square = 'pauliSet_xJx_1J2_d4PPPPpauliSet_yJy_1J2_d4PPPPpauliSet_zJz_1J2_d4PPPPpauliSet_xJx_1J3_d4PPPPpauliSet_yJy_1J3_d4PPPPpauliSet_zJz_1J3_d4PPPPpauliSet_xJx_2J4_d4PPPPpauliSet_yJy_2J4_d4PPPPpauliSet_zJz_2J4_d4PPPPpauliSet_xJx_3J4_d4PPPPpauliSet_yJy_3J4_d4PPPPpauliSet_zJz_3J4_d4'
        # self.true_model_partially_connected = 'pauliSet_1J2_xJx_d3PPPpauliSet_1J2_yJy_d3PPPpauliSet_2J3_zJz_d3'
        # self.true_model_partially_connected = 'pauliSet_xJx_1J2_d4PPPPpauliSet_yJy_1J2_d4PPPPpauliSet_zJz_2J3_d4PPPPpauliSet_yJy_3J4_d4PPPPpauliSet_zJz_3J4_d4'
        self.true_model_partially_connected = 'pauliSet_xJx_1J2_d4PPPPpauliSet_yJy_1J2_d4PPPPpauliSet_xJx_1J3_d4PPPPpauliSet_yJy_2J4_d4'
        self.three_site_chain_xxz = 'pauliSet_1J2_xJx_d3PPPpauliSet_2J3_xJx_d3PPPpauliSet_2J3_zJz_d3'
        self.four_site_xxz = 'pauliSet_1J2_xJx_d4PPPPpauliSet_1J3_zJz_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_3J4_xJx_d4PPPPpauliSet_3J4_zJz_d4'
        
        self.true_model = self.four_site_xxz
        self.true_model = database_framework.alph(self.true_model)
        self.qhl_models = [
            self.true_model,
            'pauliSet_1J2_xJx_d4PPPPpauliSet_1J3_xJx_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_3J4_xJx_d4PPPPpauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4PPPPpauliSet_1J3_xJx_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4PPPPpauliSet_1J2_zJz_d4PPPPpauliSet_1J3_xJx_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4PPPPpauliSet_1J2_zJz_d4PPPPpauliSet_1J3_xJx_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_2J4_zJz_d4PPPPpauliSet_3J4_xJx_d4'
        ]
        self.base_terms = [
            'x',
            # 'y',
            'z'
        ]
        self.num_probes = 5
        self.max_time_to_consider = 20
        # fitness calculation parameters. fitness calculation inherited.
        # 'all' # 'all' # at each generation Badassness parameter
        self.num_top_models_to_build_on =  'all'
        self.model_generation_strictness = 0  # 1 #-1
        self.fitness_win_ratio_exponent = 1
        self.fitness_minimum = 0.25
        self.fitness_maximum = 1.0
        self.min_param = 0
        self.max_param = 1
        self.check_champion_reducibility = True
        self.generation_DAG = 1

        self.tree_completed_initially = False
        self.num_processes_to_parallelise_over = 10
        self.max_num_models_by_shape = {
            # 1 : 0,
            # 2: 10,
            # 3: 10,
            2: 30,
            3: 30,
            4: 30,
            'other': 0
        }

        self.setup_growth_class()


class HeisenbergXYZPredetermined(
    HeisenbergXYZProbabilistic
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.max_time_to_consider = 20
        self.tree_completed_initially = True
        self.num_processes_to_parallelise_over = 8
        self.max_num_models_by_shape = {
            # Note dN here requires 2N qubits so d3 counts as shape 6
            1: 0,
            2: 1,
            4: 3,
            5: 2,
            6: 2,
            'other': 0
        }
        self.max_num_qubits = 3
        self.max_num_sites = 5
        self.setup_growth_class()
        self.min_param = 0
        self.max_param = 1

        # self.true_model_terms_params = {
        #     'pauliSet_1J2_xJx_d3': 4.0969217897733703, 
        #     'pauliSet_1J2_zJz_d3': 9.7007310340158401, 
        #     'pauliSet_2J3_xJx_d3':  6.7344876799395417, 
        #     'pauliSet_2J3_yJy_d3': 1.9672493478694473
        # }

        if self.tree_completed_initially == True:
            # to manually fix the models to be considered
            models = []
            list_connections = [
                [(1, 2)],  # pair of sites
                [(1, 2), (2, 3)],  # chain length 3
                [(1, 2), (1, 3), (3, 4), (2, 4)],  # square,
                [(1, 2), (2, 3), (3, 4)],  # chain,
                [(1, 2), (2, 3), (3, 4), (4, 5)],  # chain,
                # [(1,2), (2,3), (3,4), (4,5), (5,6)], # chain,
                [(1, 2), (1, 4), (2, 3), (2, 5),
                 (3, 6), (4, 5), (5, 6)]  # 3x2 grid
            ]
            for connected_sites in list_connections:

                system_size = max(max(connected_sites))
                terms = connected_lattice.pauli_like_like_terms_connected_sites(
                    connected_sites=connected_sites,
                    base_terms=['x', 'y', 'z'],
                    num_sites=system_size
                )

                p_str = 'P' * system_size
                models.append(p_str.join(terms))

            # self.initial_models = models
            # testing models which win full search
            self.true_model = 'pauliSet_1J2_xJx_d4PPPPpauliSet_1J3_zJz_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_3J4_xJx_d4PPPPpauliSet_3J4_zJz_d4'
            self.initial_models = [
                self.true_model, 
                'pauliSet_1J2_xJx_d3PPPpauliSet_1J2_zJz_d3PPPpauliSet_1J3_xJx_d3PPPpauliSet_1J3_zJz_d3',
                'pauliSet_1J2_xJx_d4PPPPpauliSet_1J2_zJz_d4PPPPpauliSet_1J3_xJx_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_2J4_zJz_d4PPPPpauliSet_3J4_xJx_d4',
                'pauliSet_1J2_xJx_d4PPPPpauliSet_1J2_zJz_d4PPPPpauliSet_1J3_xJx_d4PPPPpauliSet_3J4_xJx_d4',
                'pauliSet_1J2_xJx_d4PPPPpauliSet_1J2_zJz_d4PPPPpauliSet_1J3_zJz_d4PPPPpauliSet_2J4_xJx_d4PPPPpauliSet_3J4_xJx_d4'
            ]

            self.true_model_terms_params = {
                'pauliSet_1J2_xJx_d4': 0.27044671107574969, 
                'pauliSet_1J3_zJz_d4': 1.1396665426731736, 
                'pauliSet_2J4_xJx_d4': 0.38705331216054806, 
                'pauliSet_3J4_xJx_d4': 0.46892509638460805, 
                'pauliSet_3J4_zJz_d4': 0.45440765993845578
            }

            if self.true_model not in self.initial_models:
                self.log_print("Adding true operator to initial model list")
                self.initial_models.append(self.true_model)
        


        self.max_num_models_by_shape = {
            3: 10,
            'other': 0
        }
