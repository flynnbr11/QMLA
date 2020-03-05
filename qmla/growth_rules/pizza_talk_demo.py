import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, connected_lattice_probabilistic
from qmla import experiment_design_heuristics
from qmla import topology
from qmla import model_naming
from qmla import probe_set_generation
from qmla import database_framework


class TalkDemonstration(
    connected_lattice.ConnectedLattice,
):
    def __init__(
        self,
        growth_generation_rule, 
        **kwargs
    ):
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
        self.four_site_xxz = 'pauliSet_1J2_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4'
        self.true_model = self.four_site_xxz
        self.true_model = database_framework.alph(self.true_model)
        self.qhl_models = [
            self.true_model,
        ]
        self.base_terms = [
            'x',
            # 'y',
            'z'
        ]
        self.num_probes = 50
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

                # p_str = 'P' * system_size
                # p_str = '+'

                models.append(
                    # p_str.join(terms)
                    self.combine_terms(terms)
                )

            self.initial_models = models


    def combine_terms(
        self,
        terms,
    ):
        addition_string = '+'
        terms = sorted(terms)
        return addition_string.join(terms)
