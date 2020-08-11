import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, growth_rule
from qmla.growth_rules.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined
from qmla import construct_models

class IsingLatticeSet(
    fixed_lattice_set.LatticeSet
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
        self.base_terms = ['z']
        self.transverse_field = 'x'

        self.true_lattice = topology_predefined._4_site_chain
        self.true_model = self.model_from_lattice(self.true_lattice)

        self.available_lattices = [
            self.true_lattice, # 4 site chain

            # Ising chains
            topology_predefined._3_site_chain,
            topology_predefined._5_site_chain,
            topology_predefined._6_site_chain,

            # other lattices
            topology_predefined._4_site_square_fully_connected, 
            topology_predefined._5_site_lattice_fully_connected, 
            topology_predefined._6_site_grid

        ]

        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = self.fraction_own_experiments_for_bf

        self.timing_insurance_factor = 4
        self.true_model_terms_params = {
            'pauliLikewise_lz_1J2_2J3_3J4_d4' : 0.78,
            'pauliLikewise_lx_1_2_3_4_d4' : 0.12,
            # 'pauliLikewise_lz_1J2_d2' : 0.78,
            # 'pauliLikewise_lx_1_2_d2' : 0.12,
        }
        self.gaussian_prior_means_and_widths = {
            # 'pauliLikewise_lz_1J2_d2' : (0.78, 1e-5),
            # 'pauliLikewise_lx_1_2_d2' : (0.12, 1e-5)
        }
        self.num_processes_to_parallelise_over = len(self.available_lattices)
        self.max_num_models_by_shape = {
            3 : 2, 
            4 : 2, 
            5 : 2, 
            6 : 2, 
            'other' : 0
        }
        self.timing_insurance_factor = 0.4
        
