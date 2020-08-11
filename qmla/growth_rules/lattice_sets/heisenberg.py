import numpy as np
import itertools
import sys
import os

from qmla.growth_rules.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined

class HeisenbergLatticeSet(
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
        self.transverse_field = None # 'x'
        self.true_lattice = topology_predefined._4_site_square_fully_connected
        self.true_model = self.model_from_lattice(self.true_lattice)

        self.available_lattices = [
            self.true_lattice, # 4 site chain

            # chains
            topology_predefined._3_site_chain,
            topology_predefined._4_site_chain,
            topology_predefined._5_site_chain,
            topology_predefined._6_site_chain,

            # other lattices
            topology_predefined._4_site_square_fully_connected, 
            topology_predefined._5_site_lattice_fully_connected, 
            topology_predefined._6_site_grid
        ]
        self.max_num_models_by_shape = {
            1 : 2, 
            2 : 2, 
            3 : 2, 
            4 : 2, 
            5 : 2, 
            6 : 2, 
            'other' : 0
        }
        self.base_terms = ['x', 'z']
        self.num_processes_to_parallelise_over = len(self.available_lattices)
        self.timing_insurance_factor = 0.5