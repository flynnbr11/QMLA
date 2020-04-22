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
        self.true_lattice = topology_predefined._4_site_square
        self.true_model = self.model_from_lattice(self.true_lattice)

        self.available_lattices = [
            self.true_lattice, 
            topology_predefined._3_site_chain, 
            topology_predefined._4_site_square, 
            topology_predefined._5_site_lattice_fully_connected,
            topology_predefined._6_site_grid, 
        ]
        self.base_terms = ['x', 'z']
        