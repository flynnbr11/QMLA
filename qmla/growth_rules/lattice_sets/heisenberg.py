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
        self.transverse_field = None # 'x'
        self.base_terms = ['x', 'y', 'z']
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.timing_insurance_factor = 0.4