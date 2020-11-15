import numpy as np
import itertools
import sys
import os

from qmla.exploration_strategies.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined

class HeisenbergLatticeSet(
    fixed_lattice_set.LatticeSet
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        self.transverse_field = None # 'x'
        self.base_terms = ['x', 'y', 'z']
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.timing_insurance_factor = 0.2