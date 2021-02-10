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

class HeisenbergReducedLatticeSet(
    HeisenbergLatticeSet
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self._shared_true_parameters = False
        self.lattice_names = [
            '_2_site_chain', 
            '_3_site_chain', 
            '_4_site_chain', 

            '_3_site_lattice_fully_connected', 
            '_4_site_lattice_fully_connected',
            '_4_site_square',
        ]
            
        self._setup_target_models() # to use updated lattice_names