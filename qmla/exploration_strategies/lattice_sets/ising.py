import numpy as np
import itertools
import sys
import os

from qmla.exploration_strategies.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined
from qmla import construct_models

class IsingLatticeSet(
    fixed_lattice_set.LatticeSet
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):

        self.base_terms = ['z']
        self.transverse_field = 'x'
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        # self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        # self.true_model_terms_params = {
        #     'pauliLikewise_lx_1_2_3_4_d4': 0.2,
        #     'pauliLikewise_lz_1J2_1J3_1J4_2J3_2J4_3J4_d4': 0.7
        # }
        self.max_time_to_consider = 10
        self.timing_insurance_factor = 0.2
        
class IsingReducedLatticeSet(IsingLatticeSet):

    _vary_true_model = True
    _lattice_names = [
        '_2_site_chain', 
        '_3_site_chain', 
        '_4_site_chain', 

        '_3_site_lattice_fully_connected', 
        '_4_site_lattice_fully_connected',
        '_4_site_square',
    ]

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):

        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
            