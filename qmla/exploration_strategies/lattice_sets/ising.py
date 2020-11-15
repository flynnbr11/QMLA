import numpy as np
import itertools
import sys
import os

from qmla.exploration_strategies import connected_lattice, exploration_strategy
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

        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        self.true_model = 'pauliLikewise_lx_1_2_3_4_d4+pauliLikewise_lz_1J2_2J3_3J4_d4'
        # self.true_model = 'pauliLikewise_lx_1_2_d2+pauliLikewise_lz_1J2_d2'
        self.true_model_terms_params = {
            'pauliLikewise_lz_1J2_2J3_3J4_d4' : 0.78,
            'pauliLikewise_lx_1_2_3_4_d4' : 0.12,
        }
        self.timing_insurance_factor = 0.2
        
