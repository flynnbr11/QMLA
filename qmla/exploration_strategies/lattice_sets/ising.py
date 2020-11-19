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
        self._shared_true_parameters = True

        # self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        self.timing_insurance_factor = 0.2
        
