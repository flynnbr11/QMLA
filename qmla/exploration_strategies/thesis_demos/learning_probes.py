import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy
from qmla.exploration_strategies.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined
from qmla import construct_models
import qmla.shared_functionality.topology_predefined as topologies

class DemoProbes(
    exploration_strategy.ExplorationStrategy
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

        self.qhl_models = [
            "pauliSet_1_z_d1", 
            "pauliSet_1_x_d1+pauliSet_1_y_d1+pauliSet_1_z_d1", 
            "pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5", # ising
            "pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4" # heisenberg
        ]
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.separable_probe_dict

        self._shared_true_parameters = False # test different models at each instance
        if self._shared_true_parameters:
            true_model_idx = 0
        else:
            true_model_idx = self.qmla_id % len(self.qhl_models)  
        self.log_print(["self._shared_true_parameters = {} true_model_idx = {}".format(self._shared_true_parameters, true_model_idx)] )

        self.true_model = self.qhl_models[true_model_idx]

class DemoProbesTomographic(
    DemoProbes
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
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.tomographic_basis

class DemoProbesPlus(
    DemoProbes
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
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_probes_dict        