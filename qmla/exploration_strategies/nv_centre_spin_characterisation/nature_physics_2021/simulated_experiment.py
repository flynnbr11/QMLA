import random
import sys
import os

import pickle 

from qmla.exploration_strategies.nv_centre_spin_characterisation.nature_physics_2021 import FullAccessNVCentre, TieredGreedySearchNVCentre
import qmla.shared_functionality.qinfer_model_interface
import qmla.shared_functionality.probe_set_generation
import  qmla.shared_functionality.experiment_design_heuristics
import qmla.shared_functionality.expectation_value_functions
from qmla import construct_models


__all__ = [
    'SimulatedExperimentNVCentre',
]

class SimulatedExperimentNVCentre(
    TieredGreedySearchNVCentre
    # FullAccessNVCentre  # inherit from this
):
    r"""
    Uses all the same functionality, growth etc as
    :class:`~qmla.exploration_strategies.nv_centre_spin_characterisation.TieredGreedySearchNVCentre`,
    but uses an expectation value which traces out the environment, mimicing the Hahn echo measurement. 

    This is used to generate (ii) simulated data in the Nature Physics 2021 paper. 
    """

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        self.true_model = 'xTi+yTi+zTi+zTz'
        self.qhl_models = ['xTi', 'yTi', 'zTi']

        # self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution_double_time_reverse
        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.hahn_via_z_pi_gate
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probes_generation_subroutine = self.system_probes_generation_subroutine
        self.shared_probes = False
        self.max_time_to_consider = 5


