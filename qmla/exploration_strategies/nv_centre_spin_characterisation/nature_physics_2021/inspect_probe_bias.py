from __future__ import absolute_import
import sys
import os
import random

import qmla.model_building_utilities
from qmla.exploration_strategies.nv_centre_spin_characterisation.nature_physics_2021 import FullAccessNVCentre

__all__ = [
    'InspectProbeBiasNVCentre'
]

class InspectProbeBiasNVCentre(
    FullAccessNVCentre
):

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        r"""
        Cycle through target model.

        """
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        self.true_model = 'xTi+yTi+zTi+xTx+yTy+zTz'
        # self.true_model = 'xTi+yTi+zTi+zTz'

        if self.true_model == 'xTi+yTi+zTi+zTz':
            self.true_model_terms_params = {
                'xTi': 0.92450565,
                'yTi': 6.00664336,
                'zTi': 1.65998543,
                'zTz': 0.76546868,
            }
        else:
            self.true_model_terms_params = {
                'xTi' : -0.98288958683093952,
                'xTx' : 6.7232235286284681,  
                'yTi' : 6.4842202054983122,  
                'yTy' : 2.7377867056770397,  
                'zTi' : 0.96477790489201143, 
                'zTz' : 1.6034234519563935, 
                'xTy' : 1.5,
                'xTz' : 7, 
                'yTz' : 2 
            }

        self.max_time_to_consider = 5
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MultiParticleGuessHeuristic
        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution_double_time_reverse

        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.hahn_sequence_random_initial
        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probes_generation_subroutine = self.system_probes_generation_subroutine
        self.shared_probes = False
