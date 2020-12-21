import random
import sys
import os

import pickle 

from qmla.exploration_strategies.nv_centre_spin_characterisation.experimental_paper import FullAccessNVCentre
import qmla.shared_functionality.qinfer_model_interface
import qmla.shared_functionality.probe_set_generation
import  qmla.shared_functionality.experiment_design_heuristics
import qmla.shared_functionality.expectation_value_functions
from qmla import construct_models


__all__ = [
    'NVCentreSimulatedExperiment',
]

class NVCentreSimulatedExperiment(
    FullAccessNVCentre  # inherit from this
):
    r"""
    Uses all the same functionality, growth etc as
    default FullAccessNVCentre,
    but uses an expectation value which traces out 
    the environment, mimicing the Hahn echo measurement. 

    This is used to generate (ii) simulated data in the experimental paper. 
    """

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        
        # print("[Exploration Strategies] init nv_spin_experiment_full_tree")
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        self.true_model = 'xTiPPyTiPPzTiPPzTz'

        self.initial_models = ['xTi', 'yTi', 'zTi']
        self.tree_completed_initially = False
        self.qhl_models = [
            # 'xTiPPxTxPPxTyPPxTzPPyTiPPyTyPPyTzPPzTiPPzTz',
            'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
            'xTiPPyTiPPzTiPPzTz',
            'xTiPPyTiPPyTyPPzTiPPzTz',
            # 'yTi'
        ]
        self.max_num_parameter_estimate = 9
        self.max_spawn_depth = 8
        self.max_num_qubits = 3
        self.fixed_axis_generator = False
        self.fixed_axis = 'z'  # e.g. transverse axis

        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0.1
        self.fraction_particles_for_bf = 0.1 # testing whether reduced num particles for BF can work 


        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.NV_centre_ising_probes_plus
        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.NV_centre_ising_probes_plus
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probes_generation_subroutine = self.system_probes_generation_subroutine
        self.shared_probes = False
        self.max_time_to_consider = 5

        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.separable_probe_dict

        # params for testing p value calculation
        self.max_num_probe_qubits = 2
        self.gaussian_prior_means_and_widths = {
        }

        if self.true_model == 'xTiPPyTiPPzTiPPzTz':
            self.true_model_terms_params = {  # from Jul_05/16_40
                'xTi': 0.92450565,
                'yTi': 6.00664336,
                'zTi': 1.65998543,
                'zTz': 0.76546868,
            }
        self.gaussian_prior_means_and_widths = {
            'xTi': [4.0, 1.5],
            'yTi': [4.0, 1.5],
            'zTi': [4.0, 1.5],
            'xTx': [4.0, 1.5],
            'yTy': [4.0, 1.5],
            'zTz': [4.0, 1.5],
            'xTy': [4.0, 1.5],
            'xTz': [4.0, 1.5],
            'yTz': [4.0, 1.5],
        }

        self.max_num_models_by_shape = {
            1: 0,
            2: 18,
            'other': 1
        }


