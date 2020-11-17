import random
import sys
import os

import pickle 

from qmla.exploration_strategies.nv_centre_spin_characterisation import nv_centre_full_access
import qmla.shared_functionality.qinfer_model_interface
import qmla.shared_functionality.probe_set_generation
import  qmla.shared_functionality.experiment_design_heuristics
import qmla.shared_functionality.measurement_probabilities
from qmla import construct_models


__all__ = [
    'ExperimentNVCentre',
    'NVCentreExperimentalData'
]

class ExperimentNVCentre(
    nv_centre_full_access.ExperimentFullAccessNV  # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

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

        self.measurement_probability_function = qmla.shared_functionality.measurement_probabilitiesn_qubit_hahn_evolution
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
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
        # self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z'  # e.g. transverse axis

        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.NV_centre_ising_probes_plus
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.NV_centre_ising_probes_plus
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = False
        self.max_time_to_consider = 5

        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict

        # params for testing p value calculation
        self.max_num_probe_qubits = 2
        self.gaussian_prior_means_and_widths = {
        }

        # self.true_model_terms_params = {
        #     'xTi' : 0.602,
        #     'yTy' : 0.799

        # }
        if self.true_model == 'xTiPPyTiPPzTiPPzTz':
            self.true_model_terms_params = {  # from Jul_05/16_40
                'xTi': 0.92450565,
                'yTi': 6.00664336,
                'zTi': 1.65998543,
                'zTz': 0.76546868,
            }
        # self.gaussian_prior_means_and_widths = {
        #     'xTi': [4.0, 1.5],
        #     'yTi': [4.0, 1.5],
        #     'zTi': [4.0, 1.5],
        #     'xTx': [4.0, 1.5],
        #     'yTy': [4.0, 1.5],
        #     'zTz': [4.0, 1.5],
        #     'xTy': [4.0, 1.5],
        #     'xTz': [4.0, 1.5],
        #     'yTz': [4.0, 1.5],
        # }

        self.max_num_models_by_shape = {
            1: 0,
            2: 18,
            'other': 1
        }


class NVCentreExperimentalData(
    ExperimentNVCentre
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
        # TODO this is a hack - there is no true model so this generaates true parameter
        # for an unused term so it doesn't interfere
        # this should be looked after by not having a true model in these cases (?)
        # self.true_model = 'xTiPPyTiPPzTiPPzTz'
        self.true_model = 'xTi+yTi+zTi+zTz'
        
        # self.true_model = 'iTi'
        # self.max_spawn_depth = 3
        self.true_model = qmla.construct_models.alph(self.true_model) 
        self.measurement_probability_function = qmla.shared_functionality.measurement_probabilitieshahn_evolution
        self.qinfer_model_class =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = False
        self.max_time_to_consider = 4.24

    def get_true_parameters(
        self,
    ):        
        self.fixed_true_terms = True
        self.true_hamiltonian = None
        self.true_params_dict = {}
        self.true_params_list = []


    def get_measurements_by_time(
        self
    ):
        data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'data/NVB_rescale_dataset.p'
            )
        )
        self.log_print([
            "Getting experimental data from {}".format(data_path)
        ])
        self.measurements = pickle.load(
            open(
                data_path,
                'rb'
            )
        )
        return self.measurements