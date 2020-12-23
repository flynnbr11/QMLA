import random
import sys
import os

import pickle 

from qmla.exploration_strategies.nv_centre_spin_characterisation.experimental_paper import SimulatedExperimentNVCentre
import qmla.shared_functionality.qinfer_model_interface
import qmla.shared_functionality.probe_set_generation
import  qmla.shared_functionality.experiment_design_heuristics
import qmla.shared_functionality.expectation_value_functions
from qmla import construct_models


__all__ = [
    'NVCentreExperimentalData'
]


class NVCentreExperimentalData(
    SimulatedExperimentNVCentre
):
    r"""
    Study experimental data.

    Uses the same model generation/comparison strategies as 
    SimulatedExperimentNVCentre, FullAccessNVCentre, 
    but targets data measured from a real system. 
    This is done by using an alternative qinfer_model_subroutine,   
    which searches in the dataset for the system's likelihood, 
    rather than computing it. 
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
        # TODO this is a hack - there is no true model so this generaates true parameter
        # for an unused term so it doesn't interfere
        # this should be looked after by not having a true model in these cases (?)
        # self.true_model = 'xTiPPyTiPPzTiPPzTz'
        # self.true_model = 'xTi+yTi+zTi+zTz'
        self.true_model = 'xTi+xTx+yTi+yTy+zTi+zTz+xTy+xTz+yTz'
        
        # self.true_model = 'iTi'
        # self.max_spawn_depth = 3
        self.true_model = qmla.construct_models.alph(self.true_model) 
        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution_double_time_reverse
        self.qinfer_model_subroutine =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probes_generation_subroutine = self.system_probes_generation_subroutine
        self.shared_probes = False
        self.probe_noise_level = 1e-3
        self.max_time_to_consider = 4.24

    # def get_true_parameters(
    #     self,
    # ):        
    #     self.fixed_true_terms = True
    #     self.true_hamiltonian = None
    #     self.true_params_dict = {}
    #     self.true_params_list = []


    def get_measurements_by_time(
        self
    ):
        data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", 
                "data/NVB_rescale_dataset.p"
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