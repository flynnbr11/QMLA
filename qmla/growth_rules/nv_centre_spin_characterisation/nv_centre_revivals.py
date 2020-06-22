import sys
import os

import pickle 

from qmla.growth_rules.nv_centre_spin_characterisation import nv_centre_large_spin_bath
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla import construct_models

class NVCentreRevivals(
    nv_centre_large_spin_bath.NVLargeSpinBath  # inherit from this
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        # self.true_model = 'nv_spin_x_d6+nv_spin_y_d6+nv_spin_z_d6+nv_interaction_x_d6+nv_interaction_y_d6+nv_interaction_z_d6'        
        self.true_model = qmla.utilities.n_qubit_nv_gali_model(n_qubits = 2, coupling_terms = [])
        self.true_model = qmla.construct_models.alph(self.true_model) 

    
        self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        self.qinfer_model_class =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic

        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        self.simulator_probe_generation_function = self.probe_generation_function
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        self.shared_probes = False
        self.max_time_to_consider = 56.35
        self.qhl_models = [
            qmla.utilities.n_qubit_nv_gali_model(2),
            qmla.utilities.n_qubit_nv_gali_model(4),

            # nv_centre_large_spin_bath.gali_model_nv_centre_spin(2),
            # nv_centre_large_spin_bath.gali_model_nv_centre_spin(6),
            # nv_centre_large_spin_bath.gali_model_nv_centre_spin(7),
        ]

        self.min_param = 0
        self.max_param = 100

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
                # 'data/NV05_rescale_dataset.p'
                'data/NV_revivals.p'
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