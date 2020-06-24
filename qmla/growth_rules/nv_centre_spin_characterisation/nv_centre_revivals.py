import sys
import os

import pickle 

from qmla.growth_rules.nv_centre_spin_characterisation import nv_centre_large_spin_bath
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla import construct_models

class NVCentreRevivalsSimulated(
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
        # self.true_model = qmla.utilities.n_qubit_nv_gali_model(
        #     n_qubits = 2, 
        #     rotation_terms = ['x', 'y', 'z' ], 
        #     coupling_terms = ['z']
        # )
        # self.true_model = 'pauliSet_1_z_d2+pauliSet_1J2_zJz_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2'
        # self.true_model = qmla.construct_models.alph(self.true_model) 
    
        self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        # self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution_double_time_reverse # as experiment
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        self.simulator_probe_generation_function = self.probe_generation_function
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        self.shared_probes = False
        self.probe_noise_level = 0 
        
        self.qhl_models = [
            # qmla.utilities.n_qubit_nv_gali_model(n_qubits = 2, coupling_terms=['z']),
            qmla.utilities.n_qubit_nv_gali_model(n_qubits = 3, coupling_terms=['z']),
            qmla.utilities.n_qubit_nv_gali_model(n_qubits = 4, coupling_terms=['z']),
        ]

        self.min_param = 0
        self.max_param = 20
        self.gaussian_prior_means_and_widths = {
            'pauliSet_1J3_zJz_d3' : (0.5, 0.2),
            'pauliSet_1J3_zJz_d4' : (0.5, 0.2),
            'pauliSet_1J4_zJz_d4' : (0.5, 0.2),
        }

        short_data_rotation_x_term =  0.92450565
        short_data_rotation_y_term = 6.00664336
        short_data_rotation_z_term = 1.65998543
        short_data_coupling_z = 0.76546868

        self.max_time_to_consider = 50e-6
        self.plot_time_increment = 0.5e-6

        n_qubits = 5
        self.true_model_terms_params = {
            # spin
            # 'pauliSet_1_x_d{}'.format(n_qubits) : 1.94e9,
            'pauliSet_1_y_d{}'.format(n_qubits) : 1.94e9,
            'pauliSet_1_z_d{}'.format(n_qubits) : 1.94e9,
            
            # coupling
            'pauliSet_1J2_zJz_d{}'.format(n_qubits) : 2.14e6, 
            'pauliSet_1J3_zJz_d{}'.format(n_qubits) : 2.14e6, 
            'pauliSet_1J4_zJz_d{}'.format(n_qubits) : 2.14e6, 
            'pauliSet_1J5_zJz_d{}'.format(n_qubits) : 2.14e6, 

            'pauliSet_1J2_yJy_d{}'.format(n_qubits) : 2.14e6, 
            'pauliSet_1J3_yJy_d{}'.format(n_qubits) : 2.14e6, 
            'pauliSet_1J4_yJy_d{}'.format(n_qubits) : 2.14e6, 
            'pauliSet_1J5_yJy_d{}'.format(n_qubits) : 2.14e6, 

            # nuclear 
            'pauliSet_2_x_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_2_y_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_2_z_d{}'.format(n_qubits) : 3.5e9,

            # nuclear 3rd qubit
            'pauliSet_3_x_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_3_y_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_3_z_d{}'.format(n_qubits) : 3.5e9,

            # nuclear 3rd qubit
            'pauliSet_4_x_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_4_y_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_4_z_d{}'.format(n_qubits) : 3.5e9,

            # nuclear 3rd qubit
            'pauliSet_5_x_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_5_y_d{}'.format(n_qubits) : 3.5e9,
            'pauliSet_5_z_d{}'.format(n_qubits) : 3.5e9,

        }

        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)



class NVCentreRevivals(
    NVCentreRevivalsSimulated
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
        self.qinfer_model_class =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment


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