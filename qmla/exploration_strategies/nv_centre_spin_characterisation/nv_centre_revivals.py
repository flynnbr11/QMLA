import sys
import os

import pickle 

from qmla.exploration_strategies.nv_centre_spin_characterisation import nv_centre_large_spin_bath
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla import construct_models

class NVCentreRevivalsSimulated(
    nv_centre_large_spin_bath.NVLargeSpinBath  # inherit from this
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
        self.measurement_probability_function = qmla.shared_functionality.measurement_probabilitiesn_qubit_hahn_evolution
        # self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.SampleOrderMagnitude
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        # self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.tomographic_basis
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

        self.max_time_to_consider = 50e-6
        self.plot_time_increment = 0.5e-6

        n_qubits = 2
        self.true_model_terms_params = {
            # spin
            'pauliSet_1_z_d{}'.format(n_qubits) : 2e9,
            
            # coupling with 2nd qubit
            'pauliSet_1J2_zJz_d{}'.format(n_qubits) : 0.2e6, 
            'pauliSet_1J2_yJy_d{}'.format(n_qubits) : 0.4e6, 
            'pauliSet_1J2_xJx_d{}'.format(n_qubits) : 0.2e6, 

            # carbon nuclei - 2nd qubit
            'pauliSet_2_x_d{}'.format(n_qubits) : 66e3,
            'pauliSet_2_y_d{}'.format(n_qubits) : 66e3,
            'pauliSet_2_z_d{}'.format(n_qubits) : 15e3,
        }
        self.gaussian_prior_means_and_widths = {
            # 'pauliSet_1_z_d{}'.format(n_qubits) : (2e9, 0.02e9),
            
            # 'pauliSet_1J2_zJz_d{}'.format(n_qubits) : (2e5, 0.02e5), 
            # 'pauliSet_1J2_yJy_d{}'.format(n_qubits) : (4e5, 0.02e5), 
            # 'pauliSet_1J2_xJx_d{}'.format(n_qubits) : (2e5, 0.02e5), 
            # # 'pauliSet_1J2_zJz_d{}'.format(n_qubits) : (2e5, 0.02e5), 
            # # 'pauliSet_1J2_yJy_d{}'.format(n_qubits) : (4e5, 0.02e5), 
            # # 'pauliSet_1J2_xJx_d{}'.format(n_qubits) : (2e5, 0.02e5), 

            # 'pauliSet_2_x_d{}'.format(n_qubits) : (6.6e4, 0.02e4),
            # 'pauliSet_2_y_d{}'.format(n_qubits) : (6.6e4, 0.02e4),
            # 'pauliSet_2_z_d{}'.format(n_qubits) : (3e4, 0.02e4),

            'pauliSet_1_z_d{}'.format(n_qubits) : (5e9, 2e9),
            
            'pauliSet_1J2_zJz_d{}'.format(n_qubits) : (5e5, 2e5), 
            'pauliSet_1J2_yJy_d{}'.format(n_qubits) : (5e5, 2e5), 
            'pauliSet_1J2_xJx_d{}'.format(n_qubits) : (5e5, 2e5), 

            'pauliSet_2_x_d{}'.format(n_qubits) : (5e4, 2e4),
            'pauliSet_2_y_d{}'.format(n_qubits) : (5e4, 2e4),
            'pauliSet_2_z_d{}'.format(n_qubits) : (5e4, 2e4),
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
        exploration_rules,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.qinfer_model_class =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference

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
        measurements = pickle.load(
            open(
                data_path,
                'rb'
            )
        )
        # rescale times from microseconds -> absolute seconds
        meas_times = list(measurements.keys())
        for t in meas_times: 
            new_t = t*1e-6 # in absolute seconds
            measurements[new_t] = measurements[t]
            measurements.pop(t)


        self.measurements = measurements
        return self.measurements