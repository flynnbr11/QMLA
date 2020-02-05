import sys
import os

from qmla.growth_rules import nv_centre_full_access
from qmla import experiment_design_heuristics
from qmla import probe_set_generation
from qmla import expectation_values
from qmla import database_framework


class NVLargeSpinBath(
    nv_centre_full_access.ExperimentFullAccessNV  # inherit from this
):
    # Uses some of the same functionality as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out
    # and different model generation

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        if 'use_experimental_data' in kwargs:
            self.use_experimental_data = kwargs['use_experimental_data']
        else:
            self.use_experimental_data = False
        if self.use_experimental_data == True:
            # self.probe_generation_function = probe_set_generation.NV_centre_ising_probes_plus
            # self.probe_generation_function = probe_set_generation.restore_dec_13_probe_generation
            self.probe_generation_function = probe_set_generation.plus_probes_dict

        self.expectation_value_function = expectation_values.n_qubit_hahn_evolution

        self.true_operator = 'nv_spin_x_d2PPnv_spin_y_d2PPnv_spin_z_d2PPnv_interaction_x_d2PPnv_interaction_y_d2PPnv_interaction_z_d2'
        # self.true_operator = 'nv_spin_x_d3PPPnv_spin_y_d3PPPnv_spin_z_d3PPPnv_interaction_x_d3PPPnv_interaction_y_d3PPPnv_interaction_z_d3'
        # self.true_operator = 'nv_spin_x_d5PPPPPnv_spin_y_d5PPPPPnv_spin_z_d5PPPPPnv_interaction_x_d5PPPPPnv_interaction_y_d5PPPPPnv_interaction_z_d5'
        # self.true_operator = 'nv_spin_x_d6PPPPPPnv_spin_y_d6PPPPPPnv_spin_z_d6PPPPPPnv_interaction_x_d6PPPPPPnv_interaction_y_d6PPPPPPnv_interaction_z_d6'

        # for testing tracing
        # self.true_operator = 'nv_spin_x_d2PPnv_spin_y_d2PPnv_spin_z_d2'
        # self.true_operator = 'nv_spin_x_d3PPPnv_spin_y_d3PPPnv_spin_z_d3'
        # self.true_operator = 'nv_spin_x_d4PPPPnv_spin_y_d4PPPPnv_spin_z_d4'

        self.initial_models = [
            # self.true_operator,
            'nv_spin_x_d2PPnv_spin_y_d2PPnv_spin_z_d2PPnv_interaction_x_d2PPnv_interaction_y_d2PPnv_interaction_z_d2'
        ]
        self.qhl_models = [
            'nv_spin_x_d2PPnv_spin_y_d2PPnv_spin_z_d2PPnv_interaction_x_d2PPnv_interaction_y_d2PPnv_interaction_z_d2',
            'nv_spin_x_d3PPPnv_spin_y_d3PPPnv_spin_z_d3PPPnv_interaction_x_d3PPPnv_interaction_y_d3PPPnv_interaction_z_d3',
            'nv_spin_x_d4PPPPnv_spin_y_d4PPPPnv_spin_z_d4PPPPnv_interaction_x_d4PPPPnv_interaction_y_d4PPPPnv_interaction_z_d4',
            'nv_spin_x_d5PPPPPnv_spin_y_d5PPPPPnv_spin_z_d5PPPPPnv_interaction_x_d5PPPPPnv_interaction_y_d5PPPPPnv_interaction_z_d5',
            #    'nv_spin_x_d6PPPPPPnv_spin_y_d6PPPPPPnv_spin_z_d6PPPPPPnv_interaction_x_d6PPPPPPnv_interaction_y_d6PPPPPPnv_interaction_z_d6',
            # 'nv_spin_x_d7PPPPPPPnv_spin_y_d7PPPPPPPnv_spin_z_d7PPPPPPPnv_interaction_x_d7PPPPPPPnv_interaction_y_d7PPPPPPPnv_interaction_z_d7',
        ]
        self.max_num_parameter_estimate = 6
        self.max_spawn_depth = 9
        self.max_num_qubits = 6
        # self.plot_probe_generation_function = probe_set_generation.zero_state_probes
        self.min_param = 0
        self.max_param = 10
        # self.dataset = 'NV_revivals.p'

        self.max_num_models_by_shape = {
            1: 0,
            'other': 1
        }

        self.true_model_terms_params = {
            # test tracing out
            # 'nv_spin_x_d2' : -2.98288958683093952,
            # 'nv_spin_y_d2' : 3.4842202054983122,
            # 'nv_spin_z_d2' : 4.96477790489201143,

            # 'nv_spin_x_d3' : -2.98288958683093952,
            # 'nv_spin_y_d3' : 3.4842202054983122,
            # 'nv_spin_z_d3' : 4.96477790489201143,

            # 'nv_spin_x_d4' : -2.98288958683093952,
            # 'nv_spin_y_d4' : 3.4842202054983122,
            # 'nv_spin_z_d4' : 4.96477790489201143,


            # TEST BC result
            # 'nv_spin_x_d2' : -0.29627351871261692,
            # 'nv_spin_y_d2' : 6.0469105229387585,
            # 'nv_spin_z_d2' : 1.4620596579495875,
            # 'nv_interaction_x_d2' : 6.4175839502356062,
            # 'nv_interaction_y_d2' :  3.4929106217322978,
            # 'nv_interaction_z_d2' : 1.8283790959959856,

            # 'nv_spin_x_d2' : -0.19154543341577981,
            # 'nv_spin_y_d2' : 6.5164820324203241,
            # 'nv_spin_z_d2' : 1.0849269715625818,
            # 'nv_interaction_x_d2' : 6.0533305968705937,
            # 'nv_interaction_y_d2' :  2.9629568717976205,
            # 'nv_interaction_z_d2' : 1.8599206790493994,



            # values for simulation
            'nv_spin_x_d2': -0.98288958683093952,
            'nv_spin_y_d2': 6.4842202054983122,
            'nv_spin_z_d2': 0.96477790489201143,
            'nv_interaction_x_d2': 6.7232235286284681,
            'nv_interaction_y_d2': 2.7377867056770397,
            'nv_interaction_z_d2': 1.6034234519563935,

            'nv_spin_x_d3': -0.98288958683093952,
            'nv_spin_y_d3': 6.4842202054983122,
            'nv_spin_z_d3': 0.96477790489201143,
            'nv_interaction_x_d3': 6.7232235286284681,
            'nv_interaction_y_d3': 2.7377867056770397,
            'nv_interaction_z_d3': 1.6034234519563935,

            'nv_spin_x_d4': -0.98288958683093952,
            'nv_spin_y_d4': 6.4842202054983122,
            'nv_spin_z_d4': 0.96477790489201143,
            'nv_interaction_x_d4': 6.7232235286284681,
            'nv_interaction_y_d4': 2.7377867056770397,
            'nv_interaction_z_d4': 1.6034234519563935,

            'nv_spin_x_d5': -0.98288958683093952,
            'nv_spin_y_d5': 6.4842202054983122,
            'nv_spin_z_d5': 0.96477790489201143,
            'nv_interaction_x_d5': 6.7232235286284681,
            'nv_interaction_y_d5': 2.7377867056770397,
            'nv_interaction_z_d5': 1.6034234519563935,

            'nv_spin_x_d6': -0.98288958683093952,
            'nv_interaction_x_d6': 6.7232235286284681,
            'nv_spin_y_d6': 6.4842202054983122,
            'nv_interaction_y_d6': 2.7377867056770397,
            'nv_interaction_z_d6': 1.6034234519563935,
            'nv_spin_z_d6': 0.96477790489201143,


            'nv_spin_x_d10': -0.98288958683093952,
            'nv_interaction_x_d10': 6.7232235286284681,
            'nv_spin_y_d10': 6.4842202054983122,
            'nv_interaction_y_d10': 2.7377867056770397,
            'nv_interaction_z_d10': 1.6034234519563935,
            'nv_spin_z_d10': 0.96477790489201143,
        }

        self.max_num_models_by_shape = {
            'other': 2,
        }

    def generate_models(
        self,
        model_list,
        **kwargs
    ):

            # model_list = kwargs['model_list']
        spawn_step = kwargs['spawn_step']
        spawn_stage = kwargs['spawn_stage']

        print(
            "[model_generation.NV_centre_spin_large_bath]",
            "Spawn stage:", spawn_stage
        )

        max_num_qubits = max(
            [database_framework.get_num_qubits(mod) for mod in model_list]
        )
        new_gali_model = gali_model_nv_centre_spin(max_num_qubits + 1)
        return [new_gali_model]

    def latex_name(
        self,
        name,
        **kwargs
    ):
        from qmla import model_naming
        latex_name = model_naming.large_spin_bath_nv_system_name(
            term=name
        )
        return latex_name

    def name_branch_map(
        self,
        latex_mapping_file,
        **kwargs
    ):
        from qmla import model_naming
        name_map = model_naming.branch_is_num_dims(
            latex_mapping_file=latex_mapping_file,
            **kwargs
        )
        return name_map

    def check_tree_completed(
        self,
        current_num_qubits,
        **kwargs
    ):
        if (
            current_num_qubits
            ==
            self.max_num_qubits
        ):
            return True
        else:
            return False


# Supporting functions

def gali_model_nv_centre_spin(num_qubits):
    axes = ['x', 'y', 'z']
    p_str = 'P' * num_qubits
    model_terms = []
    for axis in axes:
        for contribution in ['interaction', 'spin']:
            model_terms.append(
                'nv_{}_{}_d{}'.format(contribution, axis, num_qubits)
            )

    model = p_str.join(model_terms)
    return model
