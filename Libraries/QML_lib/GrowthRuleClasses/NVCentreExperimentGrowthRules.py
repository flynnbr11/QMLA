import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues
import ProbeGeneration

import NVCentreFullAccess 

class nv_centre_spin_experimental_method(
    NVCentreFullAccess.nv_centre_spin_full_access # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        import Heuristics
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        if self.use_experimental_data == True:
            self.expectation_value_function = ExpectationValues.hahn_evolution
        else:
            self.expectation_value_function = ExpectationValues.n_qubit_hahn_evolution

        # self.true_operator = 'xTiPPyTy'
        self.heuristic_function = Heuristics.one_over_sigma_then_linspace
        self.measurement_type = 'hahn'
        self.true_operator = 'xTiPPyTiPPzTiPPzTz'
        # self.true_operator = 'yTi'

        self.initial_models = ['xTi', 'yTi', 'zTi'] 
        self.qhl_models =    	[
            # 'xTiPPxTxPPxTyPPxTzPPyTiPPyTyPPyTzPPzTiPPzTz',
            # 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
            # 'xTiPPxTxPPyTiPPzTi',
            'xTiPPyTiPPyTyPPzTi',
            'yTi'
        ]
        self.max_num_parameter_estimate = 9
        self.max_spawn_depth = 8
        self.max_num_qubits = 3
        self.tree_completed_initially = False
        self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z' # e.g. transverse axis
        # self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus

        self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus
        self.shared_probes = False
        self.simulator_probe_generation_function = self.probe_generation_function
        # params for testing p value calculation
        # self.gaussian_prior_means_and_widths = {
        #     'xTi' : [0.6, 0.1],
        #     'yTy' : [0.8, 0.1],
        # }

        # self.true_params = {
        #     'xTi' : 0.602, 
        #     'yTy' : 0.799

        # }
        self.true_params = { # from Jul_05/16_40
            'xTi': 0.92450565,
            'yTi': 6.00664336, 
            'zTi': 1.65998543, 
            'zTz': 0.76546868,
        }
        if self.use_experimental_data == True:
            # probes, prior etc specific to using experimental data
            # print(
            #     "[{}] Experimental data = true".format(
            #     os.path.basename(__file__))
            # )
            # self.probe_generation_function = ProbeGeneration.restore_dec_13_probe_generation
            # self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus

            # self.probe_generation_function = ProbeGeneration.plus_probes_dict
            self.gaussian_prior_means_and_widths = {
                'xTi' : [4.0, 1.5],
                'yTi' : [4.0, 1.5],
                'zTi' : [4.0, 1.5],
                'xTx' : [4.0, 1.5],
                'yTy' : [4.0, 1.5],
                'zTz' : [4.0, 1.5],
                'xTy' : [4.0, 1.5],
                'xTz' : [4.0, 1.5],
                'yTz' : [4.0, 1.5],                
            }
        # else:
            # self.true_params = {
            #     # 'yTi' : 0.44
            # }
            # self.gaussian_prior_means_and_widths = {
            #     # 'yTi' : [0.5, 0.1]
            # }
            # TODO remove below setting probes -- used fir a test where simulation only uses |+>
            # self.probe_generation_function = ProbeGeneration.restore_dec_13_probe_generation
            # self.gaussian_prior_means_and_widths = {
            # }
            # self.gaussian_prior_means_and_widths = {
            #     # fix close to true vals for conference plot
            #     'xTi' : [-1, 0.1],
            #     'yTi' : [6.5, 0.1],
            #     'zTi' : [0.96, 0.1],
            #     'xTx' : [6.71, 0.1],
            #     'yTy' : [2.7, 0.1],
            #     'zTz' : [1.6, 0.1],
            #     'xTy' : [4.0, 1.5],
            #     'xTz' : [4.0, 1.5],
            #     'yTz' : [4.0, 1.5],                
            # }
            # self.true_params = {
            #     'zTi' : 2
            # }

            # self.gaussian_prior_means_and_widths = {
            #     'xTi': [0.37, 0.2], 
            #     # 'yTi': [3.34, 0.2], 
            #     'yTy': [3.07, 0.2],
            #     # 'zTi': [1.9, 0.1], 
            #     # 'zTz': [1.9898222688414018, 0.36198177893564248]
            # }



        self.max_num_models_by_shape = {
            1 : 0,
            2 : 18, 
            'other' : 1
        }
