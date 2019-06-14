import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues
import ProbeGeneration

from NV_centre_full_access import NVCentreSpinFullAccess

class NVCentreSpinExperimentalMethod(
    NVCentreSpinFullAccess # inherit from this
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

        # self.true_operator = 'xTi'
        # self.heuristic_function = Heuristics.one_over_sigma_then_linspace
        self.measurement_type = 'hahn'
        
        self.initial_models = ['xTi', 'yTi', 'zTi'] 
        self.qhl_models =    	[
            'xTiPPxTxPPxTyPPxTzPPyTiPPyTyPPyTzPPzTiPPzTz',
            'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
            'zTi'
        ]
        self.max_num_parameter_estimate = 9
        self.max_spawn_depth = 8
        self.max_num_qubits = 3
        self.tree_completed_initially = False
        self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z' # e.g. transverse axis
        if self.use_experimental_data == True:
            # probes, prior etc specific to using experimental data
            # print(
            #     "[{}] Experimental data = true".format(
            #     os.path.basename(__file__))
            # )
            self.probe_generation_function = ProbeGeneration.restore_dec_13_probe_generation
            # self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus
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
        else:
            self.gaussian_prior_means_and_widths = {
            }


        self.max_num_models_by_shape = {
            1 : 0,
            2 : 18, 
            'other' : 1
        }
