import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues

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
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )

        self.true_operator = 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz'
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
        self.measurement_type = 'hahn'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z' # e.g. transverse axis

        self.max_num_models_by_shape = {
            1 : 0,
            2 : 18, 
            'other' : 1
        }

    def expectation_value(
        self, 
        ham,
        t,
        state,
        **kwargs
    ):      
        # print("[Growth Rules - NV] Expectation Values")
        exp_val = ExpectationValues.n_qubit_hahn_evolution(
            ham = ham, 
            t = t, 
            state = state, 
            **kwargs
        )
        return exp_val

