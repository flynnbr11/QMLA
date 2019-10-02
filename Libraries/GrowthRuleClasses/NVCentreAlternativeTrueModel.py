import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues
import ProbeGeneration

import NVCentreExperimentGrowthRules

class nv_centre_spin_experimental_method_alternative_true_model(
    NVCentreExperimentGrowthRules.nv_centre_spin_experimental_method # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )


        self.true_operator = 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz'
        # self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus
        # self.probe_generation_function = ProbeGeneration.separable_probe_dict
        # self.shared_probes = True


        if self.true_operator == 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz':
            self.true_params = {
                # Decohering param set
                # From 3000exp/20000prt, BC SelectedRuns/Nov_28/15_14/results_049
                'xTi': -0.98288958683093952, # -0.098288958683093952
                'xTx': 6.7232235286284681, # 0.67232235286284681,  
                'yTi': 6.4842202054983122,  # 0.64842202054983122, # 
                'yTy': 2.7377867056770397,  # 0.27377867056770397, 
                'zTi': 0.96477790489201143, # 0.096477790489201143, 
                'zTz': 1.6034234519563935, #0.16034234519563935,
            }


class nv_centre_spin_experimental_method_alternative_true_model_second(
    NVCentreExperimentGrowthRules.nv_centre_spin_experimental_method # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )

        self.true_operator = 'xTiPPxTxPPxTyPPyTiPPyTyPPzTiPPzTz'
        self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus
        self.simulator_probe_generation_function = self.probe_generation_function # unless specifically different set of probes required
        # self.probe_generation_function = ProbeGeneration.separable_probe_dict



