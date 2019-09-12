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

        # self.true_operator = 'xTiPPxTxPPxTyPPyTiPPyTyPPzTiPPzTz'
        # self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus
        self.probe_generation_function = ProbeGeneration.separable_probe_dict



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
        # self.probe_generation_function = ProbeGeneration.NV_centre_ising_probes_plus
        self.probe_generation_function = ProbeGeneration.separable_probe_dict



