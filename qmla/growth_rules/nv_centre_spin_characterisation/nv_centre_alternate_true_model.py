import sys
import os

from qmla.growth_rules.nv_centre_spin_characterisation import nv_centre_experiment
import qmla.shared_functionality.probe_set_generation
from qmla import expectation_values
from qmla import database_framework


class ExpAlternativeNV(
    nv_centre_experiment.ExperimentNVCentre  # inherit from this
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
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        self.true_model = 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz'
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.NV_centre_ising_probes_plus
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict
        # self.shared_probes = True

        if self.true_model == 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz':
            self.true_model_terms_params = {
                # Decohering param set
                # From 3000exp/20000prt, BC
                # SelectedRuns/Nov_28/15_14/results_049
                'xTi': -0.98288958683093952,  # -0.098288958683093952
                'xTx': 6.7232235286284681,  # 0.67232235286284681,
                'yTi': 6.4842202054983122,  # 0.64842202054983122, #
                'yTy': 2.7377867056770397,  # 0.27377867056770397,
                'zTi': 0.96477790489201143,  # 0.096477790489201143,
                'zTz': 1.6034234519563935,  # 0.16034234519563935,
            }


class ExpAlternativeNV_second(
    nv_centre_experiment.ExperimentNVCentre  # inherit from this
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
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        self.true_model = 'xTiPPxTxPPxTyPPyTiPPyTyPPzTiPPzTz'
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.NV_centre_ising_probes_plus
        # unless specifically different set of probes required
        self.simulator_probe_generation_function = self.probe_generation_function
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict
