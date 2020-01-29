import random
import sys
import os

from qmla.GrowthRuleClasses import NVCentreExperimentGrowthRules
from qmla import ProbeGeneration
from qmla import ExpectationValues
from qmla import DataBase


class presentation_plot_generation(
    NVCentreExperimentGrowthRules.nv_centre_spin_experimental_method  # inherit from this
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

        self.probe_generation_function = ProbeGeneration.separable_probe_dict
        # unless specifically different set of probes required
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = True  # i.e. system and simulator get same probes for learning

        self.single_parameter = False

        self.true_operator = 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz'
        # self.max_num_probe_qubits = 3
        # self.true_operator = 'yTxTTz'

        self.true_params = {
            # Decohering param set
            # From 3000exp/20000prt, BC SelectedRuns/Nov_28/15_14/results_049
            'xTi': -0.98288958683093952,  # -0.098288958683093952
            'xTx': 6.7232235286284681,  # 0.67232235286284681,
            'yTi': 6.4842202054983122,  # 0.64842202054983122, #
            'yTy': 2.7377867056770397,  # 0.27377867056770397,
            'zTi': 0.96477790489201143,  # 0.096477790489201143,
            'zTz': 1.6034234519563935,  # 0.16034234519563935,
        }

        self.qhl_models = [
            'zTi',
            'xTiPPyTiPPzTiPPzTz',
            # self.true_operator
        ]
        self.max_time_to_consider = 6

        self.gaussian_prior_means_and_widths = {
            'xTi': (0.9, 0.1),  # -0.098288958683093952
            'xTx': (6.7, 0.1),  # 0.67232235286284681,
            'yTi': (6.0, 0.1),  # 0.64842202054983122, #
            'yTy': (2.7, 0.1),  # 0.27377867056770397,
            'zTi': (1.65, 0.1),  # 0.096477790489201143,
            'zTz': (0.76, 0.1)  # 0.16034234519563935,
        }
        # self.true_params = { # from Jul_05/16_40
        #     'xTi': 0.92450565,
        #     'yTi': 6.00664336,
        #     'zTi': 1.65998543,
        #     'zTz': 0.76546868,
        # }

        if self.single_parameter == True:
            self.max_time_to_consider = 2
            self.true_operator = 'zTi'
            self.true_params = {
                'zTi': 6.5
            }
            self.gaussian_prior_means_and_widths = {
                'zTi': (5, 2.5)
            }
