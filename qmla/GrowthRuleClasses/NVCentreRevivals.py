import sys
import os

from qmla.GrowthRuleClasses import NVCentreLargeSpinBath
from qmla import ProbeGeneration
from qmla import expectation_values
from qmla import DataBase
from qmla import experiment_design_heuristics

class nv_centre_revival_data(
    NVCentreLargeSpinBath.nv_centre_large_spin_bath  # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # import Heuristics
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.experimental_dataset = 'NV_revivals.p'
        self.true_operator = 'nv_spin_x_d6PPPPPPnv_spin_y_d6PPPPPPnv_spin_z_d6PPPPPPnv_interaction_x_d6PPPPPPnv_interaction_y_d6PPPPPPnv_interaction_z_d6'

        self.qhl_models = [
            NVCentreLargeSpinBath.gali_model_nv_centre_spin(2),
            NVCentreLargeSpinBath.gali_model_nv_centre_spin(6),
            NVCentreLargeSpinBath.gali_model_nv_centre_spin(7),
        ]

        self.plot_probe_generation_function = ProbeGeneration.plus_probes_dict
        if self.use_experimental_data == True:
            self.probe_generation_function = ProbeGeneration.restore_dec_13_probe_generation
