import sys
import os

from qmla.growth_rules.nv_centre_spin_characterisation import nv_centre_large_spin_bath
import qmla.shared_functionality.probe_set_generation
from qmla import expectation_values
from qmla import database_framework
from qmla import experiment_design_heuristics

class ExpNVRevivals(
    nv_centre_large_spin_bath.NVLargeSpinBath  # inherit from this
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
        self.true_model = 'nv_spin_x_d6PPPPPPnv_spin_y_d6PPPPPPnv_spin_z_d6PPPPPPnv_interaction_x_d6PPPPPPnv_interaction_y_d6PPPPPPnv_interaction_z_d6'

        self.qhl_models = [
            nv_centre_large_spin_bath.gali_model_nv_centre_spin(2),
            nv_centre_large_spin_bath.gali_model_nv_centre_spin(6),
            nv_centre_large_spin_bath.gali_model_nv_centre_spin(7),
        ]

        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        # if self.use_experimental_data == True:
            # TODO previously using dec_13_probes; what should it be doing??
            # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.restore_dec_13_probe_generation
