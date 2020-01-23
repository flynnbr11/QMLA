import NVCentreLargeSpinBath
import ProbeGeneration
import ExpectationValues
import DataBase
import sys
import os
sys.path.append(os.path.abspath('..'))


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
        import Heuristics
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.experimental_dataset = 'NV_revivals.p'
        self.true_operator = 'nv_spin_x_d6PPPPPPnv_spin_y_d6PPPPPPnv_spin_z_d6PPPPPPnv_interaction_x_d6PPPPPPnv_interaction_y_d6PPPPPPnv_interaction_z_d6'

        self.qhl_models = [
            NV_centre_large_spin_bath.gali_model_nv_centre_spin(2),
            NV_centre_large_spin_bath.gali_model_nv_centre_spin(6),
            NV_centre_large_spin_bath.gali_model_nv_centre_spin(7),
            # NV_centre_large_spin_bath.gali_model_nv_centre_spin(8),
            # 'nv_spin_x_d3PPPnv_spin_y_d3PPPnv_spin_z_d3PPPnv_interaction_x_d3PPPnv_interaction_y_d3PPPnv_interaction_z_d3',
            # 'nv_spin_x_d4PPPPnv_spin_y_d4PPPPnv_spin_z_d4PPPPnv_interaction_x_d4PPPPnv_interaction_y_d4PPPPnv_interaction_z_d4',
            # 'nv_spin_x_d6PPPPPPnv_spin_y_d6PPPPPPnv_spin_z_d6PPPPPPnv_interaction_x_d6PPPPPPnv_interaction_y_d6PPPPPPnv_interaction_z_d6',
            # 'nv_spin_x_d6PPPPPPnv_spin_y_d6PPPPPPnv_spin_z_d6PPPPPPnv_interaction_x_d6PPPPPPnv_interaction_y_d6PPPPPPnv_interaction_z_d6',
        ]

        self.plot_probe_generation_function = ProbeGeneration.plus_probes_dict
        if self.use_experimental_data == True:
            self.probe_generation_function = ProbeGeneration.restore_dec_13_probe_generation
