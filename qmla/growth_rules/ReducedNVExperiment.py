import sys
import os

from qmla.growth_rules import NVCentreFullAccess


class reduced_nv_experiment(
    NVCentreFullAccess.nv_centre_spin_full_access
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.max_num_parameter_estimate = 9
        self.max_spawn_depth = 4
        self.max_num_qubits = 2

        self.initial_models = [
            'xTiPPxTxPPyTiPPyTyPPzTi',
            'xTiPPxTxPPyTiPPzTiPPzTz',
            'xTiPPyTiPPyTyPPzTiPPzTz'
        ]

        self.max_num_models_by_shape = {
            2: 7,
            'other': 0
        }

        self.overwrite_growth_class_methods(
            **kwargs
        )

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        import model_generation

        new_mods = model_generation.reduced_nv_experimental_method(
            model_list,
            **kwargs
        )
        return new_mods
