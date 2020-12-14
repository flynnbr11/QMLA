import sys
import os

from qmla.exploration_strategies.nv_centre_spin_characterisation import nv_centre_full_access


class ExperimentReducedNV(
    nv_centre_full_access.FullAccessNVCentre
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
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

        self.overwrite_exploration_class_methods(
            **kwargs
        )

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        # from qmla import model_generation
        
        new_mods = ExperimentReducedNVal_method(
            model_list,
            **kwargs
        )
        return new_mods


def ExperimentReducedNVal_method(
    model_list,
    spawn_step,
    model_dict,
    log_file,
    **kwargs
):
    # TODO this will be broken b/c removing spawn stage and spawn step from QMLA 
    # call to generate_models
    """
    For use only during development to minimise time taken for testing.
    """
    if spawn_step == 1:
        return ['xTiPPxTxPPyTiPPyTyPPzTiPPzTz']
    elif spawn_step == 2:
        kwargs['spawn_stage'].append('Complete')
        return [
            'xTiPPxTxPPxTzPPyTiPPyTyPPzTiPPzTz',
            'xTiPPxTxPPyTiPPyTyPPyTzPPzTiPPzTz',
            'xTiPPxTxPPxTyPPyTiPPyTyPPzTiPPzTz'
        ]

