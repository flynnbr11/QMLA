import sys
import os

from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.experiment_design_heuristics
import qmla.shared_functionality.probe_set_generation
from qmla import construct_models


def test_evolution_fnc(**kwargs):
    print("In test fnc")
    return 1

class GRTest(
    exploration_strategy.ExplorationStrategy  # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        # print("[Exploration Strategies] init nv_spin_experiment_full_tree")
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.log_print(["Initialising test GR"])
        self.expectation_value_function = test_evolution_fnc
        self.initial_models = ['pauliSet_xJx_1J2_d2']
        self.tree_completed_initially = False
        self.true_model = 'pauliSet_1J2_zJz_d2+pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2'
        self.true_model_terms_params = {
            'pauliSet_1J2_zJz_d2': 0.3,
            'pauliSet_1J2_xJx_d2': 0.1,
            'pauliSet_1J2_yJy_d2': 0.9
        }
        self.spawn_stage = [None]

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        # default is to just return given model list and set spawn stage to
        # complete
        if self.spawn_stage[-1] is None:
            new_models = [
                'pauliSet_yJy_1J2_d2'
            ]
            self.spawn_stage.append('round one')
        elif self.spawn_stage[-1] == 'round one':
            new_models = [
                'pauliSet_zJz_1J2_d2'
            ]
            self.spawn_stage.append('Complete')
        return new_models

    def check_tree_completed(
        self,
        spawn_step,
        **kwargs
    ):
        if self.spawn_stage[-1] == 'Complete':
            return True
        else:
            return False

