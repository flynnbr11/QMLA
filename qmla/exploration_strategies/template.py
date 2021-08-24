import sys
import os

from qmla.exploration_strategies import exploration_strategy


class ExplorationStrategyTemplate(
    exploration_strategy.ExplorationStrategy  # inherit from this
):
    r"""
    Template exploration strategy
    """

    def __init__(self, exploration_rules, **kwargs):
        # print("[Exploration Strategies] init nv_spin_experiment_full_tree")
        super().__init__(exploration_rules=exploration_rules, **kwargs)
        self.initial_models = None
        self.tree_completed_initially = False
        self.true_model = "pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2+pauliSet_1J2_zJz_d2"
        self.true_model_terms_params = {
            "pauliSet_1J2_xJx_d2": 0.1,
            "pauliSet_1J2_yJy_d2": 0.9,
            "pauliSet_1J2_zJz_d2": 0.3,
        }

    def generate_models(self, **kwargs):
        if self.spawn_step == 0:
            new_models = [
                "pauliSet_1J2_xJx_d2",
                "pauliSet_yJy_1J2_d2",
                "pauliSet_1J2_zJz_d2",
            ]
        elif self.spawn_step == 1:
            new_models = [
                "pauliSet_1J2_xJx_d2+pauliSet_zJz_1J2_d2",
                "pauliSet_1J2_yJy_d2+pauliSet_zJz_1J2_d2",
                "pauliSet_1J2_xJx_d2+pauliSet_zJz_1J2_d2",
            ]
        elif self.spawn_step == 2:
            new_models = ["pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2+pauliSet_zJz_1J2_d2"]
            self.spawn_stage.append("Complete")
        return new_models

    def check_tree_completed(self, spawn_step, **kwargs):
        if self.spawn_stage[-1] == "Complete":
            return True
        else:
            return False


class TestInstall(exploration_strategy.ExplorationStrategy):  # inherit from this
    r"""
    Template exploration strategy
    """

    def __init__(self, exploration_rules, **kwargs):
        # print("[Exploration Strategies] init nv_spin_experiment_full_tree")
        super().__init__(exploration_rules=exploration_rules, **kwargs)
        self.initial_models = None
        self.tree_completed_initially = True
        self.initial_models = [
            "pauliSet_1J2_xJx_d2"
            "pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2"
            "pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2+pauliSet_1J2_zJz_d2"
        ]
        self.true_model = "pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2+pauliSet_1J2_zJz_d2"
        self.true_model_terms_params = {
            "pauliSet_1J2_xJx_d2": 0.1,
            "pauliSet_1J2_yJy_d2": 0.9,
            "pauliSet_1J2_zJz_d2": 0.3,
        }
