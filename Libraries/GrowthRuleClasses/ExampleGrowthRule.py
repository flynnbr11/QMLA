import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues
import ProbeGeneration

import SuperClassGrowthRule

class example_growth(
    SuperClassGrowthRule.growth_rule_super_class # inherit from this
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
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        self.initial_models = ['pauliSet_xJx_1J2_d2']
        self.tree_completed_initially = False
        self.true_operator = 'pauliSet_1J2_zJz_d2PPpauliSet_1J2_xJx_d2PPpauliSet_1J2_yJy_d2'
        self.true_params = {
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
        # default is to just return given model list and set spawn stage to complete
        if self.spawn_stage[-1] == None:
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