import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.qinfer_model_interface

class ExampleES(
    exploration_strategy.ExplorationStrategy
):

    def __init__(
        self,
        exploration_rules,
        true_model=None,
        **kwargs
    ):
        self.true_model = 'pauliSet_1J2_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_3J4_zJz_d4'
        super().__init__(
            exploration_rules=exploration_rules,
            true_model=self.true_model,
            **kwargs
        )

        self.initial_models = None
        self.max_spawn_depth = 1
        self.true_model_terms_params = {
            'pauliSet_1J2_zJz_d3' : 2.5,
            'pauliSet_2J3_zJz_d3' : 7.5,
            'pauliSet_4J5_zJz_d3' : 3.5,
        }
        self.min_param = 0
        self.max_param = 10

    def generate_models(self, **kwargs):

        self.log_print(["Generating models; spawn step {}".format(self.spawn_step)])
        if self.spawn_step == 0:
            # chains up to 4 sites
            new_models = [
                'pauliSet_1J2_zJz_d4',
                'pauliSet_1J2_zJz_d4+pauliSet_2J3_zJz_d4',
                'pauliSet_1J2_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_3J4_zJz_d4',
            ]
            
        elif self.spawn_step == 1:
            new_models = [
                'pauliSet_1J2_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_3J4_zJz_d4', # ring
                'pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4', # square
            ]

        return new_models




