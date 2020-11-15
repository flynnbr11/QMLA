import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy

class Demonstration(
    exploration_strategy.ExplorationStrategy
):

    def __init__(
        self,
        exploration_rules,
        true_model=None,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            true_model=true_model,
            **kwargs
        )
        self.true_model = 'pauliSet_1_x_d1+pauliSet_1_y_d1'
        self.max_num_probe_qubits = 3

        self.true_model_terms_params = {
            'pauliSet_1_x_d1' : 0.6, 
            'pauliSet_1_y_d1' : 0.35, 
        }