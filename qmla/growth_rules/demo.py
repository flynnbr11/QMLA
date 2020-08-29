import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.growth_rules import growth_rule

class Demonstration(
    growth_rule.GrowthRule
):

    def __init__(
        self,
        growth_generation_rule,
        true_model=None,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model=true_model,
            **kwargs
        )
        self.true_model = 'pauliSet_1_x_d1+pauliSet_1_y_d1'
        self.max_num_probe_qubits = 3