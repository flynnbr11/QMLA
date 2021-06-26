import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.qinfer_model_interface


class AnalyticalLikelihood(exploration_strategy.ExplorationStrategy):
    def __init__(self, exploration_rules, true_model=None, **kwargs):
        self.true_model = "pauliSet_1_z_d1"
        super().__init__(
            exploration_rules=exploration_rules, true_model=self.true_model, **kwargs
        )

        self.qinfer_model_subroutine = (
            qmla.shared_functionality.qinfer_model_interface.QInferInterfaceAnalytical
        )
        self.true_model_terms_params = {"pauliSet_1_z_d1": 7.75}
        self.min_param = 0
        self.max_param = 100
        self.qinfer_resampler_threshold = 0.9
