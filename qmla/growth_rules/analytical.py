import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.growth_rules import growth_rule
import qmla.shared_functionality.qinfer_model_interface

class AnalyticalLikelihood(
    growth_rule.GrowthRule
):

    def __init__(
        self,
        growth_generation_rule,
        true_model=None,
        **kwargs
    ):
        self.true_model = 'pauliSet_1_z_d1'
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model=self.true_model,
            **kwargs
        )

        self.qinfer_model_class = qmla.shared_functionality.qinfer_model_interface.QInferInterfaceAnalytical
        self.true_model_terms_params = {'pauliSet_1_z_d1' : 7.75}
        self.min_param = 0
        self.max_param = 100
        self.qinfer_resampler_threshold = 0.9
    



