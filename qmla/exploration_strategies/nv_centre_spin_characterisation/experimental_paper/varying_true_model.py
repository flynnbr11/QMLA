from __future__ import absolute_import
import sys
import os
import random

import qmla.construct_models
from qmla.exploration_strategies.nv_centre_spin_characterisation.experimental_paper import FullAccessNVCentre, TieredGreedySearchNVCentre

__all__ = [
    'VariableTrueModelNVCentre'
]

class VariableTrueModelNVCentre(
    TieredGreedySearchNVCentre 
    # FullAccessNVCentre
):

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        r"""
        Cycle through target model.

        """
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        true_models = [
            'xTi+yTi+zTz',
            'xTi+zTi+xTx+zTz',
            'xTi+yTi+zTi+xTx',
            'xTi+yTi+zTi+zTz',
            'xTi+yTi+zTi+yTy',
            'xTi+yTi+zTi+xTx+zTz',
            'xTi+yTi+zTi+yTy+zTz',
            'xTi+yTi+zTi+xTx+yTy+zTz',
            'xTi+yTi+zTi+xTx+yTy+zTz+xTy',
            'xTi+yTi+zTi+xTx+yTy+zTz+xTz',
        ]

        self.true_model = true_models[self.qmla_id % len(true_models)]
        self.true_model = qmla.construct_models.alph(self.true_model)
        self._shared_true_parameters = False
        self.timing_insurance_factor = 0.5
        self.log_print(["starting rotational ES; true model is {}".format(self.true_model)])
