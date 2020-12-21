from __future__ import absolute_import
import sys
import os
import random

from qmla.exploration_strategies.exploration_strategy import ExplorationStrategy
from qmla.exploration_strategies.nv_centre_spin_characterisation.experimental_paper import FullAccessNVCentre
import qmla.shared_functionality.experiment_design_heuristics
from qmla.construct_models import alph, get_num_qubits

__all__ = [
    'VariableTrueModelNVCentre'
]

class VariableTrueModelNVCentre(
    FullAccessNVCentre
):
    r"""
    Exploration strategy for NV system described in experimental paper, 
    assuming full access to the state so the likelihood is based on 
    $\bra{++} e^{ -i\hat{H(\vec{x})} t } \ket{++}$. 

    This is the base class for results presented in the experimental paper, 
    namely Fig 2. 
    The same model generation strategy is used in each case (i), (ii), (iii):
        this ES is for (i) pure simulation. 

    """

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
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
        self.log_print(["starting rotational ES; true model is {}".format(self.true_model)])
