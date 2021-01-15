import numpy as np
import itertools
import sys
import os

from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.probe_set_generation
from qmla import construct_models

class Lindbladian(
    exploration_strategy.ExplorationStrategy
):
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
