import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy
from qmla.exploration_strategies.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined
from qmla import model_building_utilities
import qmla.shared_functionality.topology_predefined as topologies


class DemoHeuristic(exploration_strategy.ExplorationStrategy):
    def __init__(self, exploration_rules, **kwargs):

        super().__init__(exploration_rules=exploration_rules, **kwargs)

        self.qhl_models = [
            "pauliSet_1_z_d1",
            "pauliSet_1_x_d1+pauliSet_1_y_d1+pauliSet_1_z_d1",
            "pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5",  # ising
            "pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4",  # heisenberg
        ]
        self.true_model_terms_params = {
            "pauliSet_1_z_d1": 0.2,
            "pauliSet_1_x_d1": 0.75,
            "pauliSet_1_y_d1": 0.8,
            "pauliSet_1J3_zJz_d5": 0.322,
            "pauliSet_1J4_zJz_d5": 0.786,
            "pauliSet_1J5_zJz_d5": 0.734,
            "pauliSet_2J4_zJz_d5": 0.388,
            "pauliSet_2J5_zJz_d5": 0.356,
            "pauliSet_3J4_zJz_d5": 0.177,
            "pauliSet_3J5_zJz_d5": 0.105,
            "pauliSet_1J2_zJz_d4": 0.475,
            "pauliSet_1J3_zJz_d4": 0.14,
            "pauliSet_2J3_xJx_d4": 0.219,
            "pauliSet_2J3_zJz_d4": 0.744,
            "pauliSet_2J4_xJx_d4": 0.108,
            "pauliSet_3J4_zJz_d4": 0.145,
        }

        self.max_time_to_consider = 1000
        self._shared_true_parameters = False  # test different models at each instance
        if self._shared_true_parameters:
            true_model_idx = 0
        else:
            true_model_idx = self.qmla_id % len(self.qhl_models) - 1
        self.log_print(
            [
                "self._shared_true_parameters = {} true_model_idx = {}".format(
                    self._shared_true_parameters, true_model_idx
                )
            ]
        )

        self.true_model = self.qhl_models[true_model_idx]


class DemoHeuristicPGH(DemoHeuristic):
    def __init__(self, exploration_rules, **kwargs):

        super().__init__(exploration_rules=exploration_rules, **kwargs)
        self.model_heuristic_subroutine = (
            qmla.shared_functionality.experiment_design_heuristics.MultiParticleGuessHeuristic
        )


class DemoHeuristicNineEighths(DemoHeuristic):
    def __init__(self, exploration_rules, **kwargs):

        super().__init__(exploration_rules=exploration_rules, **kwargs)
        self.model_heuristic_subroutine = (
            qmla.shared_functionality.experiment_design_heuristics.FixedNineEighthsToPowerK
        )


class DemoHeuristicTimeList(DemoHeuristic):
    def __init__(self, exploration_rules, **kwargs):

        super().__init__(exploration_rules=exploration_rules, **kwargs)
        self.model_heuristic_subroutine = (
            qmla.shared_functionality.experiment_design_heuristics.TimeList
        )


class DemoHeuristicRandom(DemoHeuristic):
    def __init__(self, exploration_rules, **kwargs):

        super().__init__(exploration_rules=exploration_rules, **kwargs)
        self.model_heuristic_subroutine = (
            qmla.shared_functionality.experiment_design_heuristics.RandomTimeUpperBounded
        )
