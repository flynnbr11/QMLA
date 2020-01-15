import SuperClassGrowthRule
import Heuristics
import SystemTopology
import ModelGeneration
import ModelNames
import ProbeGeneration
import DataBase
import numpy as np
import itertools
import sys
import os
sys.path.append(os.path.abspath('..'))


class basic_lindbladian(
    SuperClassGrowthRule.growth_rule_super_class
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
