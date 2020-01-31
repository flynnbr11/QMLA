import numpy as np
import itertools
import sys
import os

from qmla.GrowthRuleClasses import SuperClassGrowthRule
from qmla import experiment_design_heuristics
from qmla import topology
from qmla import model_generation
from qmla import ModelNames
from qmla import ProbeGeneration
from qmla import DataBase

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
