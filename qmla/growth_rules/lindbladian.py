import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import growth_rule_super
from qmla import experiment_design_heuristics
from qmla import topology
from qmla import model_generation
from qmla import model_naming
from qmla import probe_set_generation
from qmla import database_framework

class basic_lindbladian(
    growth_rule_super.GrowthRuleSuper
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
