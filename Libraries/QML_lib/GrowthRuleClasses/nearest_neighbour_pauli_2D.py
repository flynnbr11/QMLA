import numpy as np
import itertools
import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelNames
import Heuristics

import SuperClassGrowthRule
import NV_centre_large_spin_bath
import NV_grow_by_fitness
import Spin_probabilistic


class nearestNeighbourPauli2D(
    Spin_probabilistic.SpinProbabilistic
):

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )


        





