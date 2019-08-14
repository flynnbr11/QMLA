import numpy as np
import itertools
import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelNames
import ModelGeneration
import SystemTopology
import Heuristics

import SuperClassGrowthRule
import NVCentreLargeSpinBath
import NVGrowByFitness
import SpinProbabilistic
import ConnectedLattice


class heisenberg_xyz_probabilistic(
    ConnectedLattice.connected_lattice
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
        
        self.lattice_dimension = 2
        self.initial_num_sites = 2
        self.lattice_connectivity_max_distance = 1
        self.lattice_connectivity_linear_only = True
        self.lattice_full_connectivity = False

        self.true_operator = 'pauliSet_zJz_1J2_d4PPPPpauliSet_yJy_1J2_d4PPPPpauliSet_xJx_2J4_d4PPPPpauliSet_yJy_3J4_d4'
        self.true_operator = DataBase.alph(self.true_operator)
        self.qhl_models = [self.true_operator]
        self.base_terms = [
            'x', 
            'y', 
            'z'
        ]
        # fitness calculation parameters. fitness calculation inherited.
        self.num_top_models_to_build_on = 1 # 'all' # at each generation Badassness parameter
        self.model_generation_strictness = 0 #1 #-1 
        self.fitness_win_ratio_exponent = 3

        self.generation_DAG = 1
        self.max_num_sites = 4
        self.tree_completed_initially = False
        self.num_processes_to_parallelise_over = 10
        self.max_num_models_by_shape = {
            'other' : 2
        }
        self.setup_growth_class()
