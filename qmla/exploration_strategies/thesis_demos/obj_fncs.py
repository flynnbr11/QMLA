import numpy as np
import itertools
import sys
import os
import random
import copy
import scipy
import time

import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure

import qmla.construct_models

from qmla.exploration_strategies.genetic_algorithms import Genetic


class DemoObjectiveFunctions(
    Genetic
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        true_model = 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5'
        self.true_model = qmla.construct_models.alph(true_model)
        num_sites = qmla.construct_models.get_num_qubits(true_model)
        terms = []
        for i in range(1, 1 + num_sites):
            for j in range(i + 1, 1 + num_sites):
                for t in ['x', 'y', 'z']:
                    new_term = 'pauliSet_{i}J{j}_{o}J{o}_d{N}'.format(
                        i= i, j=j, o=t, N=num_sites, 
                    )
                    terms.append(new_term)
        
        super().__init__(
            exploration_rules = exploration_rules,
            genes = terms, 
            true_model = self.true_model, 
            **kwargs
        )
        # true_model = 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5'

        self.initial_models = [
            'pauliSet_1J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5', # F=0, k=3
            'pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5', # F=0.2, k=4
            'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5', # F=0.4, k=5
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5', # F=0.5, k=7
            # 'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5', # F=0.6, k=5
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5', # F=0.7, k=6
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5', # F=0.8, k=5
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5', # F=0.8, k=7
        ]

        self.initial_models = [
            qmla.construct_models.alph(m) for m in self.initial_models
        ]
        self.initial_num_models = len(self.initial_models)
        self.fitness_method = 'f_score'
        self.branch_comparison_strategy = 'all' # 'minimal'
        
        # speed things up
        self.fraction_experiments_for_bf = 0.1
        self.fraction_own_experiments_for_bf = 0.1
        self.fraction_opponents_experiments_for_bf = 0.1
        self.fraction_particles_for_bf = 0.1 # testing whether reduced num particles for BF can work 

        self.tree_completed_initially = True
        self.max_num_models_by_shape = {
            self.num_sites : self.initial_num_models / 10,
            'other': 0
        }
        self.num_processes_to_parallelise_over = int(self.initial_num_models)
 