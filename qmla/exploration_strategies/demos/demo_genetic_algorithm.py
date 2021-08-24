import numpy as np
import itertools
import sys
import os
import random
import copy
import scipy
import time

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure

import qmla.model_building_utilities

from qmla.exploration_strategies.genetic_algorithms import Genetic


class DemoGeneticAlgorithm(Genetic):
    def __init__(self, exploration_rules, **kwargs):
        true_model = "pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4"
        self.true_model = qmla.model_building_utilities.alph(true_model)
        num_sites = qmla.model_building_utilities.get_num_qubits(true_model)
        terms = []
        for i in range(1, 1 + num_sites):
            for j in range(i + 1, 1 + num_sites):
                for t in ["x", "y", "z"]:
                    new_term = "pauliSet_{i}J{j}_{o}J{o}_d{N}".format(
                        i=i,
                        j=j,
                        o=t,
                        N=num_sites,
                    )
                    terms.append(new_term)

        super().__init__(
            exploration_rules=exploration_rules,
            genes=terms,
            true_model=self.true_model,
            **kwargs
        )
        self.max_spawn_depth = 10
        self.max_num_probe_qubits = self.num_sites
        self.initial_num_models = 28
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.fitness_method = "f_score"
        self.branch_comparison_strategy = "minimal"

        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            self.num_sites: (self.initial_num_models * self.max_spawn_depth) / 10,
            "other": 0,
        }
        self.num_processes_to_parallelise_over = self.initial_num_models


class ParamSweepGeneticAlgorithm(Genetic):
    def __init__(
        self,
        exploration_rules,
        num_sites=5,
        num_generations=10,
        num_initial_models=32,
        **kwargs
    ):

        # Generate terms for all possible connections between sites
        # num_sites = qmla.model_building_utilities.get_num_qubits(true_model)
        terms = []
        for i in range(1, 1 + num_sites):
            for j in range(i + 1, 1 + num_sites):
                for t in ["x", "y", "z"]:
                    new_term = "pauliSet_{i}J{j}_{o}J{o}_d{N}".format(
                        i=i,
                        j=j,
                        o=t,
                        N=num_sites,
                    )
                    terms.append(new_term)

        # Set true model (maybe randomly for tests)
        # true_model = 'pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4'
        random_subset_of_terms = np.random.choice(np.array(terms), int(len(terms) / 2))
        true_model = "+".join(random_subset_of_terms)
        self.true_model = qmla.model_building_utilities.alph(true_model)

        super().__init__(
            exploration_rules=exploration_rules,
            genes=terms,
            true_model=self.true_model,
            **kwargs
        )
        self.max_spawn_depth = num_generations
        self.initial_num_models = num_initial_models
        self.max_num_probe_qubits = num_sites
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.fitness_method = "f_score"
        self.branch_comparison_strategy = "minimal"

        # self.tree_completed_initially = True
        self.max_num_models_by_shape = {
            self.num_sites: (self.initial_num_models * self.max_spawn_depth) / 10,
            "other": 0,
        }
        self.num_processes_to_parallelise_over = self.initial_num_models
