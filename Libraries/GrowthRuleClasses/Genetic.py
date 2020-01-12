import numpy as np
import itertools
import sys, os
import random
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelNames
import ModelGeneration
import SystemTopology
import Heuristics
import GeneticAlgorithm

import SuperClassGrowthRule

flatten = lambda l: [item for sublist in l for item in sublist]  # flatten list of lists


class genetic_algorithm(
    SuperClassGrowthRule.growth_rule_super_class    
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

        self.num_sites = 4
        self.base_terms = [
            'x', 'y', #'z'
        ]
        self.mutation_probability = 0.1

        self.genetic_algorithm = GeneticAlgorithm.GeneticAlgorithmQMLA(
            num_sites = self.num_sites, 
            base_terms = self.base_terms, 
            mutation_probability = self.mutation_probability
        )

        self.true_operator = 'pauliSet_xJx_1J2_d4+pauliSet_yJy_1J2_d4'
        self.max_num_probe_qubits = self.num_sites
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models = 3
        )
        self.max_spawn_depth = 3


    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):
        model_points = kwargs['branch_model_points']
        print("Model points:", model_points)
        print("kwargs: ", kwargs)
        model_fitnesses = {}
        for m in list(model_points.keys()):
            mod = kwargs['model_names_ids'][m]
            model_fitnesses[mod] = model_points[m]

        print("Model fitnesses:", model_fitnesses)
        new_models = self.genetic_algorithm.genetic_algorithm_step(
            model_fitnesses = model_fitnesses
        )

        return new_models





