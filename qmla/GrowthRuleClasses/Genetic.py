import numpy as np
import itertools
import sys
import os
import random

from qmla.GrowthRuleClasses import SuperClassGrowthRule
from qmla import GeneticAlgorithm
from qmla import Heuristics
from qmla import SystemTopology
from qmla import ModelGeneration
from qmla import ModelNames
from qmla import ProbeGeneration
from qmla import DataBase

# flatten list of lists
def flatten(l): return [item for sublist in l for item in sublist]


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
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        # self.true_operator = 'pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_2J3_yJy_d4+pauliSet_1J4_yJy_d4'
        # self.true_operator = 'pauliSet_1J2_xJx_d3+pauliSet_1J2_yJy_d3+pauliSet_2J3_yJy_d3+pauliSet_2J3_zJz_d3'
        # self.ising_full_connectivity = 'pauliSet_1J2_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4'
        self.ising_full_connectivity = 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5'
        self.true_operator = self.ising_full_connectivity
        self.true_operator = DataBase.alph(self.true_operator)
        self.num_sites = DataBase.get_num_qubits(self.true_operator)
        
        self.base_terms = [
            'z',
            # 'x', 'y',  'z'
        ]
        self.mutation_probability = 0.1

        self.genetic_algorithm = GeneticAlgorithm.GeneticAlgorithmQMLA(
            num_sites=self.num_sites,
            base_terms=self.base_terms,
            mutation_probability=self.mutation_probability,
            log_file=self.log_file
        )

        self.true_chromosome = self.genetic_algorithm.map_model_to_chromosome(
            self.true_operator
        )
        self.true_chromosome_string = self.genetic_algorithm.chromosome_string(
            self.true_chromosome
        )

        # self.true_operator = 'pauliSet_xJx_1J2_d3+pauliSet_yJy_1J2_d3'
        self.max_num_probe_qubits = self.num_sites
        self.initial_num_models = 4
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.hamming_distance_by_generation_step = {
            0: [
                hamming_distance(
                    self.true_chromosome_string,
                    self.genetic_algorithm.chromosome_string(
                        self.genetic_algorithm.map_model_to_chromosome(
                            mod
                        )
                    )
                )
                for mod in self.initial_models
            ]
        }
        self.fitness_at_step = {}

        self.max_spawn_depth = 4
        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            4: self.initial_num_models * self.max_spawn_depth,
            'other': 0
        }

        self.max_time_to_consider = 5
        self.min_param = 0.499
        self.max_param = 0.501
        self.num_processes_to_parallelise_over = 10

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        # print("[Genetic] Calling generate_models")
        self.log_print(
            [
                "Spawn step:", kwargs['spawn_step']
            ]
        )
        model_points = kwargs['branch_model_points']
        # print("Model points:", model_points)
        # print("kwargs: ", kwargs)
        self.fitness_at_step[kwargs['spawn_step']] = model_points
        model_fitnesses = {}
        for m in list(model_points.keys()):
            mod = kwargs['model_names_ids'][m]
            model_fitnesses[mod] = model_points[m]

        # print("Model fitnesses:", model_fitnesses)
        new_models = self.genetic_algorithm.genetic_algorithm_step(
            model_fitnesses=model_fitnesses,
            num_pairs_to_sample=self.initial_num_models / 2
        )

        hamming_distances = [
            hamming_distance(
                self.true_chromosome_string,
                self.genetic_algorithm.chromosome_string(
                    self.genetic_algorithm.map_model_to_chromosome(
                        mod
                    )
                )
            )
            for mod in new_models
        ]
        self.hamming_distance_by_generation_step[kwargs['spawn_step']
                                                 ] = hamming_distances

        return new_models

    def latex_name(
        self,
        name,
        **kwargs
    ):
        # print("[latex name fnc] name:", name)
        core_operators = list(sorted(DataBase.core_operator_dict.keys()))
        num_sites = DataBase.get_num_qubits(name)
        p_str = 'P' * num_sites
        p_str = '+'
        separate_terms = name.split(p_str)

        site_connections = {}
        for c in list(itertools.combinations(list(range(num_sites + 1)), 2)):
            site_connections[c] = []

        term_type_markers = ['pauliSet', 'transverse']
        transverse_axis = None
        for term in separate_terms:
            components = term.split('_')
            if 'pauliSet' in components:
                components.remove('pauliSet')

                for l in components:
                    if l[0] == 'd':
                        dim = int(l.replace('d', ''))
                    elif l[0] in core_operators:
                        operators = l.split('J')
                    else:
                        sites = l.split('J')
                sites = tuple([int(a) for a in sites])
                # assumes like-like pauli terms like xx, yy, zz
                op = operators[0]
                site_connections[sites].append(op)
            elif 'transverse' in components:
                components.remove('transverse')
                for l in components:
                    if l[0] == 'd':
                        transverse_dim = int(l.replace('d', ''))
                    elif l in core_operators:
                        transverse_axis = l

        ordered_connections = list(sorted(site_connections.keys()))
        latex_term = ""

        for c in ordered_connections:
            if len(site_connections[c]) > 0:
                this_term = r"\sigma_{"
                this_term += str(c)
                this_term += "}"
                this_term += "^{"
                for t in site_connections[c]:
                    this_term += "{}".format(t)
                this_term += "}"
                latex_term += this_term
        if transverse_axis is not None:
            latex_term += 'T^{}_{}'.format(transverse_axis, transverse_dim)
        latex_term = "${}$".format(latex_term)
        return latex_term


def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))
