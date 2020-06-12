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

import qmla.growth_rules.genetic_algorithms.genetic_growth_rule
from qmla.growth_rules.genetic_algorithms.genetic_growth_rule import Genetic
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.expectation_values
import qmla.database_framework


class NVCentreGenticAlgorithm(
    Genetic
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

        self.log_print(["Running GA for NV centre."])

        self.qinfer_model_class =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = False
        self.max_time_to_consider = 4.24
        self.true_model = 'pauliSet_1_x_d2+pauliSet_1_y_d2+pauliSet_1_z_d2+pauliSet_1J2_zJz_d2'
        available_terms = [
            'xTi', 
            'yTi', 
            'zTi', 
            'xTx', 
            'yTy', 
            'zTz', 
            'xTy', 
            'xTz', 
            'yTz'
        ]
        self.true_model = 'xTi+yTi+zTi+zTz'
        self.num_sites = qmla.database_framework.get_num_qubits(self.true_model)

        self.genetic_algorithm = qmla.growth_rules.genetic_algorithms.genetic_algorithm.GeneticAlgorithmQMLA(
            num_sites=self.num_sites,
            true_model = self.true_model,
            genes = available_terms, 
            mutation_probability=0.25,
            num_protected_elite_models = 2, 
            unchanged_elite_num_generations_cutoff = 5, 
            log_file=self.log_file
        )
        self.initial_models = self.genetic_algorithm.random_initial_models(16)

        self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        self.branch_comparison_strategy = 'optimal_graph'
        self.tree_completed_initially = False
        self.fitness_method =  'f_scores' # 'elo_ratings' 
        self.max_spawn_depth = 16
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 1
        self.max_time_to_consider = 20 
        self.min_param = 0.4
        self.max_param = 0.6

    def latex_name(
        self,
        name
    ):
        if name == 'x' or name == 'y' or name == 'z':
            return '$' + name + '$'

        num_qubits = qmla.database_framework.get_num_qubits(name)
        # terms = name.split('PP')
        terms = name.split('+')
        rotations = ['xTi', 'yTi', 'zTi']
        hartree_fock = ['xTx', 'yTy', 'zTz']
        transverse = ['xTy', 'xTz', 'yTz', 'yTx', 'zTx', 'zTy']

        present_r = []
        present_hf = []
        present_t = []

        for t in terms:
            if t in rotations:
                present_r.append(t[0])
            elif t in hartree_fock:
                present_hf.append(t[0])
            elif t in transverse:
                string = t[0] + t[-1]
                present_t.append(string)
            # else:
            #     print("Term",t,"doesn't belong to rotations, Hartree-Fock or transverse.")
            #     print("Given name:", name)
        present_r.sort()
        present_hf.sort()
        present_t.sort()

        r_terms = ','.join(present_r)
        hf_terms = ','.join(present_hf)
        t_terms = ','.join(present_t)

        latex_term = ''
        if len(present_r) > 0:
            latex_term += r'\hat{S}_{' + r_terms + '}'
        if len(present_hf) > 0:
            latex_term += r'\hat{A}_{' + hf_terms + '}'
        if len(present_t) > 0:
            latex_term += r'\hat{T}_{' + t_terms + '}'

        final_term = '$' + latex_term + '$'
        if final_term != '$$':
            return final_term

        else:
            plus_string = ''
            for i in range(num_qubits):
                plus_string += 'P'
            individual_terms = name.split(plus_string)
            individual_terms = sorted(individual_terms)

            latex_term = '+'.join(individual_terms)
            final_term = '$' + latex_term + '$'
            return final_term
