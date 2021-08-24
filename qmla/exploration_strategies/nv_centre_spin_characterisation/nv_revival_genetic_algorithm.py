import numpy as np
import itertools
import sys
import os
import random
import copy
import scipy
import time
import pandas as pd
import pickle

import qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy
from qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy import (
    Genetic,
)
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
import qmla.shared_functionality.expectation_value_functions
import qmla.model_building_utilities


class NVCentreRevivalSimulation(Genetic):
    def __init__(self, exploration_rules, true_model=None, **kwargs):
        # Fundamental set up
        self.target_num_qubits = 3
        self._set_true_params()
        available_terms = self._get_available_terms_secular_approximation()

        super().__init__(
            exploration_rules=exploration_rules,
            true_model=self.true_model,
            genes=available_terms,
            **kwargs
        )
        self._set_true_params()
        self._setup_prior()
        # self._setup_test_learn_ghz_params()

        # Modular functions
        self.expectation_value_subroutine = (
            qmla.shared_functionality.n_qubit_hahn_evolution
        )
        self.system_probes_generation_subroutine = (
            qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        )
        self.num_sites = qmla.model_building_utilities.get_num_qubits(self.true_model)
        self.model_heuristic_subroutine = (
            qmla.shared_functionality.experiment_design_heuristics.SampleOrderMagnitude
        )

        # Genetic algorithm options
        self.tree_completed_initially = False
        self.branch_comparison_strategy = "optimal_graph"

        num_models_per_generation = 14
        self.max_spawn_depth = 10

        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models_per_generation
        )
        self.initial_models = [
            qmla.model_building_utilities.alph(m) for m in self.initial_models
        ]

        # Logistics
        self.fraction_particles_for_bf = 0.25
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0.5

        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites: (len(self.initial_models) * self.max_spawn_depth) / 8,
            "other": 0,
        }
        # self.num_processes_to_parallelise_over = 16
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.25

    def _set_true_params(self):
        n_qubits = self.target_num_qubits
        self.true_model_terms_params = {
            # spin
            "pauliSet_1_z_d{}".format(n_qubits): 2e9,
            # coupling with 2nd qubit
            "pauliSet_1J2_zJz_d{}".format(n_qubits): 0.2e6,
            # 'pauliSet_1J2_yJy_d{}'.format(n_qubits) : 0.4e6,
            # 'pauliSet_1J2_xJx_d{}'.format(n_qubits) : 0.2e6,
            # carbon nuclei - 2nd qubit
            "pauliSet_2_x_d{}".format(n_qubits): 66e3,
            "pauliSet_2_y_d{}".format(n_qubits): 66e3,
            "pauliSet_2_z_d{}".format(n_qubits): 15e3,
        }
        self.true_model = "+".join((self.true_model_terms_params.keys()))
        self.true_model = qmla.model_building_utilities.alph(self.true_model)
        self.availalbe_pauli_terms = ["x", "y", "z"]

    def _setup_prior(self):
        self.max_time_to_consider = 50e-6
        self.plot_time_increment = 0.5e-6

        self.min_param = 0
        self.max_param = 1e6

        max_num_qubits = self.target_num_qubits
        test_prior_info = {}
        paulis_to_include = self.availalbe_pauli_terms

        for pauli in paulis_to_include:

            for num_qubits in range(1, 1 + max_num_qubits):

                spin_rotation_term = "pauliSet_1_{p}_d{N}".format(p=pauli, N=num_qubits)
                test_prior_info[spin_rotation_term] = (5e9, 2e9)

                for j in range(2, 1 + num_qubits):

                    nuclei_rotation = "pauliSet_{j}_{p}_d{N}".format(
                        j=j, p=pauli, N=num_qubits
                    )
                    test_prior_info[nuclei_rotation] = (5e4, 2e4)

                    coupling_w_spin = "pauliSet_1J{j}_{p}J{p}_d{N}".format(
                        j=j, p=pauli, N=num_qubits
                    )
                    test_prior_info[coupling_w_spin] = (5e5, 2e5)

                    # TODO add transverse terms

        self.gaussian_prior_means_and_widths = test_prior_info

    def _setup_test_learn_ghz_params(self):
        # make uncertainty on all other parmaters thin around true params
        # to force it to learn GHz param only.
        n_qubits = self.target_num_qubits
        self.gaussian_prior_means_and_widths = {
            # spin
            "pauliSet_1_z_d{}".format(n_qubits): (5e9, 2e9),
            # coupling with 2nd qubit
            "pauliSet_1J2_zJz_d{}".format(n_qubits): (0.2e6, 1e2),
            "pauliSet_1J2_yJy_d{}".format(n_qubits): (0.4e6, 1e2),
            "pauliSet_1J2_xJx_d{}".format(n_qubits): (0.2e6, 1e2),
            # carbon nuclei - 2nd qubit
            "pauliSet_2_x_d{}".format(n_qubits): (66e3, 1e1),
            "pauliSet_2_y_d{}".format(n_qubits): (66e3, 1e1),
            "pauliSet_2_z_d{}".format(n_qubits): (15e3, 1e1),
        }

    def _get_available_terms_secular_approximation(self):
        num_qubits = self.target_num_qubits

        available_terms = [
            # electron spin rotation terms
            "pauliSet_1_z_d{}".format(num_qubits),
        ]

        for k in range(2, num_qubits + 1):

            coupling_terms = ["pauliSet_1J{k}_zJz_d{n}".format(k=k, n=num_qubits)]
            rotation_terms = [
                "pauliSet_{k}_{p}_d{n}".format(k=k, p=pauli_term, n=num_qubits)
                for pauli_term in ["x", "y", "z"]
            ]

            available_terms.extend(coupling_terms)
            available_terms.extend(rotation_terms)

        return available_terms

    def _get_available_terms_full_two_qubits(self, num_qubits=2):
        available_terms = [
            # electron spin rotation terms
            "pauliSet_1_x_d2",
            "pauliSet_1_y_d2",
            "pauliSet_1_z_d2",
            # 2nd qubits
            # coupling with spin
            "pauliSet_1J2_xJx_d2",
            "pauliSet_1J2_yJy_d2",
            "pauliSet_1J2_zJz_d2",
            # rotation
            "pauliSet_2_x_d2",
            "pauliSet_2_y_d2",
            "pauliSet_2_z_d2",
        ]
