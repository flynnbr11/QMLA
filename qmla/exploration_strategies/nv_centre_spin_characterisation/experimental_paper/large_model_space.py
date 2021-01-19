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
import pickle

import qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy
from qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy import Genetic
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
import qmla.shared_functionality.expectation_value_functions
import qmla.construct_models

class NVCentreGenticAlgorithmPrelearnedParameters(
    Genetic
):
    r"""
    Exploration strategy for studying large model space through a genetic algorithm, 
        considering a nitrogen vacancy centre through the Gali approximation. 
    
    Model generation is through the genetic algorithm exploration strategy.
    This ES sets up the true model as an NV centre spin interacting with a number 
    of nuclei, and makes a wider number of nuceli searchable by the genetic algorithm. 
    Candidate models are assumed to have been learned extremely well by a parameter esimation 
    algorithm, which may be unrealistic in some cases. 
    In the genetic algorithm, to assess candidate models, 
    we use an objective function which computes the average residual between 
    the candidate and the system's dynamics, against a representative dataset. 

    """

    def __init__(
        self,
        exploration_rules,
        true_model=None,
        **kwargs
    ):

        # Fundamental set up
        self.true_n_qubits = 6
        self.available_axes = ['x', 'y', 'z']

        # Set up the target model/parameters
        self._set_true_params() 
        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)

        # Add genetic algorithm parameters to kwargs, which the Genetic ES passes to GA class
        kwargs['selection_truncation_rate'] = 1 / self.true_n_qubits
        kwargs['unchanged_elite_num_generations_cutoff'] = 3*self.true_n_qubits
        kwargs['num_protected_elite_models'] = 2

        # Instantiate exploration strategy super class
        super().__init__(
            exploration_rules=exploration_rules,
            true_model = self.true_model,
            genes = self.available_terms,
            **kwargs
        )

        self._set_true_params() # call again in case something was over written by parent __init__
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)
        self.qhl_models = [self.true_model]
        self.qhl_models = [
            qmla.construct_models.alph(m) for m in self.qhl_models
        ]

        # Modular functions
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.separable_probe_dict # doesn't matter here
        self.evaluation_probe_generation_subroutine = qmla.shared_functionality.probe_set_generation.tomographic_basis
        self.simulator_probes_generation_subroutine = self.system_probes_generation_subroutine
        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution

        # Training parameters
        self.num_eval_probes = 36
        self.num_eval_points = 100
        self.shared_probes = True
        self.num_probes = 5
        self.iqle_mode = False
        self.qinfer_resampler_a = 1
        self.qinfer_resampler_threshold = 0.0

        # Genetic algorithm options
        self.tree_completed_initially =  False
        self.branch_comparison_strategy = 'minimal'  # no need to compare with other models since objective fnc is absolute
        self.fitness_method = 'rs_mean_sq' # squared mean residual 

        test = False # ensure it runs quickly without performing full search
        if test:
            num_models_per_generation = 4
            self.max_spawn_depth = 2
        else:
            num_models_per_generation =  12*self.true_n_qubits # TODO INCREASE NUM EVAL POINTS
            self.max_spawn_depth =  10*self.true_n_qubits

        # Get initial generation's models
        self.initial_models = self.genetic_algorithm.random_initial_models(num_models_per_generation)
        self.initial_models = [ 
            qmla.construct_models.alph(m) for m in self.initial_models
        ]
        if self.tree_completed_initially:
            self.initial_models = self.qhl_models               
        self.initial_num_models = len(self.initial_models)

        # Logistics
        self.force_evaluation = True
        self.fraction_particles_for_bf = 0.1 # BF not meaningful here so minimise cost
        self.fraction_own_experiments_for_bf = 0.1
        self.fraction_opponents_experiments_for_bf = 0
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.num_processes_to_parallelise_over = min(16, len(self.initial_models))
        self.timing_insurance_factor = 0.15

    def _set_true_params(self):
        r"""
        Set up the target model: 
        call a series of subroutines to define the true model,
        as well as setting the parameters to represent the physics appropriately.
        """

        self.true_model_terms_params = self._get_secular_approx_true_params(
            num_qubits = 4, # num qubits in target system
            total_num_qubits = self.true_n_qubits
        )
        self._setup_available_terms_gali_model(
            n_qubits = self.n_qubits, 
            available_axes = self.available_axes
        )
        self._setup_prior_by_parameters()

        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)
        
        self.max_time_to_consider = 100e-6
        self.plot_time_increment = self.max_time_to_consider / 100

    def _get_secular_approx_true_params(
        self, 
        num_qubits = 2,
        total_num_qubits = 5, 
    ):
        r"""
        Using the secular approximation, define true parameters for all present terms. 

        :param int num_qubits: number of qubits in the target model
        :param int total_num_qubits: number of qubits of the search space, 
            i.e. terms will be defined in this dimension, even if the system is not 
            expected to be this large. 

        :returns dict true_params: frequencies of each term to include in the true model
        """

        nuclei_terms = {
            'x' : 66e3, 
            'y' : 66e3, 
            'z' : 15e3
        }
        
        true_params = {}      

        # Spin rotation terms only about Z axis
        spin_term = 'pauliSet_1_z_d{N}'.format(N=total_num_qubits)
        true_params[spin_term] = 2e9
        
        # Coupling and nuclear terms between the spin and all other qubits
        for n in range(2, 1+num_qubits):
            coupling_term = 'pauliSet_1J{n}_zJz_d{N}'.format(
                n=n, N=total_num_qubits)
            true_params[coupling_term] = 0.2e6

            # Nuclei rotations independent of the spin
            for pauli in ['x', 'y', 'z']:

                    nuclei_rotation = 'pauliSet_{n}_{p}_d{N}'.format(
                        n = n, 
                        p = pauli, 
                        N = total_num_qubits
                    )
                    true_params[nuclei_rotation] = nuclei_terms[pauli]

        return true_params

    def _setup_prior_by_parameters(self):
        r"""
        Constructs the prior distribution to assign true parameters in the model. 

        These are set in the gaussian_prior_means_and_widths attribute of this 
        exploration strategy class.

        """

        test_prior_info = {}      

        for pauli in self.available_axes:
            for num_qubits in range(1, 1+self.true_n_qubits):
                
                # Spin rotation
                spin_rotation_term = 'pauliSet_1_{p}_d{N}'.format(
                    p=pauli, N=num_qubits)
                test_prior_info[spin_rotation_term] = (5e9, 2e9)

                # Nuclear terms
                for j in range(2, 1+num_qubits):
                    
                    # Nuclei independent rotation
                    nuclei_rotation = 'pauliSet_{j}_{p}_d{N}'.format(
                        j = j, 
                        p = pauli, 
                        N = num_qubits
                    )
                    test_prior_info[nuclei_rotation] = (5e4, 2e4)

                    # Coupling between spin and nuclei
                    coupling_w_spin = 'pauliSet_1J{j}_{p}J{p}_d{N}'.format(
                        j = j, 
                        p = pauli,
                        N = num_qubits
                    )
                    test_prior_info[coupling_w_spin] = (5e5, 2e5)

        self.gaussian_prior_means_and_widths = test_prior_info

    def _setup_available_terms_gali_model(
        self, 
        n_qubits=2, 
        available_axes=['z']
    ):
        r"""
        Generates the set of terms to include in the genetic algorithm. 

        Terms are stored as an attribute of the class.

        :param int n_qubits: number of qubits to construct terms up to
        :param lsit available_axes: axes about which to generate terms, 
            under the Gali approximation

        """

        available_terms = []
        
        # spin_terms
        for i in range(1, 1+n_qubits):
            for p in available_axes:
                t = 'pauliSet_{i}_{p}_d{N}'.format(i=i, p=p, N=n_qubits)
                available_terms.append(t)
        
        # axial coupling terms between electron and nuclei
        for i in range(2, 1+n_qubits):
            for p in available_axes:
                t = 'pauliSet_1J{i}_{p}J{p}_d{N}'.format(i=i, p=p, N=n_qubits)
                available_terms.append(t)
        
        self.available_terms = available_terms


    def get_prior(self, model_name, **kwargs):
        r"""
        Given a candidate model, constructs a very thin prior. 
        
        This is done to skip the model training stage, and assumes the training has performed 
        extremely well. 
        This method is called by QMLA in constructing candidate models. 

        :param str model_name: string representing the model which is being tested.
        :returns prior: QInfer object, used for sampling parameter values when considering 
            the given model
        """

        prior = qmla.shared_functionality.prior_distributions.prelearned_true_parameters_prior(
            model_name = model_name, 
            true_parameters = self.true_model_terms_params, 
            prior_specific_terms=self.gaussian_prior_means_and_widths,
            default_parameter = 0, 
            default_width = 1e-1, 
            fraction_true_param_found_within=1e-9,
            fraction_true_parameter_width=1e-8, 
            log_file = self.log_file, 
            log_identifier= 'PrelearnedPrior'
        )
        return prior

    def generate_evaluation_data(
        self, 
        num_times = 100, 
        **kwargs
    ):
        r"""
        Generates sequential, equally spaced times for evaluating the candidate models against. 

        :param int num_times: number of datapoints to generate
        :returns dict eval_data: set of experiments for model evaluation
        """
        times = np.arange(
            self.plot_time_increment, 
            self.max_time_to_consider, 
            self.plot_time_increment
        )
        self.log_print([
            "Generating evaluation data with max time={}".format(max(times))
        ])
        eval_data = super().generate_evaluation_data(
            num_probes = self.num_eval_probes, 
            evaluation_times = times, 
            num_eval_points = self.num_eval_points,
            **kwargs
        )

        return eval_data

    def get_evaluation_prior(
        self, 
        model_name, 
        estimated_params, 
        cov_mt, 
        **kwargs
    ):
        r"""
        Generate a QInfer distribution representing the trained model's paramterisation, 
        in order to evaluate that model. 

        :param str model_name: string representing the candidate model
        :param dict estimated_params: average values of the posterior distribution 
            after training, representing the parameter estimates for the model
        :param np.array cov_mt: covariance matrix, i.e. the relationship between 
            parameters after training
        """

        posterior_distribution = self.get_prior(model_name = model_name)
        return posterior_distribution


class NVPrelearnedTest(
    NVCentreGenticAlgorithmPrelearnedParameters
):
    def __init__(
        self,
        exploration_rules,
        true_model=None,
        **kwargs
    ):
        r""" as above but few models/generations to test on"""
        self.true_n_qubits = 6
        self.available_axes = ['x', 'y', 'z']
        self._set_true_params()
        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)

        super().__init__(
            exploration_rules=exploration_rules,
            true_model = self.true_model,
            # genes = self.available_terms,
            **kwargs
        )
        num_models_per_generation = 4
        self.max_spawn_depth = 3

        self.initial_models = self.genetic_algorithm.random_initial_models(num_models_per_generation)
        self.initial_models = [ 
            qmla.construct_models.alph(m) for m in self.initial_models
        ]
