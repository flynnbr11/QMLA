from __future__ import absolute_import
from __future__ import print_function  # so print doesn't show brackets

import math
import numpy as np
import os as os
import sys as sys
import pandas as pd
import time as time
from time import sleep
import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import redis

# QMLA functionality
import qmla.analysis
import qmla.database_framework as database_framework
import qmla.database_launch as database_launch
import qmla.get_growth_rule as get_growth_rule
import qmla.redis_settings as rds
from qmla.remote_bayes_factor import remote_bayes_factor_calculation
from qmla.remote_model_learning import remote_learn_model_parameters

pickle.HIGHEST_PROTOCOL = 4  # TODO if >python3, can use higher protocol
plt.switch_backend('agg')

__all__ = [
    'QuantumModelLearningAgent'
]

class QuantumModelLearningAgent():
    """
    - This class manages quantum model development.
    - This is done by controlling a pandas database,
        sending model specifications
        to remote actors (via RQ) to compute QHL,
        and also Bayes factors, generating
        a next set of models iteratively.
    - This is done in a tree like growth mechanism where
        new branches consist of
        models generated considering previously determined "good" models.
    - Model generation rules are given in model_generation.
    - Database control is given in database_framework.
    - Remote functions for computing QHL/Bayes factors are in
    - remote_model_learning and remote_bayes_factor respectively.
    - Redis databases are used to ensure QMD parameters are accessible to
        remote models (since shared memory is not available).
        Relevant QMD parameters and info are pickled to redis.

    """

    def __init__(self,
                 qmla_controls = None, 
                #  generator_list=[],
                 first_layer_models=['x'],
                 probe_dict=None,
                 sim_probe_dict=None,
                 model_priors=None,  # needed for further QHL mode
                 experimental_measurements=None,
                 # TODO get exp measurements from global variables
                 results_directory='',
                 use_exp_custom=True,  # TODO either remove custom exponentiation method or fix
                 plot_times=[0, 1],
                 sigma_threshold=1e-13,
                 **kwargs
                 ):
        self._start_time = time.time()  # to measure run-time

        # Configure this QMLA instance
        if qmla_controls is None: 
            self.qmla_controls = qmla.controls_qmla.parse_cmd_line_args(
                args = {}
            )
        else:
            self.qmla_controls = qmla_controls
        self.growth_class = self.qmla_controls.growth_class

        # Basic settings, path definitions etc
        self._fundamental_settings()

        # Info on true model
        self._true_model_definition()

        # Parameters related to learning/comparing models
        self._set_learning_and_comparison_parameters(
            model_priors=model_priors,
            system_probe_dict=probe_dict,
            simulation_probe_dict=sim_probe_dict,
            experimental_measurements=experimental_measurements,
            plot_times=plot_times
        )

        # Redundant terms -- TODO remove calls to them and then attributes
        self._potentially_redundant_setup(
            first_layer_models=first_layer_models,
            use_exp_custom=use_exp_custom,
            sigma_threshold=sigma_threshold,
        )

        # set up all attributes related to growth rules and tree management
        self._setup_tree_and_growth_rules(
            # generator_list=generator_list,
        )

        # check if QMLA should run in parallel and set up accordingly
        self._setup_parallel_requirements()

        # resources potentiall reallocated based on number of
        # parameters/dimension
        self._compute_base_resources()

        # QMLA core info stored on redis server
        self._compile_and_store_qmla_info_summary()

        # Database used to keep track of models tested
        self._initiate_database()

    ##########
    # Section: Initialisation
    ##########
    def log_print(self, to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'QMLA {}'.format(self.qmla_id)

        )

    def _fundamental_settings(self):
        self.qmla_id = self.qmla_controls.qmd_id
        self.use_experimental_data = self.qmla_controls.use_experimental_data
        self.redis_host_name = self.qmla_controls.host_name
        self.redis_port_number = self.qmla_controls.port_number
        self.log_file = self.qmla_controls.log_file
        self.models_learned = []
        self.qhl_mode = self.qmla_controls.qhl_mode
        self.qhl_mode_multiple_models = self.qmla_controls.qhl_mode_multiple_models
        self.results_directory = self.qmla_controls.results_directory
        if not self.results_directory.endswith('/'):
            self.results_directory += '/'
        
        if self.qmla_controls.latex_mapping_file is None: 
            self.latex_name_map_file_path = os.path.join(
                self.results_directory, 
                'LatexMapping.txt'
            )
        else: 
            self.latex_name_map_file_path = self.qmla_controls.latex_mapping_file
        self.log_print(["Retrieving databases from redis"])
        self.redis_databases = rds.databases_from_qmd_id(
            self.redis_host_name,
            self.redis_port_number,
            self.qmla_id,
            # tree_identifiers=self.tree_identifiers
        )
        self.redis_databases['any_job_failed'].set('Status', 0)

    def _true_model_definition(self):
        self.true_model_name = database_framework.alph(
            self.qmla_controls.true_model_name)
        self.true_model_dimension = database_framework.get_num_qubits(
            self.true_model_name)
        self.true_model_constituent_operators = self.qmla_controls.true_model_terms_matrices
        self.true_model_num_params = self.qmla_controls.true_model_class.num_constituents
        self.true_param_list = self.qmla_controls.true_params_list
        self.true_param_dict = self.qmla_controls.true_params_dict
        self.true_model_branch = -1 # overwrite if considered
        self.true_model_considered = False
        self.true_model_found = False
        self.log_print(
            [
                "True model:", self.true_model_name
            ]
        )

    def _setup_tree_and_growth_rules(
        self,
        # generator_list,
    ):
        # Models and Bayes factors lists
        self.all_bayes_factors = {}
        self.bayes_factor_pair_computed = []
        self.model_name_id_map = {}
        self.highest_model_id = 0  # so first created model gets model_id=0
        self.models_branches = {}
        self.model_initial_branch = {}
        self.model_initial_ids = {}

        # Growth rule setup
        # self.growth_rules_list = generator_list
        self.growth_rules_list = self.qmla_controls.generator_list
        # print(
        #     "[QMLA 189] Generator list {} \n controls all growth rules: {}".format(
        #         generator_list, 
        #         self.qmla_controls
        #     )
        
        # )
        self.growth_rules_initial_models = {}
        self.growth_rule_of_true_model = self.qmla_controls.growth_generation_rule
        zeroth_gen = self.growth_rules_list[0]
        matching_gen_idx = self.growth_rules_list.index(
            self.growth_rule_of_true_model)
        if self.growth_rules_list[0] != self.growth_rule_of_true_model:
            self.growth_rules_list[0] = self.growth_rule_of_true_model
            self.growth_rules_list[matching_gen_idx] = zeroth_gen
        self.unique_growth_rule_instances = self.qmla_controls.unique_growth_rule_instances
        self.log_print(
            ["Growth rule instances: {}".format(self.unique_growth_rule_instances)]
        )
        # self.unique_growth_rule_instances = {
        #     self.growth_rule_of_true_model: self.growth_class
        # }

        self.spawn_depth_by_growth_rule = {}

        # Tree/growth management
        self.spawn_depth = 0
        self.tree_identifiers = [self.growth_rule_of_true_model]
        self.spawn_stage = {}
        self.misc_growth_info = {}
        self.tree_completed = {}

        # branch management
        self.branch_bayes_points = {}
        self.branch_rankings = {}
        self.branch_parents = {}
        self.branch_champions = {}
        self.branch_champs_active_list = []
        self.branch_highest_id = 0
        self.branch_num_models = {}
        self.branch_num_model_pairs = {}
        self.branch_model_learning_complete = {}
        self.branch_comparisons_complete = {}
        self.branch_num_precomputed_models = {}
        self.branch_comparisons_completed = {}
        self.branch_resident_model_names = {}
        self.branch_models_precomputed = {}
        self.branch_resident_model_ids = {}
        self.branch_growth_rules = {}
        self.branch_growth_rule_instances = {}
        self.branch_initial_models = []
        self.branch_champs_by_dimension = {}
        self.ghost_branch_list = []
        self.ghost_branches = {}

        self._setup_all_growth_rules()

    def _setup_all_growth_rules(self):
        initial_id_counter = 0
        models_already_added_to_a_branch = []
        for i in range(len(self.growth_rules_list)):
            # TODO remove use of self.InitialModList -- first layer models got here
            # to match this newly created branch with corresponding dicts
            # filled here
            gen = self.growth_rules_list[i]
            growth_class_gen = get_growth_rule.get_growth_generator_class(
                growth_generation_rule=gen,
                use_experimental_data=self.use_experimental_data,
                log_file=self.log_file
            )
            self.tree_completed[gen] = growth_class_gen.tree_completed_initially
            self.growth_rules_initial_models[gen] = growth_class_gen.initial_models

            self.branch_champs_by_dimension[gen] = {}
            initial_models_this_gen = self.growth_rules_initial_models[gen]
            self.log_print(
                [
                    "initialising generator {} with models: {}".format(
                        gen,
                        initial_models_this_gen)
                ]
            )
            self.branch_initial_models.extend(initial_models_this_gen)
            num_new_models = len(initial_models_this_gen)
            self.branch_resident_model_ids[i] = []
            self.branch_resident_model_names[i] = []
            self.branch_models_precomputed[i] = []
            self.branch_num_precomputed_models[i] = 0

            for mod in initial_models_this_gen:
                mod = database_framework.alph(mod)
                self.branch_resident_model_names[i].append(mod)
                # latest branch to claim it
                self.models_branches[initial_id_counter] = i
                self.branch_highest_id = i
                # self.branch_resident_model_ids[i].append(initial_id_counter)
                if mod in models_already_added_to_a_branch:
                    orig_mod_id = self.model_initial_ids[mod]
                    self.log_print(
                        [
                            mod,
                            "already added as",
                            orig_mod_id
                        ]
                    )
                    self.branch_resident_model_ids[i].append(orig_mod_id)
                    self.branch_models_precomputed[i].append(mod)
                    self.branch_num_precomputed_models[i] += 1
                else:
                    self.log_print(
                        [
                            mod,
                            "not added yet. List:",
                            models_already_added_to_a_branch
                        ]
                    )
                    self.branch_resident_model_ids[i].append(
                        initial_id_counter)
                    self.model_initial_ids[mod] = initial_id_counter
                    self.model_initial_branch[mod] = i
                    models_already_added_to_a_branch.append(mod)
                    initial_id_counter += 1

            # self.highest_model_id += num_new_models
            self.branch_comparisons_completed[i] = False
            self.branch_model_learning_complete[i] = False
            self.branch_comparisons_complete[i] = False

            self.branch_num_models[i] = (
                len(self.growth_rules_initial_models[gen])
            )
            self.branch_num_model_pairs[i] = (
                num_pairs_in_list(len(
                    self.growth_rules_initial_models[gen])
                )
            )
            self.spawn_depth_by_growth_rule[gen] = 0
            self.spawn_stage[gen] = [None]
            self.misc_growth_info[gen] = {}
            self.branch_growth_rules[i] = gen

            # if gen not in list(self.unique_growth_rule_instances.keys()):
            #     self.unique_growth_rule_instances[gen] = growth_class_gen
            self.branch_growth_rule_instances[i] = self.unique_growth_rule_instances[gen]

        # self.branch_highest_id = max(self.model_initial_branch.values())
        self.log_print(
            [
                "After initial branches. model_initial_ids:",
                self.model_initial_ids
            ]
        )
        # to ensure everywhere we use range(qmd.highest_model_id) goes to the
        # right number
        self.highest_model_id = max(self.model_initial_ids.values()) + 1
        self.model_count = len(self.model_initial_ids.keys())
        self.log_print(
            [
                "After setting up initial branches, highest branch id:",
                self.branch_highest_id,
                "highest model id:", self.highest_model_id,
                "initial models:", self.model_name_id_map
            ]
        )

        # i.e. Trees only stem from unique generators
        self.tree_count = len(self.growth_rules_list)
        # print("[QMD] num trees:", self.tree_count)
        self.tree_count_completed = np.sum(
            list(self.tree_completed.values())
        )

    def _set_learning_and_comparison_parameters(
        self,
        model_priors,
        system_probe_dict,
        simulation_probe_dict,
        experimental_measurements,
        plot_times
    ):
        self.model_priors = model_priors
        self.num_particles = self.qmla_controls.num_particles
        self.num_experiments = self.qmla_controls.num_experiments
        self.num_experiments_for_bayes_updates = self.qmla_controls.num_times_bayes
        self.bayes_threshold_lower = self.qmla_controls.bayes_lower
        self.bayes_threshold_upper = self.qmla_controls.bayes_upper
        self.qinfer_resample_threshold = self.qmla_controls.resample_threshold
        self.qinfer_resampler_a = self.qmla_controls.resample_a
        self.qinfer_PGH_heuristic_factor = self.qmla_controls.pgh_factor
        self.qinfer_PGH_heuristic_exponent = self.qmla_controls.pgh_exponent
        self.reallocate_resources = self.qmla_controls.reallocate_resources
        self.model_f_scores = {}
        self.model_precisions = {}
        self.model_sensitivities = {}

        if system_probe_dict is None:
            # ensure there is a probe set
            self.log_print(
                [
                    "Generating probes within QMLA"
                ]
            )
            self.growth_class.generate_probes(
                experimental_data=self.qmla_controls.use_experimental_data,
                noise_level=self.qmla_controls.probe_noise_level,
                minimum_tolerable_noise=0.0,
            )
            self.probes_system = self.growth_class.probes_system
            self.probes_simulator = self.probes_system
            self.probe_number = self.growth_class.num_probes
        else:
            self.probe_number = self.qmla_controls.num_probes
            self.log_print(
                [
                    "Probe dict provided to QMLA."
                ]
            )
            self.probes_system = system_probe_dict
            self.probes_simulator = simulation_probe_dict

        self.experimental_measurements = experimental_measurements
        if self.experimental_measurements is not None:
            self.experimental_measurement_times = (
                sorted(list(self.experimental_measurements.keys()))
            )
        else:
            self.experimental_measurement_times = None

        self.times_to_plot = plot_times
        self.times_to_plot_reduced_set = self.times_to_plot[0::10]
        self.probes_plot_file = self.qmla_controls.probes_plot_file
        if self.probes_plot_file is None: 

            print(
                "Generating plot probes.",
                "max num probe qubits: ", self.growth_class.max_num_probe_qubits,
                "true dimension: ", self.true_model_dimension
            )

            self.probes_for_plots = self.growth_class.plot_probe_generator(
                true_model=self.true_model_name,
                growth_generator=self.growth_class.growth_generation_rule,
                probe_maximum_number_qubits = self.growth_class.max_num_probe_qubits, 
                experimental_data=self.use_experimental_data,
                noise_level=0.0000001,
            )
        else: 
            self.probes_for_plots = pickle.load(
                open(self.probes_plot_file, 'rb')
            )

        if self.use_experimental_data == False:
            # TODO is this doing anything useful?
            # at least put in separate method
            self.experimental_measurements = {}
            self.true_model_hamiltonian = self.qmla_controls.true_hamiltonian
            self.log_print(
                [
                    "Getting expectation values for simulated model",
                    "(len {})".format(len(self.times_to_plot)),
                    "\n Times computed:\n", self.times_to_plot
                ]
            )

            for t in self.times_to_plot:
                self.experimental_measurements[t] = (
                    self.growth_class.expectation_value(
                        ham=self.qmla_controls.true_hamiltonian,
                        t=t,
                        state=self.probes_for_plots[self.true_model_dimension],
                        log_file=self.log_file,
                        log_identifier='[QMLA Init]'
                    )

                )

    def _potentially_redundant_setup(
        self,
        first_layer_models,
        use_exp_custom,
        sigma_threshold,
    ):
        # testing whether these are used anywhere
        # Either remove, or find appropriate place for initialisation and use
        # many are included in qmla_core_info_dict dict sent to workers; check if they are
        # used thereafter
        self.models_first_layer = first_layer_models
        self.use_qle = False  # Set to False for IQLE # TODO remove - redundant
        # self.measurement_class = self.qmla_controls.measurement_type
        self.use_custom_exponentiation = use_exp_custom
        # should only matter when using custom exponentiation package
        self.enable_sparse_exponentiation = True
        self.exponentiation_tolerance = None
        self.sigma_threshold = sigma_threshold
        self.model_fitness_scores = {}
        self.debug_directory = None
        self.use_time_dependent_true_model = False
        self.num_time_dependent_true_params = 0
        self.time_dependent_params = None
        # self.gaussian = self.qmla_controls.gaussian  # TODO remove?
        self.bayes_factors_store_directory = str(
            self.results_directory
            + 'BayesFactorsTimeRecords/'
        )
        if not os.path.exists(self.bayes_factors_store_directory):
            try:
                os.makedirs(self.bayes_factors_store_directory)
            except BaseException:
                # reached at exact same time as another process; don't crash
                pass
        self.bayes_factors_store_times_file = str(
            self.bayes_factors_store_directory
            + 'BayesFactorsPairsTimes_'
            + str(self.qmla_controls.long_id)
            + '.txt'
        )

    def _setup_parallel_requirements(self):
        self.use_rq = self.qmla_controls.use_rq
        self.rq_timeout = self.qmla_controls.rq_timeout
        self.rq_log_file = self.log_file
        self.write_log_file = open(self.log_file, 'a')

        try:
            from rq import Connection, Queue, Worker
            self.redis_conn = redis.Redis(
                host=self.redis_host_name, 
                port=self.redis_port_number
            )
            test_workers = self.use_rq
            self.rq_queue = Queue(
                self.qmla_id, # queue number redis will assign tasks via
                connection=self.redis_conn,
                async=test_workers,
                default_timeout=self.rq_timeout
            )
            parallel_enabled = True
        except BaseException:
            self.log_print("Importing rq failed")
            parallel_enabled = False

        self.run_in_parallel = parallel_enabled

    def _compute_base_resources(self):
        # TODO remove base_num_qubits stuff?
        # TODO put option inside growth rule
        # and functionality as method of ModelForLearning
        if self.reallocate_resources:
            base_num_qubits = 3
            base_num_terms = 3
            for op in self.models_first_layer:
                if database_framework.get_num_qubits(op) < base_num_qubits:
                    base_num_qubits = database_framework.get_num_qubits(op)
                num_terms = len(
                    database_framework.get_constituent_names_from_name(op))
                if (
                    num_terms < base_num_terms
                ):
                    base_num_terms = num_terms

            self.base_resources = {
                'num_qubits': base_num_qubits,
                'num_terms': base_num_terms,
            }
        else:
            self.base_resources = {
                'num_qubits': 1,
                'num_terms': 1,
            }

    def _compile_and_store_qmla_info_summary(
        self
    ):
        number_hamiltonians_to_exponentiate = (
            self.num_particles *
            (self.num_experiments + self.num_experiments_for_bayes_updates)
        )
        self.latex_config = str(
            '$P_{' + str(self.num_particles) +
            '}E_{' + str(self.num_experiments) +
            '}B_{' + str(self.num_experiments_for_bayes_updates) +
            '}RT_{' + str(self.qinfer_resample_threshold) +
            '}RA_{' + str(self.qinfer_resampler_a) +
            '}RP_{' + str(self.qinfer_PGH_heuristic_factor) +
            '}H_{' + str(number_hamiltonians_to_exponentiate) +
            r'}|\psi>_{' + str(self.probe_number) +
            '}PN_{' + str(self.qmla_controls.probe_noise_level) +
            '}$'
        )

        self.qmla_settings = {
            'probes_plot_file': self.probes_plot_file,
            'plot_times': self.times_to_plot,
            'true_name': self.true_model_name,
            'true_oplist': self.true_model_constituent_operators,
            'true_model_terms_params': self.true_param_list,
            'num_particles': self.num_particles,
            'num_experiments': self.num_experiments,
            'results_directory': self.results_directory,
            'plots_directory': self.qmla_controls.plots_directory,
            'long_id': self.qmla_controls.long_id,
            'prior_specific_terms': self.growth_class.gaussian_prior_means_and_widths,
            'model_priors': self.model_priors,
            'use_experimental_data': self.use_experimental_data, 
            'experimental_measurements': self.experimental_measurements, # could be path to unpickle within model?
            'base_resources': self.base_resources, # put inside growth rule
            'reallocate_resources': self.reallocate_resources, # put inside growth rule
            'resampler_thresh': self.qinfer_resample_threshold, # TODO put this inside growth rule, does it need to be top level control? 
            'resampler_a': self.qinfer_resampler_a,  # TODO put this inside growth rule, does it need to be top level control? 
            'pgh_prefactor': self.qinfer_PGH_heuristic_factor,  # TODO put this inside growth rule
            'pgh_exponent': self.qinfer_PGH_heuristic_exponent, # TODO put this inside growth rule
            'increase_pgh_time': self.qmla_controls.increase_pgh_time, # TODO put this inside growth rule
            'store_particles_weights': False,  # from growth rule or unneeded
            'qhl_plots': False,  # from growth rule or unneeded
            'sigma_threshold': self.sigma_threshold, # from growth rule or unneeded
            'experimental_measurement_times': self.experimental_measurement_times, 
            'num_probes': self.probe_number, # from growth rule or unneeded
        }

        # Store qmla_settings and probe dictionaries on the redis database, accessible by all workers
        # These are retrieved by workers to inform them 
        # of parameters to use when learning/comparing models.
        compressed_qmla_core_info = pickle.dumps(self.qmla_settings, protocol=4)
        compressed_probe_dict = pickle.dumps(self.probes_system, protocol=4)
        compressed_sim_probe_dict = pickle.dumps(
            self.probes_simulator, protocol=4)
        qmla_core_info_database = self.redis_databases['qmla_core_info_database']
        qmla_core_info_database.set('qmla_settings', compressed_qmla_core_info)
        qmla_core_info_database.set('ProbeDict', compressed_probe_dict)
        qmla_core_info_database.set('SimProbeDict', compressed_sim_probe_dict)
        
        self.qmla_core_info_database = {
            'qmla_settings' : self.qmla_settings, 
            'ProbeDict' : self.probes_system, 
            'SimProbeDict' : self.probes_simulator
        }
        self.log_print(["Saved QMLA instance info to ", qmla_core_info_database])

    def _initiate_database(self):
        self.model_database, self.model_lists = \
            database_launch.launch_db(
                new_model_branches=self.model_initial_branch,
                new_model_ids=self.model_initial_ids,
                plot_probes=self.probes_for_plots, 
                log_file=self.log_file,
                true_model_terms_matrices=self.true_model_constituent_operators,
                true_model_terms_params=self.true_param_list,
                qmla_core_info_database=self.qmla_core_info_database,
                qid=self.qmla_id,
                host_name=self.redis_host_name,
                port_number=self.redis_port_number
            )

        for mod in list(self.model_initial_ids.keys()):
            mod_id = self.model_initial_ids[mod]
            if database_framework.alph(mod) == self.true_model_name:
                self.true_model_id = mod_id
            print("mod id:", mod_id)
            self.model_name_id_map[int(mod_id)] = mod

        self.log_print(
            [
                "After initiating DB, models:", self.model_name_id_map
            ]
        )

    ##########
    # Section: Setup, configuration and branch/database management functions
    ##########

    def add_model_to_database(
        self,
        model,
        branch_id=0,
        force_create_model=False
    ):
        model = database_framework.alph(model)
        add_model_to_database_result = database_launch.add_model(
            model_name=model,
            model_id=self.model_count,
            branch_id=branch_id,
            running_database=self.model_database,
            model_lists=self.model_lists,
            true_model_terms_matrices=self.true_model_constituent_operators,
            true_model_terms_params=self.true_param_list,
            plot_probes=self.probes_for_plots,
            qmla_core_info_database=self.qmla_core_info_database,
            host_name=self.redis_host_name,
            port_number=self.redis_port_number,
            qid=self.qmla_id,
            log_file=self.log_file,
            force_create_model=force_create_model,
        )
        if add_model_to_database_result == True:  # keep track of how many models/branches in play
            if database_framework.alph(
                    model) == database_framework.alph(self.true_model_name):
                self.true_model_id = self.model_count
                self.true_model_branch = branch_id
            self.highest_model_id += 1
            # print("Setting model ", model, "to ID:", self.model_count)
            model_id = self.model_count
            self.model_name_id_map[model_id] = model
            self.model_count += 1
        else:
            try:
                model_id = database_framework.model_id_from_name(
                    db=self.model_database,
                    name=model
                )
            except BaseException:
                self.log_print(
                    [
                        "Couldn't find model id for model:", model,
                        "model_names_ids:",
                        self.model_name_id_map
                    ]
                )
                raise

        add_model_output = {
            'is_new_model': add_model_to_database_result,
            'model_id': model_id,
        }

        return add_model_output

    def delete_unpicklable_attributes(self):
        del self.redis_conn
        del self.rq_queue
        del self.redis_databases
        del self.write_log_file

    def new_branch(
        self,
        growth_rule,
        model_list
    ):
        model_list = list(set(model_list))  # remove possible duplicates
        self.branch_highest_id += 1
        branch_id = int(self.branch_highest_id)
        self.log_print(
            [
                "NEW BRANCH {}. growth rule= {}".format(branch_id, growth_rule)
            ]
        )
        self.branch_comparisons_completed[branch_id] = False
        num_models = len(model_list)
        self.branch_num_models[branch_id] = num_models
        self.branch_num_model_pairs[branch_id] = num_pairs_in_list(
            num_models
        )
        self.branch_model_learning_complete[branch_id] = False
        self.branch_comparisons_complete[branch_id] = False
        self.branch_growth_rules[branch_id] = growth_rule
        self.branch_growth_rule_instances[branch_id] = self.unique_growth_rule_instances[growth_rule]

        self.log_print(
            [
                'Branch {} growth rule {} has {} new models {}'.format(
                    branch_id,
                    growth_rule,
                    num_models,
                    model_list
                )
            ]
        )
        pre_computed_models = []
        num_models_already_computed_this_branch = 0
        model_id_list = []

        for model in model_list:
            # addModel returns whether adding model was successful
            # if false, that's because it's already been computed
            add_model_info = self.add_model_to_database(
                model,
                branch_id=branch_id
            )
            already_computed = not(
                add_model_info['is_new_model']
            )
            model_id = add_model_info['model_id']
            model_id_list.append(model_id)
            if already_computed == False:  # first instance of this model
                self.models_branches[model_id] = branch_id

            self.log_print(
                [
                    'Model ', model,
                    '\n\tcomputed already: ', already_computed,
                    '\n\tID:', model_id
                ]
            )
            num_models_already_computed_this_branch += bool(
                already_computed
            )
            if bool(already_computed) == True:
                pre_computed_models.append(model)

        self.branch_num_precomputed_models[branch_id] = num_models_already_computed_this_branch
        self.branch_resident_model_names[branch_id] = model_list
        self.branch_models_precomputed[branch_id] = pre_computed_models

        self.branch_resident_model_ids[branch_id] = model_id_list
        self.log_print(
            [
                'Num models already computed on branch ',
                branch_id,
                '=', num_models_already_computed_this_branch,
            ]
        )
        return branch_id

    def get_model_storage_instance_by_id(self, model_id):
        return database_framework.reduced_model_instance_from_id(
            self.model_database, model_id)

    def update_database_model_info(self):
        for mod_id in range(self.highest_model_id):
            try:
                # TODO remove this try/except when reduced-champ-model instance
                # is update-able
                mod = self.get_model_storage_instance_by_id(mod_id)
                mod.model_update_learned_values(
                    fitness_parameters=self.model_fitness_scores
                )
            except BaseException:
                pass

    def update_model_record(
        self,
        field,
        name=None,
        model_id=None,
        new_value=None,
        increment=None
    ):
        database_framework.update_field(
            db=self.model_database,
            name=name,
            model_id=model_id,
            field=field,
            new_value=new_value,
            increment=increment
        )

    def get_model_data_by_field(self, name, field):
        return database_framework.pull_field(
            self.model_database, 
            name, 
            field
    )

    def change_model_status(self, model_name, new_status='Saturated'):
        self.model_database.loc[self.model_database['<Name>'] == model_name, 'Status'] = new_status

    ##########
    # Section: Calculation of models parameters and Bayes factors
    ##########

    def learn_models_on_given_branch(
        self,
        branch_id,
        use_rq=True,
        blocking=False
    ):
        model_list = self.branch_resident_model_names[branch_id]
        self.log_print(
            [
                "learn_models_on_given_branch. branch",
                branch_id,
                " has models:",
                model_list
            ]
        )
        active_branches_learning_models = (
            self.redis_databases['active_branches_learning_models']
        )
        num_models_already_set_this_branch = (
            self.branch_num_precomputed_models[branch_id]
        )
        active_branches_learning_models.set(
            int(branch_id),
            num_models_already_set_this_branch
        )
        unlearned_models_this_branch = list(
            set(model_list) -
            set(self.branch_models_precomputed[branch_id])
        )

        self.log_print(
            [
                "branch {} has {} precomputed models and unlearned models {}".format(
                    branch_id,
                    self.branch_models_precomputed[branch_id],
                    unlearned_models_this_branch
                )
            ]
        )
        if len(unlearned_models_this_branch) == 0:
            self.ghost_branch_list.append(branch_id)

        for model_name in unlearned_models_this_branch:
            self.log_print(
                [
                    "Model ", model_name,
                    "being passed to learnModel function"
                ]
            )
            self.learn_model(
                model_name=model_name,
                use_rq=self.use_rq,
                blocking=blocking
            )
            if blocking is True:
                self.log_print(
                    [
                        "Blocking on; model finished:",
                        model_name
                    ]
                )
            self.update_model_record(
                field='Completed',
                name=model_name,
                new_value=True
            )
        self.log_print(
            [
                'Learning models from branch {} finished.'.format(branch_id)
            ]
        )

    def learn_model(
        self,
        model_name,
        use_rq=True,
        blocking=False
    ):
        model_already_exists = database_framework.check_model_exists(
            model_name=model_name,
            model_lists=self.model_lists,
            db=self.model_database
        )
        if model_already_exists:
            model_id = database_framework.model_id_from_name(
                self.model_database,
                name=model_name
            )
            self.models_learned.append(model_id)
            branch_id = self.models_branches[model_id]
            if self.run_in_parallel and use_rq:
                # i.e. use a job queue rather than sequentially doing it.
                from rq import Connection, Queue, Worker
                queue = Queue(
                    self.qmla_id,
                    connection=self.redis_conn,
                    async=self.use_rq,
                    default_timeout=self.rq_timeout
                )
                queued_model = queue.enqueue(
                    remote_learn_model_parameters,
                    model_name,
                    model_id,
                    growth_generator=self.branch_growth_rules[branch_id],
                    branch_id=branch_id,
                    remote=True,
                    host_name=self.redis_host_name,
                    port_number=self.redis_port_number,
                    qid=self.qmla_id,
                    log_file=self.rq_log_file,
                    result_ttl=-1,
                    timeout=self.rq_timeout
                )

                self.log_print(
                    [
                        "Model",
                        model_name,
                        "added to queue."
                    ]
                )
                if blocking == True:  # i.e. wait for result when called.
                    self.log_print(
                        [
                            "Blocking, ie waiting for",
                            model_name,
                            "to finish on redis queue."
                        ]
                    )
                    while not queued_model.is_finished:
                        if queued_model.is_failed:
                            self.log_print(
                                [
                                    "Model", model_name,
                                    "has failed on remote worker."
                                ]
                            )
                            raise NameError("Remote QML failure")
                            break
                        time.sleep(0.1)
                    self.log_print(
                        ['Blocking RQ model learned:', model_name]
                    )

            else:
                self.log_print(
                    [
                        "Locally calling learn model function.",
                        "model:", model_name,
                        " ID:", model_id
                    ]
                )
                # why is this happening here??
                self.qmla_settings['probe_dict'] = self.probes_system
                updated_model_info = remote_learn_model_parameters(
                    model_name,
                    model_id,
                    growth_generator=self.branch_growth_rules[branch_id],
                    branch_id=branch_id,
                    qmla_core_info_dict=self.qmla_settings,
                    remote=True,
                    host_name=self.redis_host_name,
                    port_number=self.redis_port_number,
                    qid=self.qmla_id, 
                    log_file=self.rq_log_file
                )

                del updated_model_info
        else:
            self.log_print(
                [
                    "Model",
                    model_name,
                    "does not yet exist."
                ]
            )

    def get_pairwise_bayes_factor(
        self,
        model_a_id,
        model_b_id,
        return_job=False,
        branch_id=None,
        interbranch=False,
        remote=True,
        # bayes_threshold=None,
        wait_on_result=False
    ):
        # if bayes_threshold is None:
        #     bayes_threshold = self.bayes_threshold_upper

        if branch_id is None:
            interbranch = True
        unique_id = database_framework.unique_model_pair_identifier(
            model_a_id,
            model_b_id
        )
        if (
            unique_id not in self.bayes_factor_pair_computed
        ):
            # ie not yet considered
            self.bayes_factor_pair_computed.append(
                unique_id
            )

        if self.use_rq:
            from rq import Connection, Queue, Worker
            queue = Queue(self.qmla_id, connection=self.redis_conn,
                          async=self.use_rq, default_timeout=self.rq_timeout
                          )
            job = queue.enqueue(
                remote_bayes_factor_calculation,
                # the function is the first argument to RQ workers
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                branch_id=branch_id,
                interbranch=interbranch,
                times_record=self.bayes_factors_store_times_file,
                bf_data_folder=self.bayes_factors_store_directory,
                num_times_to_use=self.num_experiments_for_bayes_updates,
                # trueModel=self.true_model_name,
                bayes_threshold=self.bayes_threshold_lower,
                host_name=self.redis_host_name,
                port_number=self.redis_port_number,
                qid=self.qmla_id,
                log_file=self.rq_log_file,
                result_ttl=-1,
                timeout=self.rq_timeout
            )
            self.log_print(
                [
                    "Bayes factor calculation queued. Model IDs",
                    model_a_id,
                    model_b_id
                ]
            )
            if wait_on_result == True:
                while job.is_finished == False:
                    if job.is_failed == True:
                        raise("Remote BF failure")
                    sleep(0.1)
            elif return_job == True:
                return job
        else:
            remote_bayes_factor_calculation(
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                # trueModel=self.true_model_name,
                bf_data_folder=self.bayes_factors_store_directory,
                times_record=self.bayes_factors_store_times_file,
                num_times_to_use=self.num_experiments_for_bayes_updates,
                branch_id=branch_id,
                interbranch=interbranch,
                bayes_threshold=self.bayes_threshold_lower,
                host_name=self.redis_host_name,
                port_number=self.redis_port_number,
                qid=self.qmla_id,
                log_file=self.rq_log_file
            )
        if wait_on_result == True:
            pair_id = database_framework.unique_model_pair_identifier(
                model_a_id,
                model_b_id
            )
            bf_from_db = self.redis_databases['bayes_factors_db'].get(pair_id)
            bayes_factor = float(bf_from_db)

            return bayes_factor

    def get_bayes_factors_from_list(
        self,
        model_id_list,
        remote=True,
        wait_on_result=False,
        recompute=False,
        # bayes_threshold=None
    ):
        # if bayes_threshold is None:
        #     bayes_threshold = self.bayes_threshold_lower

        remote_jobs = []
        num_models = len(model_id_list)
        for i in range(num_models):
            a = model_id_list[i]
            for j in range(i, num_models):
                b = model_id_list[j]
                if a != b:
                    unique_id = database_framework.unique_model_pair_identifier(
                        a, b)
                    if (
                        unique_id not in self.bayes_factor_pair_computed
                        or recompute == True
                    ):
                        # ie not yet considered
                        remote_jobs.append(
                            self.get_pairwise_bayes_factor(
                                a,
                                b,
                                remote=remote,
                                return_job=wait_on_result,
                                # bayes_threshold=bayes_threshold
                            )
                        )

        if wait_on_result and self.use_rq:
            self.log_print(
                [
                    "Waiting on result of ",
                    "Bayes comparisons from given list:",
                    model_id_list
                ]
            )
            for job in remote_jobs:
                while job.is_finished == False:
                    if job.is_failed == True:
                        raise NameError("Remote QML failure")
                    time.sleep(0.01)
        else:
            self.log_print(
                [
                    "Not waiting on results of BF calculations",
                    "since we're not using RQ workers here."
                ]
            )

    def get_bayes_factors_by_branch_id(
        self,
        branch_id,
        remote=True,
        # bayes_threshold=None,
        recompute=False
    ):
        # if bayes_threshold is None:
        #     bayes_threshold = self.bayes_threshold_upper

        active_branches_bayes = self.redis_databases['active_branches_bayes']
        model_id_list = self.branch_resident_model_ids[branch_id]
        self.log_print(
            [
                'get_bayes_factors_by_branch_id',
                branch_id,
                'model id list:',
                model_id_list
            ]
        )

        active_branches_bayes.set(int(branch_id), 0)  # set up branch 0
        num_models = len(model_id_list)
        for i in range(num_models):
            a = model_id_list[i]
            for j in range(i, num_models):
                b = model_id_list[j]
                if a != b:
                    unique_id = database_framework.unique_model_pair_identifier(
                        a, b)
                    if (
                        unique_id not in self.bayes_factor_pair_computed
                        or
                        recompute == True
                    ):
                        # ie not yet considered or recomputing
                        self.log_print(
                            [
                                "Computing BF for pair",
                                unique_id
                            ]
                        )

                        self.get_pairwise_bayes_factor(
                            a,
                            b,
                            remote=remote,
                            branch_id=branch_id,
                            # bayes_threshold=bayes_threshold
                        )
                    elif unique_id in self.bayes_factor_pair_computed:
                        # if this is already computed,
                        # tell this branch not to wait on it.
                        active_branches_bayes.incr(
                            int(branch_id),
                            1
                        )

    def process_remote_bayes_factor(
        self,
        a=None,
        b=None,
        pair=None,
        # bayes_threshold=None
    ):

        # if bayes_threshold is None:
        #     bayes_threshold = self.bayes_threshold_lower
        # bayes_threshold = self.bayes_threshold_lower
        bayes_factors_db = self.redis_databases['bayes_factors_db']
        if pair is not None:
            model_ids = pair.split(',')
            a = (float(model_ids[0]))
            b = (float(model_ids[1]))
        elif a is not None and b is not None:
            a = float(a)
            b = float(b)
            pair = database_framework.unique_model_pair_identifier(a, b)
        else:
            self.log_print(
                [
                    "Must pass either two model ids, or a \
                pair name string, to process Bayes factors."]
            )
        try:
            bayes_factor = float(
                bayes_factors_db.get(pair)
            )
        except TypeError:
            self.log_print(
                [
                    "On bayes_factors_db for pair id",
                    pair,
                    "value=",
                    bayes_factors_db.get(pair)
                ]
            )

        # bayes_factor refers to calculation BF(pair), where pair
        # is defined (lower, higher) for continuity
        lower_id = min(a, b)
        higher_id = max(a, b)

        mod_low = self.get_model_storage_instance_by_id(lower_id)
        mod_high = self.get_model_storage_instance_by_id(higher_id)
        if higher_id in mod_low.model_bayes_factors:
            mod_low.model_bayes_factors[higher_id].append(bayes_factor)
        else:
            mod_low.model_bayes_factors[higher_id] = [bayes_factor]

        if lower_id in mod_high.model_bayes_factors:
            mod_high.model_bayes_factors[lower_id].append((1.0 / bayes_factor))
        else:
            mod_high.model_bayes_factors[lower_id] = [(1.0 / bayes_factor)]

        if bayes_factor > self.bayes_threshold_lower:
            champ = mod_low.model_id
        elif bayes_factor < (1.0 / self.bayes_threshold_lower):
            champ = mod_high.model_id

        return champ

    def compare_all_models_in_branch(
        self,
        branch_id,
        # bayes_threshold=None
    ):

        active_models_in_branch_old = database_framework.active_model_ids_by_branch_id(
            self.model_database,
            branch_id
        )
        active_models_in_branch = self.branch_resident_model_ids[branch_id]
        self.log_print(
            [
                'compare_all_models_in_branch', branch_id,
                'active_models_in_branch_old:', active_models_in_branch_old,
                'active_models_in_branch_new:', active_models_in_branch,
            ]
        )

        # if bayes_threshold is None:
        #     bayes_threshold = self.bayes_threshold_lower

        models_points = {}
        for model_id in active_models_in_branch:
            models_points[model_id] = 0

        for i in range(len(active_models_in_branch)):
            mod1 = active_models_in_branch[i]
            for j in range(i, len(active_models_in_branch)):
                mod2 = active_models_in_branch[j]
                if mod1 != mod2:
                    res = self.process_remote_bayes_factor(a=mod1, b=mod2)
                    models_points[res] += 1
                    self.log_print(
                        [
                            "[compare_all_models_in_branch {}]".format(
                                branch_id),
                            "Point to", res,
                            "(comparison {}/{})".format(mod1, mod2),
                        ]
                    )
                    # if res == "a":
                    #     models_points[mod1] += 1
                    #     losing_model_id = mod2
                    # elif res == "b":
                    #     models_points[mod2] += 1
                    #     losing_model_id = mod1
                    # todo if more than one model has max points

        max_points = max(models_points.values())
        max_points_branches = [
            key for key, val in models_points.items()
            if val == max_points
        ]

        if len(max_points_branches) > 1:
            # todo: recompare. Fnc: compareListOfModels (rather than branch
            # based)
            self.log_print(
                [
                    "Multiple models have same number of points within \
                    branch.\n",
                    models_points
                ]
            )
            self.get_bayes_factors_from_list(
                model_id_list=max_points_branches,
                remote=True,
                recompute=True,
                # bayes_threshold=self.bayes_threshold_lower,
                wait_on_result=True
            )

            champ_id = self.compare_models_from_list(
                max_points_branches,
                # bayes_threshold=bayes_threshold,
                models_points_dict=models_points
            )
        else:
            champ_id = max(models_points, key=models_points.get)
        champ_id = int(champ_id)
        # champ_name = database_framework.model_name_from_id(
        #     self.model_database,
        #     champ_id
        # )
        champ_name = self.model_name_id_map[champ_id]

        champ_num_qubits = database_framework.get_num_qubits(champ_name)
        self.branch_champions[int(branch_id)] = champ_id
        if champ_id not in self.branch_champs_active_list:
            self.branch_champs_active_list.append(champ_id)
        growth_rule = self.branch_growth_rules[int(branch_id)]
        try:
            self.branch_champs_by_dimension[growth_rule][champ_num_qubits].append(
                champ_name)
        except BaseException:
            self.branch_champs_by_dimension[growth_rule][champ_num_qubits] = [
                champ_name]

        for model_id in active_models_in_branch:
            self.update_model_record(
                model_id=model_id,
                field='Status',
                new_value='Deactivated'
            )

        self.update_model_record(
            # name=database_framework.model_name_from_id(self.model_database, champ_id),
            name=self.model_name_id_map[champ_id],
            field='Status',
            new_value='Active'
        )
        ranked_model_list = sorted(
            models_points,
            key=models_points.get,
            reverse=True
        )

        if self.branch_comparisons_completed[int(float(branch_id))] == False:
            # only update self.branch_rankings the first time branch is
            # considered
            self.branch_rankings[int(float(branch_id))] = ranked_model_list
            self.branch_comparisons_completed[int(float(branch_id))] = True

        self.log_print(
            [
                "Model points for branch",
                branch_id,
                models_points
            ]
        )
        self.log_print(
            [
                "Champion of branch ",
                branch_id,
                " is ",
                champ_name,
                "({})".format(champ_id)
            ]
        )
        self.branch_bayes_points[branch_id] = models_points
        # self.branch_growth_rules[branch_id]

        if branch_id in self.ghost_branch_list:
            models_to_deactivate = list(
                set(active_models_in_branch)
                - set([champ_id])
            )
            # Ghost branches are to compare
            # already computed models from
            # different branches.
            # So deactivate losers since they shouldn't
            # progress if they lose in a ghost branch.
            for losing_model_id in models_to_deactivate:
                try:
                    self.update_model_record(
                        model_id=losing_model_id,
                        field='Status',
                        new_value='Deactivated'
                    )
                except BaseException:
                    self.log_print(
                        [
                            "not deactivating",
                            losing_model_id,
                            "ActiveBranchChampList:",
                            self.branch_champs_active_list
                        ]
                    )
                try:
                    self.branch_champs_active_list.remove(
                        losing_model_id
                    )
                    self.log_print(
                        [
                            "Ghost Branch",
                            branch_id,
                            "deactivating model",
                            losing_model_id
                        ]
                    )
                except BaseException:
                    pass
        return models_points, champ_id

    def compare_models_from_list(
        self,
        model_list,
        # bayes_threshold=None,
        models_points_dict=None,
        num_times_to_use='all'
    ):
        # if bayes_threshold is None:
        #     bayes_threshold = self.bayes_threshold_lower

        models_points = {}
        for mod in model_list:
            models_points[mod] = 0

        for i in range(len(model_list)):
            mod1 = model_list[i]
            for j in range(i, len(model_list)):
                mod2 = model_list[j]
                if mod1 != mod2:

                    res = self.process_remote_bayes_factor(a=mod1, b=mod2)
                    if res == mod1:
                        loser = mod2
                    elif res == mod2:
                        loser = mod1
                    models_points[res] += 1
                    self.log_print(
                        [
                            "[compare_models_from_list]",
                            "Point to", res,
                            "(comparison {}/{})".format(mod1, mod2)
                        ]
                    )

        max_points = max(models_points.values())
        max_points_branches = [key for key, val in models_points.items()
                               if val == max_points]
        if len(max_points_branches) > 1:
            self.log_print(
                [
                    "Multiple models \
                    have same number of points in compare_models_from_list:",
                    max_points_branches
                ]
            )
            self.log_print(["Recompute Bayes bw:"])
            for i in max_points_branches:
                self.log_print(
                    [
                        database_framework.model_name_from_id(self.model_database, i)
                    ]
                )
            self.log_print(["Points:\n", models_points])
            self.get_bayes_factors_from_list(
                model_id_list=max_points_branches,
                remote=True,
                recompute=True,
                # bayes_threshold=self.bayes_threshold_lower,
                wait_on_result=True
            )
            champ_id = self.compare_models_from_list(
                max_points_branches,
                # bayes_threshold=self.bayes_threshold_lower
            )
        else:
            self.log_print(["After comparing list:", models_points])
            champ_id = max(models_points, key=models_points.get)
        # champ_name = database_framework.model_name_from_id(self.model_database, champ_id)
        champ_name = self.model_name_id_map[champ_id]

        return champ_id

    def perform_final_bayes_comparisons(
        self,
    ):

        bayes_factors_db = self.redis_databases['bayes_factors_db']
        branch_champions = self.branch_champs_active_list
        job_list = []
        job_finished_count = 0
        # if a spawned model is this much better than its parent, parent is
        # deactivated
        interbranch_collapse_threshold = 1e5  # to justify deactivating a parent/child
        num_champs = len(branch_champions)

        self.log_print(
            [
                "Active branch champs at start of final Bayes comp:",
                self.branch_champs_active_list
            ]
        )
        children_branches = list(self.branch_parents.keys())
        for child_id in branch_champions:
            # branch this child sits on
            child_branch = self.models_branches[child_id]

            try:
                # TODO make parent relationships more explicit by model rather
                # than alway parent branch champ
                parent_branch = self.branch_parents[child_branch]
                parent_id = self.branch_champions[parent_branch]

                if (
                    child_id in self.branch_champs_active_list
                    and
                    parent_id in self.branch_champs_active_list
                ):

                    job_list.append(
                        self.get_pairwise_bayes_factor(
                            model_a_id=parent_id,
                            model_b_id=child_id,
                            return_job=True,
                            remote=self.use_rq
                        )
                    )

                    self.log_print(
                        [
                            "Comparing child ",
                            child_id,
                            "with parent",
                            parent_id
                        ]
                    )
                else:
                    self.log_print(
                        [
                            "Either parent or child not in active branch champs list",
                            "Child:", child_id,
                            "Parent:", parent_id
                        ]
                    )
            except BaseException:
                self.log_print(
                    [
                        "Model",
                        child_id,
                        "doesn't have a parent to compare with."
                    ]
                )

        self.log_print(
            [
                "Final Bayes Comparisons.",
                "\nEntering while loop in final bayes fnc.",
                "\nactive branch champs: ", branch_champions

            ]
        )

        if self.use_rq:
            self.log_print(
                [
                    "Waiting on parent/child Bayes factors."
                ]
            )
            for k in range(len(job_list)):
                self.log_print(
                    [
                        "Waiting on parent/child Bayes factors."
                    ]
                )
                while job_list[k].is_finished == False:
                    if job_list[k].is_failed == True:
                        raise NameError("Remote QML failure")
                    sleep(0.01)
            self.log_print(
                [
                    "Parent/child Bayes factors jobs all launched."
                ]
            )

        else:
            self.log_print(
                [
                    "Jobs all finished because not on RQ"
                ]
            )
        # now deactivate parent/children based on those bayes factors
        models_to_remove = []
        for child_id in branch_champions:
            # branch this child sits on
            child_branch = self.models_branches[child_id]
            try:
                parent_branch = self.branch_parents[child_branch]
                parent_id = self.branch_champions[parent_branch]

                mod1 = min(parent_id, child_id)
                mod2 = max(parent_id, child_id)

                pair_id = database_framework.unique_model_pair_identifier(
                    mod1,
                    mod2
                )
                bf_from_db = bayes_factors_db.get(pair_id)
                bayes_factor = float(bf_from_db)
                self.log_print(
                    [
                        "parent/child {}/{} has bf {}".format(
                            parent_id,
                            child_id,
                            bayes_factor
                        )
                    ]
                )

                if bayes_factor > interbranch_collapse_threshold:
                    # bayes_factor heavily favours mod1, so deactive mod2
                    self.log_print(
                        [
                            "Parent model,",
                            mod1,
                            "stronger than spawned; deactivating model",
                            mod2
                        ]
                    )
                    self.update_model_record(
                        model_id=mod2,
                        field='Status',
                        new_value='Deactivated'
                    )
                    try:
                        models_to_remove.append(mod2)
                        # self.branch_champs_active_list.remove(mod2)
                    except BaseException:
                        pass
                elif bayes_factor < (1.0 / interbranch_collapse_threshold):
                    self.log_print(
                        [
                            "Spawned model",
                            mod2,
                            "stronger than parent; deactivating model",
                            mod1
                        ]
                    )
                    self.update_model_record(
                        model_id=mod1,
                        field='Status',
                        new_value='Deactivated'
                    )
                    try:
                        models_to_remove.append(mod1)
                        # self.branch_champs_active_list.remove(mod1)
                    except BaseException:
                        pass

                # Add bayes factors to BayesFactor dict for each model
                mod_a = self.get_model_storage_instance_by_id(mod1)
                mod_b = self.get_model_storage_instance_by_id(mod2)
                if mod2 in mod_a.model_bayes_factors:
                    mod_a.model_bayes_factors[mod2].append(bayes_factor)
                else:
                    mod_a.model_bayes_factors[mod2] = [bayes_factor]

                if mod1 in mod_b.model_bayes_factors:
                    mod_b.model_bayes_factors[mod1].append(
                        (1.0 / bayes_factor))
                else:
                    mod_b.model_bayes_factors[mod1] = [(1.0 / bayes_factor)]
            except Exception as exc:
                self.log_print(
                    [
                        "child doesn't have active parent",
                    ]
                )
                self.log_print(
                    [
                        "Error:", exc
                    ]
                )
                # raise
        self.branch_champs_active_list = list(
            set(self.branch_champs_active_list) -
            set(models_to_remove)
        )
        self.log_print(
            [
                "Parent/child comparisons and deactivations complete."
            ]
        )
        self.log_print(
            [
                "Active branch champs after ",
                "parental collapse (final Bayes comp):",
                self.branch_champs_active_list
            ]
        )
        # make ghost branches of all individidual trees
        # individual trees correspond to separate growth rules.
        self.active_growth_rule_branch_champs = {}
        for gen in self.growth_rules_list:
            self.active_growth_rule_branch_champs[gen] = []

        for active_champ in self.branch_champs_active_list:
            branch_id_of_champ = self.models_branches[active_champ]
            gen = self.branch_growth_rules[branch_id_of_champ]
            self.active_growth_rule_branch_champs[gen].append(active_champ)

        self.log_print(
            [
                "ActiveTreeBranchChamps:",
                self.active_growth_rule_branch_champs
            ]
        )
        # self.final_trees = []
        for gen in list(self.active_growth_rule_branch_champs.keys()):
            models_for_tree_ghost_branch = self.active_growth_rule_branch_champs[gen]
            mod_names = [
                self.model_name_id_map[m]
                for m in models_for_tree_ghost_branch
            ]
            new_branch_id = self.new_branch(
                model_list=mod_names,
                growth_rule=gen
            )

            # self.final_trees.append(
            #     new_branch_id
            # )
            self.branch_model_learning_complete[new_branch_id] = True
            self.learn_models_on_given_branch(new_branch_id)
            self.get_bayes_factors_by_branch_id(new_branch_id)
            # self.get_bayes_factors_by_branch_id(new_branch_id)

        active_branches_learning_models = (
            self.redis_databases[
                'active_branches_learning_models'
            ]
        )
        active_branches_bayes = self.redis_databases[
            'active_branches_bayes'
        ]
        still_learning = True

        # print("[QMD]Entering final while loop")
        while still_learning:
            branch_ids_on_db = list(
                active_branches_learning_models.keys()
            )
            for branchID_bytes in branch_ids_on_db:
                branch_id = int(branchID_bytes)
                if (
                    (int(active_branches_learning_models.get(branch_id)) ==
                     self.branch_num_models[branch_id])
                    and
                    (self.branch_model_learning_complete[branch_id] == False)
                ):
                    self.branch_model_learning_complete[branch_id] = True
                    self.get_bayes_factors_by_branch_id(branch_id)

                if branchID_bytes in active_branches_bayes.keys():
                    branch_id = int(branchID_bytes)
                    num_bayes_done_on_branch = (
                        active_branches_bayes.get(branchID_bytes)
                    )

                    if (int(num_bayes_done_on_branch) ==
                                self.branch_num_model_pairs[branch_id]
                            and
                            self.branch_comparisons_complete[branch_id] == False
                            ):
                        self.branch_comparisons_complete[branch_id] = True
                        self.compare_all_models_in_branch(branch_id)

            if (
                np.all(
                    np.array(
                        list(
                            self.branch_model_learning_complete.values()))
                    == True
                )
                and
                np.all(np.array(list(
                    self.branch_comparisons_complete.values())) == True
                )
            ):
                still_learning = False  # i.e. break out of this while loop

        self.log_print(["Final tree comparisons complete."])

        # Finally, compare all remaining active models,
        # which should just mean the tree champions at this point.
        active_models = database_framework.all_active_model_ids(self.model_database)
        # self.surviving_champions = database_framework.all_active_model_ids(
        #     self.model_database
        # )
        self.log_print(
            [
                "After initial interbranch comparisons, \
                remaining active branch champions:",
                active_models
            ]
        )
        num_active_models = len(active_models)

        self.get_bayes_factors_from_list(
            model_id_list=active_models,
            remote=True,
            recompute=True,
            wait_on_result=True,
            # bayes_threshold=bayes_threshold
        )

        branch_champions_points = {}
        for c in active_models:
            branch_champions_points[c] = 0

        for i in range(num_active_models):
            mod1 = active_models[i]
            for j in range(i, num_active_models):
                mod2 = active_models[j]
                if mod1 != mod2:
                    res = self.process_remote_bayes_factor(
                        a=mod1,
                        b=mod2
                    )
                    self.log_print(
                        [
                            "[perform_final_bayes_comparisons]",
                            "Point to", res,
                            "(comparison {}/{})".format(mod1, mod2)
                        ]
                    )

                    branch_champions_points[res] += 1
                    # if res == "a":
                    #     branch_champions_points[mod1] += 1
                    # elif res == "b":
                    #     branch_champions_points[mod2] += 1
        self.ranked_champions = sorted(
            branch_champions_points,
            reverse=True
        )
        self.log_print(
            [
                "After final Bayes comparisons (of branch champions)",
                branch_champions_points
            ]
        )

        max_points = max(branch_champions_points.values())
        max_points_branches = [
            key for key, val in branch_champions_points.items()
            if val == max_points
        ]
        if len(max_points_branches) > 1:
            # todo: recompare. Fnc: compareListOfModels (rather than branch
            # based)
            self.log_print(
                [
                    "No distinct champion, recomputing bayes "
                    "factors between : ",
                    max_points_branches
                ]
            )
            champ_id = self.compare_models_from_list(
                max_points_branches,
                # bayes_threshold=self.bayes_threshold_lower,
                models_points_dict=branch_champions_points
            )
        else:
            champ_id = max(
                branch_champions_points,
                key=branch_champions_points.get
            )
        # champ_name = database_framework.model_name_from_id(self.model_database, champ_id)
        champ_name = self.model_name_id_map[champ_id]

        branch_champ_names = [
            # database_framework.model_name_from_id(self.model_database, mod_id)
            self.model_name_id_map[mod_id]
            for mod_id in active_models
        ]
        self.change_model_status(
            champ_name,
            new_status='Active'
        )
        return champ_name, branch_champ_names

    ##########
    # Section: QMLA algorithm subroutines
    ##########

    def spawn_from_branch(
        self,
        branch_id,
        growth_rule,
        num_models=1
    ):

        self.spawn_depth_by_growth_rule[growth_rule] += 1
        self.spawn_depth += 1
        # self.log_print(["Spawning, spawn depth:", self.spawn_depth])
        self.log_print(
            [
                "Spawning. Growth rule: {}. Depth: {}".format(
                    growth_rule,
                    self.spawn_depth_by_growth_rule[growth_rule]
                )
            ]
        )
        all_models_this_branch = self.branch_rankings[branch_id]
        best_models = self.branch_rankings[branch_id][:num_models]
        best_model_names = [
            # database_framework.model_name_from_id(self.model_database, mod_id) for
            self.model_name_id_map[mod_id]
            for mod_id in best_models
        ]
        # new_models = model_generation.new_model_list(
        current_champs = [
            self.model_name_id_map[i] for i in
            list(self.branch_champions.values())
        ]
        # print("[QMD] fitness parameters:", self.model_fitness_scores)

        new_models = self.branch_growth_rule_instances[branch_id].generate_models(
            # generator = growth_rule,
            model_list=best_model_names,
            spawn_step=self.spawn_depth_by_growth_rule[growth_rule],
            ghost_branches=self.ghost_branches,
            branch_champs_by_qubit_num=self.branch_champs_by_dimension[growth_rule],
            model_dict=self.model_lists,
            log_file=self.log_file,
            current_champs=current_champs,
            spawn_stage=self.spawn_stage[growth_rule],
            branch_model_points=self.branch_bayes_points[branch_id],
            model_names_ids=self.model_name_id_map,
            miscellaneous=self.misc_growth_info[growth_rule]
        )
        new_models = list(set(new_models))
        new_models = [database_framework.alph(mod) for mod in new_models]

        self.log_print(
            [
                "After model generation for growth rule",
                growth_rule,
                "SPAWN STAGE:",
                self.spawn_stage[growth_rule],
                "\nnew models:",
                new_models
            ]
        )

        new_branch_id = self.new_branch(
            model_list=new_models,
            growth_rule=growth_rule
        )

        self.branch_parents[new_branch_id] = branch_id

        self.log_print(
            [
                "Models to add to new branch (",
                new_branch_id,
                "): ",
                new_models
            ]
        )

        try:
            new_model_dimension = database_framework.get_num_qubits(
                new_models[0]
            )
        except BaseException:
            # TODO this is only during development -- only for cases where
            # spawn step determines termination
            new_model_dimension = database_framework.get_num_qubits(
                best_model_names[0]
            )

        self.learn_models_on_given_branch(
            new_branch_id,
            blocking=False,
            use_rq=True
        )
        tree_completed = self.branch_growth_rule_instances[branch_id].check_tree_completed(
            spawn_step=self.spawn_depth_by_growth_rule[growth_rule],
            current_num_qubits=new_model_dimension
        )

        if self.spawn_stage[growth_rule][-1] == 'Complete':
            tree_completed = True
        return tree_completed

    def inspect_remote_job_crashes(self):
        if self.redis_databases['any_job_failed']['Status'] == b'1':
            # TODO better way to detect errors? For some reason the
            # log print isn't being hit, but raising error seems to be.
            self.log_print(
                [
                    "Failure on remote node. Terminating QMD."
                ]
            )
            raise NameError('Remote QML Failure')

    def finalise_qmla(self):
        # Final functions at end of QMD
        # Fill in champions result dict for further analysis.

        champ_model = self.get_model_storage_instance_by_id(
            self.champion_model_id)
        
        for i in range(self.highest_model_id):
            # Dict of all Bayes factors for each model considered.
            self.all_bayes_factors[i] = (
                self.get_model_storage_instance_by_id(i).model_bayes_factors
            )
            self.compute_model_f_score(i)
        # self.f_score = self.model_f_scores[self.champion_model_id]            
        self.get_statistical_metrics()

        self.log_print(["computing expect vals for mod ", champ_model.model_id])
        champ_model.compute_expectation_values(
            times=self.times_to_plot,
            # plot_probe_path = self.probes_plot_file
        )
        self.log_print(["computed expect vals"])

        self.ChampionFinalParams = (
            champ_model.final_learned_params
        )

        champ_op = database_framework.Operator(self.ChampionName)
        num_params_champ_model = champ_op.num_constituents

        correct_model = misfit = underfit = overfit = 0
        self.log_print(
            [
                "Num params - champ:", num_params_champ_model,
                "; \t true:", self.true_model_num_params]
        )

        self.model_id_to_name_map = {}
        for k in self.model_name_id_map:
            v = self.model_name_id_map[k]
            self.model_id_to_name_map[v] = k

        if database_framework.alph(
                self.ChampionName) == database_framework.alph(self.true_model_name):
            correct_model = 1
        elif (
            num_params_champ_model == self.true_model_num_params
            and
            database_framework.alph(
                self.ChampionName) != database_framework.alph(
                self.true_model_name)
        ):
            misfit = 1
        elif num_params_champ_model > self.true_model_num_params:
            overfit = 1
        elif num_params_champ_model < self.true_model_num_params:
            underfit = 1

        num_params_difference = self.true_model_num_params - num_params_champ_model
        num_qubits_champ_model = database_framework.get_num_qubits(
            self.ChampionName)
        self.LearnedParamsChamp = (
            self.get_model_storage_instance_by_id(
                self.champion_model_id).learned_parameters_qhl
        )
        self.champ_final_sigmas = (
            self.get_model_storage_instance_by_id(
                self.champion_model_id).final_sigmas_qhl
        )
        number_hamiltonians_to_exponentiate = (
            self.num_particles *
            (self.num_experiments + self.num_experiments_for_bayes_updates)
        )

        config = str('config' +
                     '_p' + str(self.num_particles) +
                     '_e' + str(self.num_experiments) +
                     '_b' + str(self.num_experiments_for_bayes_updates) +
                     '_ra' + str(self.qinfer_resampler_a) +
                     '_rt' + str(self.qinfer_resample_threshold) +
                     '_rp' + str(self.qinfer_PGH_heuristic_factor)
                     )

        time_now = time.time()
        time_taken = time_now - self._start_time

        n_qubits = database_framework.get_num_qubits(champ_model.model_name)
        if n_qubits > 3:
            # only compute subset of points for plot
            # otherwise takes too long
            self.log_print(
                [
                    "getting new set of times to plot expectation values for"
                ]
            )
            expec_val_plot_times = self.times_to_plot_reduced_set
        else:
            self.log_print(
                [
                    "Using default times to plot expectation values for",
                    "num qubits:", n_qubits
                ]
            )
            expec_val_plot_times = self.times_to_plot

        self.champion_name_latex = champ_model.model_name_latex
        # equivalent to sleepfchampion_results

        self.champion_results = {
            'NameAlphabetical': database_framework.alph(self.ChampionName),
            'NameNonAlph': self.ChampionName,
            'FinalParams': self.ChampionFinalParams,
            'LatexName': champ_model.model_name_latex,
            # 'LatexName' : database_framework.latex_name_ising(self.ChampionName),
            'NumParticles': self.num_particles,
            'NumExperiments': champ_model.num_experiments,
            'NumBayesTimes': self.num_experiments_for_bayes_updates,
            'ResampleThreshold': self.qinfer_resample_threshold,
            'ResamplerA': self.qinfer_resampler_a,
            'PHGPrefactor': self.qinfer_PGH_heuristic_factor,
            'LogFile': self.log_file,
            'ParamConfiguration': config,
            'ConfigLatex': self.latex_config,
            'Time': time_taken,
            'QID': self.qmla_id,
            'CorrectModel': correct_model,
            'Underfit': underfit,
            'Overfit': overfit,
            'Misfit': misfit,
            'NumQubits': num_qubits_champ_model,
            'NumParams': num_params_champ_model,
            'LearnedParameters': self.LearnedParamsChamp,
            'FinalSigmas': self.champ_final_sigmas,
            'QuadraticLosses': champ_model.quadratic_losses_record,
            'ExpectationValues': champ_model.expectation_values,
            'Trackplot_parameter_estimates': champ_model.track_parameter_estimates,
            'TrackVolume': champ_model.volume_by_epoch,
            'TrackTimesLearned': champ_model.times_learned_over,
            'FinalRSquared': champ_model.r_squared(
                plot_probes=self.probes_for_plots,
                times=expec_val_plot_times
            ),
            'Fscore': self.model_f_scores[self.champion_model_id],
            'Precision': self.model_precisions[self.champion_model_id],
            'Sensitivity': self.model_sensitivities[self.champion_model_id],
            'PValue': champ_model.p_value,
            'LearnedHamiltonian': champ_model.learned_hamiltonian,
            'GrowthGenerator': champ_model.growth_rule_of_this_model,
            'Heuristic': champ_model.model_heuristic_class,
            'ChampLatex': champ_model.model_name_latex,
            'TrueModel': database_framework.alph(self.true_model_name),
            'NumParamDifference': num_params_difference,
            'TrueModelConsidered' : self.true_model_considered, 
            'TrueModelFound' : self.true_model_found,
            'TrueModelBranch' : self.true_model_branch,
            'NumModels' : self.highest_model_id,
            'StatisticalMetrics' : self.generational_statistical_metrics,
            'GenerationalFscore'  : self.generational_f_score,
        }

    def check_champion_reducibility(
        self,
    ):
        champ_mod = self.get_model_storage_instance_by_id(
            self.champion_model_id)
        self.log_print(
            [
                "Checking reducibility of champ model:",
                self.ChampionName,
                "\nParams:\n", champ_mod.learned_parameters_qhl,
                "\nSigmas:\n", champ_mod.final_sigmas_qhl
            ]
        )

        params = list(champ_mod.learned_parameters_qhl.keys())
        to_remove = []
        removed_params = {}
        idx = 0
        for p in params:
            # if champ_mod.final_sigmas_qhl[p] > champ_mod.learned_parameters_qhl[p]:
            #     to_remove.append(p)
            #     removed_params[p] = np.round(
            #         champ_mod.learned_parameters_qhl[p],
            #         2
            #     )

            if (
                np.abs(champ_mod.learned_parameters_qhl[p])
                < self.growth_class.learned_param_limit_for_negligibility
            ):
                to_remove.append(p)
                removed_params[p] = np.round(
                    champ_mod.learned_parameters_qhl[p], 2
                )

        if len(to_remove) >= len(params):
            self.log_print(
                [
                    "Attempted champion reduction failed due to",
                    "all params found neglibible.",
                    "Check method of determining negligibility.",
                    "(By default, parameter removed if sigma of that",
                    "parameters final posterior > parameter.",
                    "i.e. 0 within 1 sigma of distriubtion"
                ]
            )
            return
        if len(to_remove) > 0:
            new_model_terms = list(
                set(params) - set(to_remove)
            )
            dim = database_framework.get_num_qubits(new_model_terms[0])
            p_str = 'P' * dim
            new_mod = p_str.join(new_model_terms)
            new_mod = database_framework.alph(new_mod)

            self.log_print(
                [
                    "Some neglibible parameters found:", removed_params,
                    "\nReduced champion model suggested:", new_mod
                ]
            )

            reduced_mod_info = self.add_model_to_database(
                model=new_mod,
                force_create_model=True
            )
            reduced_mod_id = reduced_mod_info['model_id']
            reduced_mod_instance = self.get_model_storage_instance_by_id(
                reduced_mod_id
            )

            reduced_mod_terms = sorted(
                database_framework.get_constituent_names_from_name(
                    new_mod
                )
            )

            # champ_attributes = list(champ_mod.__dict__.keys())
            # Here we need to fill in learned_info dict on redis with required attributes
            # fill in redis_databases['learned_models_info_db'][reduced_mod_id]
            # for att in champ_attributes:
            #     reduced_mod_instance.__setattr__(
            #         att,
            #         champ_mod.__getattribute__(att)
            #     )

            # get champion leared info
            reduced_champion_info = pickle.loads(
                self.redis_databases['learned_models_info_db'].get(
                    str(self.champion_model_id))
            )

            reduced_params = {}
            reduced_sigmas = {}
            for term in reduced_mod_terms:
                reduced_params[term] = champ_mod.learned_parameters_qhl[term]
                reduced_sigmas[term] = champ_mod.final_sigmas_qhl[term]

            learned_params = [reduced_params[t] for t in reduced_mod_terms]
            sigmas = np.array([reduced_sigmas[t] for t in reduced_mod_terms])
            final_params = np.array(list(zip(learned_params, sigmas)))

            new_cov_mat = np.diag(
                sigmas**2
            )
            import qinfer  # TODO remove? added here bc importing at top slowing down import of QMLA in general
            new_prior = qinfer.MultivariateNormalDistribution(
                learned_params,
                new_cov_mat
            )

            # reduce learned info where appropriate
            reduced_champion_info['name'] = new_mod
            reduced_champion_info['model_terms_names'] = reduced_mod_terms
            reduced_champion_info['final_cov_mat'] = new_cov_mat
            reduced_champion_info['final_params'] = final_params
            reduced_champion_info['learned_parameters'] = reduced_params
            reduced_champion_info['model_id'] = reduced_mod_id
            reduced_champion_info['final_prior'] = new_prior
            reduced_champion_info['est_mean'] = np.array(learned_params)
            reduced_champion_info['final_sigmas'] = reduced_sigmas
            reduced_champion_info['initial_params'] = reduced_sigmas

            compressed_reduced_champ_info = pickle.dumps(
                reduced_champion_info,
                protocol=4
            )

            # TODO fill in values for ModelInstanceForStorage
            self.redis_databases['learned_models_info_db'].set(
                str(float(reduced_mod_id)),
                compressed_reduced_champ_info
            )

            self.get_model_storage_instance_by_id(
                reduced_mod_id).model_update_learned_values()

            bayes_factor = self.get_pairwise_bayes_factor(
                model_a_id=int(self.champion_model_id),
                model_b_id=int(reduced_mod_id),
                wait_on_result=True
            )
            self.log_print(
                [
                    "[QMD] BF b/w champ and reduced champ models:",
                    bayes_factor
                ]
            )

            if (
                (
                    bayes_factor
                    < (1.0 / self.growth_class.reduce_champ_bayes_factor_threshold)
                )
                # or True
            ):
                # overwrite champ id etc

                self.log_print(
                    [
                        "Replacing champion model ({}) with reduced champion model ({} - {})".format(
                            self.champion_model_id,
                            reduced_mod_id,
                            new_mod
                        ),
                        "\n i.e. removing negligible parameter terms:\n{}".format(
                            removed_params
                        )

                    ]
                )
                original_champ_id = self.champion_model_id
                self.champion_model_id = reduced_mod_id
                self.ChampionName = new_mod

                self.get_model_storage_instance_by_id(self.champion_model_id).model_bayes_factors = (
                    self.get_model_storage_instance_by_id(
                        original_champ_id).model_bayes_factors
                )

            # TODO check if BF > threshold; if so, reassign self.champion_model_id and
            # self.ChampionName

        else:
            self.log_print(
                [
                    "Parameters non-negligible; not replacing champion model."
                ]
            )

    ##########
    # Section: Run available algorithms
    ##########

    def run_quantum_hamiltonian_learning(
        self,
        force_qhl=False
    ):

        if (
            (
                self.qhl_mode 
                and
                self.true_model_name not in list(self.models_branches.keys())
            )
            or force_qhl
        ):
            self.new_branch(
                growth_rule=self.growth_rule_of_true_model,
                model_list=[self.true_model_name]
            )

        mod_to_learn = self.true_model_name
        self.log_print(
            [
                "QHL test on:", mod_to_learn
            ]
        )

        self.learn_model(
            model_name=mod_to_learn,
            use_rq=self.use_rq,
            blocking=True
        )

        mod_id = database_framework.model_id_from_name(
            db=self.model_database,
            name=mod_to_learn
        )
        self.true_model_id = mod_id
        self.champion_model_id = mod_id
        self.true_model_found = True
        self.true_model_considered = True # these don't really matter for QHL
        self.log_print(
            [
                "Learned:",
                mod_to_learn,
                ". ID=",
                mod_id
            ]
        )
        mod = self.get_model_storage_instance_by_id(mod_id)
        self.log_print(["Mod (reduced) name:", mod.model_name])
        mod.model_update_learned_values()

        n_qubits = database_framework.get_num_qubits(mod.model_name)
        if n_qubits > 3:
            # only compute subset of points for plot
            # otherwise takes too long
            self.log_print(
                [
                    "getting new set of times to plot expectation values for"
                ]
            )
            expec_val_plot_times = self.times_to_plot_reduced_set
        else:
            self.log_print(
                [
                    "Using default times to plot expectation values for",
                    "num qubits:", n_qubits
                ]
            )
            expec_val_plot_times = self.times_to_plot

        mod.compute_expectation_values(
            times=expec_val_plot_times,
        )
        self.log_print(
            [
                "Finished computing expectation values for", mod.model_name,
            ]
        )

        self.compute_model_f_score(
            model_id=mod_id
        )
        # self.f_score = self.model_f_scores[self.champion_model_id]
        self.get_statistical_metrics()

        time_now = time.time()
        time_taken = time_now - self._start_time

        self.champion_results = {
            'NumParticles': self.num_particles,
            'NumExperiments': mod.num_experiments,
            'NumBayesTimes': self.num_experiments_for_bayes_updates,
            'ResampleThreshold': self.qinfer_resample_threshold,
            'ResamplerA': self.qinfer_resampler_a,
            'PHGPrefactor': self.qinfer_PGH_heuristic_factor,
            'ConfigLatex': self.latex_config,
            'Time': time_taken,
            'QID': self.qmla_id,
            'RSquaredTrueModel': mod.r_squared(
                times=expec_val_plot_times,
                plot_probes=self.probes_for_plots
            ),
            'QuadraticLosses': mod.quadratic_losses_record,
            'NameAlphabetical': database_framework.alph(mod.model_name),
            'LearnedParameters': mod.learned_parameters_qhl,
            'FinalSigmas': mod.final_sigmas_qhl,
            'Trackplot_parameter_estimates': mod.track_parameter_estimates,
            'TrackVolume': mod.volume_by_epoch,
            'TrackTimesLearned': mod.times_learned_over,
            'ExpectationValues': mod.expectation_values,
            'FinalRSquared': mod.final_r_squared,
            'Fscore': self.model_f_scores[mod_id],
            'Precision': self.model_precisions[mod_id],
            'Sensitivity': self.model_sensitivities[mod_id],
            'p-value': mod.p_value,
            'LearnedHamiltonian': mod.learned_hamiltonian,
            'GrowthGenerator': mod.growth_rule_of_this_model,
            'Heuristic': mod.model_heuristic_class,
            'ChampLatex': mod.model_name_latex,
            'TrueModelConsidered' : self.true_model_considered, 
            'TrueModelFound' : self.true_model_found,
            'TrueModelBranch' : self.true_model_branch,
            'NumModels' : self.highest_model_id,
            'StatisticalMetrics' : self.generational_statistical_metrics,
            'GenerationalFscore'  : self.generational_f_score,
        }

        self.log_print(
            [
                "Stored results dict. Finished testQHL function"
            ]
        )

    def run_quantum_hamiltonian_learning_multiple_models(
            self, model_names=None):
        if model_names is None:
            model_names = self.growth_rule.qhl_models

        current_models = list(
            self.models_branches.keys()
        )
        models_to_add = []
        for mod in model_names:
            if mod not in current_models:
                models_to_add.append(mod)
        # create branch with models not automatically included within the
        # GR.qhl_models
        if len(models_to_add) > 0:
            self.new_branch(
                growth_rule=self.growth_rule_of_true_model,
                model_list=models_to_add
            )
        self.qhl_mode_multiple_models = True
        self.champion_model_id = -1,  # TODO just so not to crash during dynamics plot
        self.qhl_mode_multiple_models_model_ids = [
            database_framework.model_id_from_name(
                db=self.model_database,
                name=mod_name
            ) for mod_name in model_names
        ]
        self.log_print(
            [
                'Running QHL for multiple models:', model_names,
            ]
        )

        learned_models_ids = self.redis_databases['learned_models_ids']
        for mod_name in model_names:
            print("Trying to get mod id for", mod_name)
            mod_id = database_framework.model_id_from_name(
                db=self.model_database,
                name=mod_name
            )
            learned_models_ids.set(
                str(mod_id), 0
            )
            self.learn_model(
                model_name=mod_name,
                use_rq=self.use_rq,
                blocking=False
            )

        running_models = learned_models_ids.keys()
        self.log_print(
            [
                'Running Models:', running_models,
            ]
        )
        for k in running_models:
            while int(learned_models_ids.get(k)) != 1:
                sleep(0.01)
                self.inspect_remote_job_crashes()

        self.log_print(
            [
                'Finished waiting on queue, for all:', running_models,
            ]
        )
        time_now = time.time()
        time_taken = time_now - self._start_time

        for mod_name in model_names:
            mod_id = database_framework.model_id_from_name(
                db=self.model_database, name=mod_name
            )
            mod = self.get_model_storage_instance_by_id(mod_id)
            mod.model_update_learned_values(
                fitness_parameters=self.model_fitness_scores
            )
            self.compute_model_f_score(
                model_id=mod_id
            )
            n_qubits = database_framework.get_num_qubits(mod.model_name)
            if n_qubits > 5:
                # only compute subset of points for plot
                # otherwise takes too long
                expec_val_plot_times = self.times_to_plot_reduced_set
            else:
                expec_val_plot_times = self.times_to_plot

            mod.compute_expectation_values(
                times=expec_val_plot_times,
            )

        self.get_statistical_metrics()

        for mod_name in model_names:
            mod_id = database_framework.model_id_from_name(
                db=self.model_database, name=mod_name
            )
            mod = self.get_model_storage_instance_by_id(mod_id)
            # for this stage, consider the considered model as champion
            self.champion_model_id = mod_id
            # self.f_score = self.model_f_scores[self.champion_model_id]

            # equivalent to self.champion_results
            mod.results_dict = {
                'NumParticles': mod.num_particles,
                'NumExperiments': mod.num_experiments,
                'NumBayesTimes': self.num_experiments_for_bayes_updates,
                'ResampleThreshold': self.qinfer_resample_threshold,
                'ResamplerA': self.qinfer_resampler_a,
                'PHGPrefactor': self.qinfer_PGH_heuristic_factor,
                'ConfigLatex': self.latex_config,
                'Time': time_taken,
                'QID': self.qmla_id,
                'ChampID': self.champion_model_id,
                'QuadraticLosses': mod.quadratic_losses_record,
                'RSquaredTrueModel': mod.r_squared(
                    plot_probes=self.probes_for_plots,
                    times=expec_val_plot_times
                ),
                'NameAlphabetical': database_framework.alph(mod.model_name),
                'LearnedParameters': mod.learned_parameters_qhl,
                'FinalSigmas': mod.final_sigmas_qhl,
                'Trackplot_parameter_estimates': mod.track_parameter_estimates,
                'TrackVolume': mod.volume_by_epoch,
                'TrackTimesLearned': mod.times_learned_over,
                # 'TrackCovarianceMatrices' : mod.track_covariance_matrices,
                'ExpectationValues': mod.expectation_values,
                # 'RSquaredByEpoch' : mod.r_squared_by_epoch(
                #     times = expec_val_plot_times,
                #     plot_probes = self.probes_for_plots
                # ),
                # 'FinalRSquared' : mod.final_r_squared,
                'FinalRSquared': mod.r_squared(
                    plot_probes=self.probes_for_plots,
                    times=expec_val_plot_times
                ),
                'p-value': mod.p_value,
                'Fscore': self.model_f_scores[mod_id],
                'Precision': self.model_precisions[mod_id],
                'Sensitivity': self.model_sensitivities[mod_id],
                'LearnedHamiltonian': mod.learned_hamiltonian,
                'GrowthGenerator': mod.growth_rule_of_this_model,
                'Heuristic': mod.model_heuristic_class,
                'ChampLatex': mod.model_name_latex,
                'TrueModelConsidered' : self.true_model_considered, 
                'TrueModelFound' : self.true_model_found,
                'TrueModelBranch' : self.true_model_branch,
                'NumModels' : self.highest_model_id,
                'StatisticalMetrics' : self.generational_statistical_metrics,
                'GenerationalFscore'  : self.generational_f_score,
            }
            self.model_id_to_name_map = {}
            for k in self.model_name_id_map:
                v = self.model_name_id_map[k]
                self.model_id_to_name_map[v] = k

    def run_complete_qmla(
        self,
        num_exp=40,
        num_spawns=1,
        max_branches=None,
        # max_num_qubits=None,
        # max_num_models=None,
        spawn=True,
        just_given_models=False
    ):
        active_branches_learning_models = (
            self.redis_databases['active_branches_learning_models']
        )
        active_branches_bayes = self.redis_databases['active_branches_bayes']

        if self.tree_count > 1:
            for i in list(self.branch_resident_model_names.keys()):
                # print("[QMD runMult] launching branch ", i)
                # ie initial branches
                self.learn_models_on_given_branch(
                    i,
                    blocking=False,
                    use_rq=True
                )
                while(
                    int(active_branches_learning_models.get(i))
                        < self.branch_num_models[i]
                ):
                    # don't do comparisons till all models on this branch are
                    # done
                    sleep(0.1)
                    # print("num models learned on br", i,
                    #     ":", int(active_branches_learning_models[i])
                    # )
                self.branch_model_learning_complete[i] = True
                self.get_bayes_factors_by_branch_id(i)
                while (
                    int(active_branches_bayes.get(i))
                        < self.branch_num_model_pairs[i]
                ):  # bayes comparisons not done
                    sleep(0.1)
                self.log_print(
                    [
                        "Models computed and compared for branch", i
                    ]
                )
        else:
            for i in list(self.branch_resident_model_names.keys()):
                # print("[QMD runMult] launching branch ", i)
                # ie initial branches
                self.learn_models_on_given_branch(
                    i,
                    blocking=False,
                    use_rq=True
                )

        # max_spawn_depth_reached = False
        all_comparisons_complete = False

        branch_ids_on_db = list(
            active_branches_learning_models.keys()
        )
        self.log_print(
            [
                "Entering while loop of spawning model layers and comparing."
            ]
        )
        # while max_spawn_depth_reached==False:
        while self.tree_count_completed < self.tree_count:
            branch_ids_on_db = list(
                active_branches_learning_models.keys()
            )

            # print("[QMD] branches:", branch_ids_on_db)
            self.inspect_remote_job_crashes()

            for branchID_bytes in branch_ids_on_db:
                branch_id = int(branchID_bytes)
                if (
                    int(
                        active_branches_learning_models.get(
                            branch_id)
                    ) == self.branch_num_models[branch_id]
                    and
                    self.branch_model_learning_complete[branch_id] == False
                ):
                    self.log_print([
                        "All models on branch",
                        branch_id,
                        "have finished learning."]
                    )
                    self.branch_model_learning_complete[branch_id] = True
                    models_this_branch = self.branch_resident_model_ids[branch_id]
                    for mod_id in models_this_branch:
                        mod = self.get_model_storage_instance_by_id(mod_id)
                        mod.model_update_learned_values()

                    self.get_bayes_factors_by_branch_id(branch_id)

            for branchID_bytes in active_branches_bayes.keys():

                branch_id = int(branchID_bytes)
                bayes_calculated = active_branches_bayes.get(
                    branchID_bytes
                )

                if (int(bayes_calculated) ==
                    self.branch_num_model_pairs[branch_id]
                        and
                    self.branch_comparisons_complete[branch_id] == False
                    ):
                    self.branch_comparisons_complete[branch_id] = True
                    self.compare_all_models_in_branch(branch_id)
                    # print("[QMD] getting growth rule for branch_id", branch_id)
                    # print("[QMD] dict:", self.branch_growth_rules)
                    this_branch_growth_rule = self.branch_growth_rules[branch_id]
                    if self.tree_completed[this_branch_growth_rule] == False:
                        growth_rule_tree_complete = self.spawn_from_branch(
                            # will return True if this brings it to
                            # self.MaxSpawnDepth
                            branch_id=branch_id,
                            growth_rule=this_branch_growth_rule,
                            num_models=1
                        )

                        if (
                            growth_rule_tree_complete == True
                        ):
                            self.tree_completed[this_branch_growth_rule] = True
                            self.tree_count_completed += 1
                            print(
                                "[QMD] Num trees now completed:",
                                self.tree_count_completed,
                                "Tree completed dict:",
                                self.tree_completed
                            )
                            # max_spawn_depth_reached = True
                    else:
                        print(
                            "Finished tree for growth rule:",
                            this_branch_growth_rule
                        )

        self.log_print(
            [
                "All trees have completed.",
                "Num complete:",
                self.tree_count_completed
            ]
        )
        # let any branches which have just started finish before moving to
        # analysis
        still_learning = True

        while still_learning:
            branch_ids_on_db = list(active_branches_learning_models.keys())
            # branch_ids_on_db.remove(b'LOCKED')
            for branchID_bytes in branch_ids_on_db:
                branch_id = int(branchID_bytes)
                if (
                    (int(active_branches_learning_models.get(branch_id)) ==
                     self.branch_num_models[branch_id])
                    and
                    (self.branch_model_learning_complete[branch_id] == False)
                ):
                    self.branch_model_learning_complete[branch_id] = True
                    self.get_bayes_factors_by_branch_id(branch_id)

                if branchID_bytes in active_branches_bayes:
                    num_bayes_done_on_branch = (
                        active_branches_bayes.get(branchID_bytes)
                    )
                    # print(
                    #     "branch", branch_id,
                    #     "num complete:", num_bayes_done_on_branch
                    # )
                    if (int(num_bayes_done_on_branch) ==
                            self.branch_num_model_pairs[branch_id] and
                            self.branch_comparisons_complete[branch_id] == False
                            ):
                        self.branch_comparisons_complete[branch_id] = True
                        self.compare_all_models_in_branch(branch_id)

            if (
                np.all(
                    np.array(
                        list(
                            self.branch_model_learning_complete.values()))
                    == True
                )
                and
                np.all(np.array(list(
                    self.branch_comparisons_complete.values())) == True
                )
            ):
                still_learning = False  # i.e. break out of this while loop

        self.log_print(["Finalising QMLA."])
        final_winner, final_branch_winners = self.perform_final_bayes_comparisons()
        self.ChampionName = final_winner
        self.champion_model_id = self.get_model_data_by_field(
            name=final_winner,
            field='ModelID'
        )
        try:
            self.true_model_considered = True
            if self.champion_model_id == self.true_model_id:
                self.true_model_found = True
            else:
                self.true_model_found = False
        except:
            self.true_model_considered = False 
            self.true_model_found = False
        self.update_database_model_info()

        # Check if final winner has negligible parameters; potentially change
        # champion
        if (
            self.growth_class.check_champion_reducibility == True
            and
            self.growth_class.tree_completed_initially == False
        ):
            self.check_champion_reducibility()

        if self.ChampionName == database_framework.alph(self.true_model_name):
            self.log_print(
                [
                    "True model found: {}".format(
                        database_framework.alph(self.true_model_name)
                    )
                ]
            )
        self.log_print(
            [
                "True model considered: {}. on branch {}.".format(
                    self.true_model_considered,
                    self.true_model_branch
                )
            ]
        )
        self.finalise_qmla()
        self.log_print(
            [
                "Final winner:", self.ChampionName, 
                "has F-score ", self.model_f_scores[self.champion_model_id]
            ]
        )

    ##########
    # Section: Analysis/plotting functions
    ##########

    def compute_model_f_score(
        self,
        model_id,
        beta=1  # beta=1 for F1-score. Beta is relative importance of sensitivity to precision
    ):
        # TODO set precision, f-score etc as model instance attributes and
        # return those in champion_results
        true_set = self.growth_class.true_model_terms
        growth_class = self.get_model_storage_instance_by_id(
            model_id).growth_class
        terms = [
            growth_class.latex_name(
                term
            )
            for term in
            database_framework.get_constituent_names_from_name(
                self.model_name_id_map[model_id]
            )
        ]
        learned_set = set(sorted(terms))


        total_positives = len(true_set)
        true_positives = len(true_set.intersection(learned_set))
        false_positives = len(learned_set - true_set)
        false_negatives = len(true_set - learned_set)
        precision = true_positives / \
            (true_positives + false_positives)
        sensitivity = true_positives / total_positives
        try:
            # self.f_score = (
            f_score = (
                (1 + beta**2) * (
                    (precision * sensitivity)
                    / (beta**2 * precision + sensitivity)
                )
            )
        except BaseException:
            # both precision and sensitivity=0 as true_positives=0
            # self.f_score = 0
            f_score = 0

        f_score = f_score
        self.model_f_scores[model_id] = f_score
        self.model_precisions[model_id] = precision
        self.model_sensitivities[model_id] = sensitivity
        return f_score 

    def get_statistical_metrics(
        self,
        save_to_file=None
    ):
        model_ids = self.models_learned
        self.log_print(
            ["Getting statistical metrics for {}".format(model_ids)]
        )
        model_branches = {
            i : self.get_model_data_by_field(
                name = self.model_name_id_map[i], 
                field = 'branch_id'
            )
            for i in model_ids
        }
        generations = sorted(set(model_branches.values()))
        self.log_print(
            [
                "[get_statistical_metrics",
                "model branches:", model_branches, 
                "model ids: ", model_ids, 
                "generations: ", generations
            ]
        )

        generational_sensitivity = {
            b : []
            for b in generations
        }
        generational_f_score = {
            b : []
            for b in generations
        }
        generational_precision = {
            b : []
            for b in generations
        }
        for m in model_ids: 
            b = model_branches[m]
            generational_sensitivity[b].append(self.model_sensitivities[m])
            generational_precision[b].append(self.model_precisions[m]) 
            generational_f_score[b].append(self.model_f_scores[m])    
        include_plots = [
            {'name' : 'F-score', 'data' : generational_f_score, 'colour' : 'red'}, 
            {'name' : 'Precision', 'data' : generational_precision, 'colour': 'blue'}, 
            {'name' : 'Sensitivity', 'data' : generational_sensitivity, 'colour' : 'green'}, 
        ]
        self.generational_f_score = generational_f_score
        self.generational_statistical_metrics = {
            k['name'] : k['data']
            for k in include_plots
        }
        self.alt_generational_statistical_metrics = {
            b : {
                'Precision' : generational_precision[b],
                'Sensitivity' : generational_sensitivity[b], 
                'F-score' : generational_f_score[b]
            }
            for b in generations 
        }

        fig = plt.figure(
            figsize=(15, 5),
            # constrained_layout=True,
            tight_layout=True
        )
        gs = GridSpec(
            nrows=1,
            ncols=len(include_plots),
            # figure=fig # not available on matplotlib 2.1.1 (on BC)
        )
        plot_col = 0

        for plotting_data in include_plots: 

            ax = fig.add_subplot(gs[0, plot_col])
            data = plotting_data['data']
            ax.plot(
                generations,
                [np.median(data[b]) for b in generations], 
                label = "{} median".format(plotting_data['name']),
                color = plotting_data['colour'],
                marker = 'o'
            )
            ax.fill_between(
                generations,
                [np.min(data[b]) for b in generations], 
                [np.max(data[b]) for b in generations], 
                alpha = 0.2, 
                label = "{} min/max".format(plotting_data['name']),
                color = plotting_data['colour']
            )
            ax.set_ylabel("{}".format(plotting_data['name']))
            ax.set_xlabel("Generation")
            ax.legend()
            ax.set_ylim(0,1)
            plot_col += 1

        self.log_print(["getting statistical metrics complete"])
        if save_to_file is not None: 
            plt.savefig(save_to_file)



    def plot_branch_champs_quadratic_losses(
        self,
        save_to_file=None,
    ):
        qmla.analysis.plot_quadratic_loss(
            qmd=self,
            champs_or_all='champs',
            save_to_file=save_to_file
        )

    def plot_branch_champs_volumes(self, model_id_list=None, branch_champions=False,
                                   branch_id=None, save_to_file=None
                                   ):

        plt.clf()
        plot_descriptor = '\n(' + str(self.num_particles) + 'particles; ' + \
            str(self.num_experiments) + 'experiments).'

        if branch_champions:
            # only plot for branch champions
            model_id_list = list(self.branch_champions.values())
            plot_descriptor += '[Branch champions]'

        elif branch_id is not None:
            model_id_list = database_framework.list_model_id_in_branch(
                self.model_database, branch_id)
            plot_descriptor += '[Branch' + str(branch_id) + ']'

        elif model_id_list is None:
            self.log_print(["Plotting volumes for all models by default."])

            model_id_list = range(self.highest_model_id)
            plot_descriptor += '[All models]'

        plt.title('Volume evolution through QMD ' + plot_descriptor)
        plt.xlabel('Epoch')
        plt.ylabel('Volume')

        for i in model_id_list:
            vols = self.get_model_storage_instance_by_id(i).volume_by_epoch
            plt.semilogy(vols, label=str('ID:' + str(i)))
#            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        ax = plt.subplot(111)

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                        fancybox=True, shadow=True, ncol=4)

        if save_to_file is None:
            plt.show()
        else:
            plt.savefig(
                save_to_file, bbox_extra_artists=(
                    lgd,), bbox_inches='tight')

    def store_bayes_factors_to_csv(self, save_to_file, names_ids='latex'):
        qmla.analysis.model_bayes_factorsCSV(
            self, save_to_file, names_ids=names_ids)

    def store_bayes_factors_to_shared_csv(self, bayes_csv):
        print("[QMD] writing Bayes CSV")
        qmla.analysis.updateAllBayesCSV(self, bayes_csv)

    def plot_parameter_learning_single_model(
        self,
        model_id=0,
        true_model=False,
        save_to_file=None
    ):
        if true_model:
            model_id = database_framework.model_id_from_name(
                db=self.model_database, name=self.true_model_name)

        qmla.analysis.plot_parameter_estimates(qmd=self,
                                               model_id=model_id,
                                               use_experimental_data=self.use_experimental_data,
                                               save_to_file=save_to_file
                                               )

    def plot_branch_champions_dynamics(
        self,
        all_models=False,
        model_ids=None,
        include_bayes_factors_in_dynamics_plots=True,
        include_param_estimates_in_dynamics_plots=False,
        include_times_learned_in_dynamics_plots=True,
        save_to_file=None,
    ):
        if all_models == True:
            model_ids = list(sorted(self.model_name_id_map.keys()))
        elif model_ids is None:
            model_ids = list(
                sorted(self.branch_champions.values())
            )

        qmla.analysis.plot_learned_models_dynamics(
            qmd=self,
            include_bayes_factors=include_bayes_factors_in_dynamics_plots,
            include_times_learned=include_times_learned_in_dynamics_plots,
            include_param_estimates=include_param_estimates_in_dynamics_plots,
            model_ids=model_ids,
            save_to_file=save_to_file,
        )

    def plot_volume_after_qhl(self,
                              model_id=None,
                              true_model=True,
                              show_resamplings=True,
                              save_to_file=None
                              ):
        qmla.analysis.plot_volume_after_qhl(
            qmd=self,
            model_id=model_id,
            true_model=true_model,
            show_resamplings=show_resamplings,
            save_to_file=save_to_file
        )

    def plot_qmla_tree(
        self,
        modlist=None,
        only_adjacent_branches=True,
        save_to_file=None
    ):
        qmla.analysis.plot_qmla_single_instance_tree(
            self,
            modlist=modlist,
            only_adjacent_branches=only_adjacent_branches,
            save_to_file=save_to_file
        )

    def plot_qmla_radar_scores(self, modlist=None, save_to_file=None):
        plot_title = str("Radar Plot QMD " + str(self.qmla_id))
        if modlist is None:
            modlist = list(self.branch_champions.values())
        qmla.analysis.plotRadar(
            self,
            modlist,
            save_to_file=save_to_file,
            plot_title=plot_title
        )

    def plot_r_squared_by_epoch_for_model_list(
        self,
        modlist=None,
        save_to_file=None
    ):
        if modlist is None:
            modlist = []
            try:
                modlist.append(self.champion_model_id)
            except BaseException:
                pass
            try:
                modlist.append(self.true_model_id)
            except BaseException:
                pass

        qmla.analysis.r_squared_from_epoch_list(
            qmd=self,
            model_ids=modlist,
            save_to_file=save_to_file
        )

    def one_qubit_probes_bloch_sphere(self):
        print("In jupyter, include the following to view sphere: %matplotlib inline")
        # import qutip as qt
        bloch = qt.Bloch()
        for i in range(self.probe_number):
            state = self.probes_system[i, 1]
            a = state[0]
            b = state[1]
            A = a * qt.basis(2, 0)
            B = b * qt.basis(2, 1)
            vec = (A + B)
            print(vec)
            bloch.add_states(vec)
        bloch.show()


##########
# Section: Miscellaneous functions called within QMLA
##########


def num_pairs_in_list(num_models):
    if num_models <= 1:
        return 0

    n = num_models
    k = 2  # ie. nCk where k=2 since we want pairs

    try:
        a = math.factorial(n) / math.factorial(k)
        b = math.factorial(n - k)
    except BaseException:
        print("Numbers too large to compute number pairs. n=", n, "\t k=", k)

    return a / b
