from __future__ import absolute_import
from __future__ import print_function

import math
import numpy as np
import os as os
import sys as sys
import itertools
import pandas as pd
import time
from time import sleep
import random
import logging

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import redis
import rq
import seaborn as sns

# QMLA functionality
import qmla.analysis
import qmla.database_framework as database_framework
import qmla.get_growth_rule as get_growth_rule
import qmla.redis_settings as rds
import qmla.model_for_storage
from qmla.remote_bayes_factor import remote_bayes_factor_calculation
from qmla.remote_model_learning import remote_learn_model_parameters
import qmla.growth_rule_tree
import qmla.utilities

pickle.HIGHEST_PROTOCOL = 4  # if <python3, must use lower protocol
plt.switch_backend('agg')

__all__ = [
    'QuantumModelLearningAgent'
]


class QuantumModelLearningAgent():
    r"""
    QMLA manager class.

    Controls the infrastructure which determines which models are learned and compared.
    By interpreting user defined :class:`~qmla.growth_rules.GrowthRule`,
    grows :class:`~qmla.GrowthRuleTree` objects which hold numerous models
    on :class:`~qmla.BranchQMLA` objects.
    All models on branches are learned and then compared.
    The comparisons on a branch inform the next set of models generated on that tree.

    First calls a series of setup functions to implement
    infrastructure used throughout.

    The available algorithms, and their corresponding methods, are:
        - Quantum Hamilontian Learning:

            :meth:`~qmla.QuantumModelLearningAgent.run_quantum_hamiltonian_learning`
        - Quantum Hamilontian Learning multiple models:

            :meth:`~qmla.QuantumModelLearningAgent.run_quantum_hamiltonian_learning_multiple_models`
        - Quantum Model Learning Agent:

            :meth:`~qmla.QuantumModelLearningAgent.run_complete_qmla`

    :param ControlsQMLA qmla_controls: Storage for configuration of a QMLA instance.
    :param dict model_priors: values of means/widths to enfore on given models,
        specifically for further_qhl mode.
    :param dict experimental_measurements: expectation values by time of the
        underlying true/target model.

    """

    def __init__(self,
                 qmla_controls=None,
                 model_priors=None,
                 experimental_measurements=None,
                 **kwargs
                 ):
        # logging.info('QMLA class instance')
        self._start_time = time.time()  # to measure run-time

        # Configure this QMLA instance
        if qmla_controls is None:
            self.qmla_controls = qmla.controls_qmla.parse_cmd_line_args(
                args={}
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
            experimental_measurements=experimental_measurements,
        )

        # Resources potentially reallocated
        self._compute_base_resources()

        # Redundant attributes, retained for legacy; to be removed
        self._potentially_redundant_setup()

        # Check if QMLA should run in parallel and set up accordingly
        self._setup_parallel_requirements()

        # QMLA core info stored on redis server
        self._compile_and_store_qmla_info_summary()

        # Set up infrastructure related to growth rules and tree management
        self._setup_tree_and_growth_rules()

    ##########
    # Section: Initialisation and setup
    ##########

    def _fundamental_settings(self):
        r""" Basic settings, path definitions etc."""

        # Extract info from Controls
        self.qmla_id = self.qmla_controls.qmla_id
        self.redis_host_name = self.qmla_controls.host_name
        self.redis_port_number = self.qmla_controls.port_number
        self.log_file = self.qmla_controls.log_file
        self.qhl_mode = self.qmla_controls.qhl_mode
        self.qhl_mode_multiple_models = self.qmla_controls.qhl_mode_multiple_models
        self.latex_name_map_file_path = self.qmla_controls.latex_mapping_file
        self.results_directory = self.qmla_controls.results_directory

        # Databases for storing learning/comparison data
        self.redis_databases = rds.get_redis_databases_by_qmla_id(
            self.redis_host_name,
            self.redis_port_number,
            self.qmla_id,
        )
        self.redis_databases['any_job_failed'].set('Status', 0)
        print("[QMLA] Redis databases set.")

        # Logistics
        self.models_learned = []
        self.timings = {
            # track times spent in some subroutines
            'inspect_job_crashes': 0,
            'jobs_finished': 0
        }
        self.call_counter = {
            # track number of calls to some subroutines
            'job_crashes': 0,
            'jobs_finished': 0,
        }
        self.sleep_duration = 10

    def _true_model_definition(self):
        r""" Information related to true (target) model."""

        self.true_model_name = database_framework.alph(
            self.qmla_controls.true_model_name)
        self.true_model_dimension = database_framework.get_num_qubits(
            self.true_model_name)
        self.true_model_constituent_operators = self.qmla_controls.true_model_terms_matrices
        self.true_model_constituent_terms_latex = [
            self.growth_class.latex_name(term)
            for term in
            qmla.database_framework.get_constituent_names_from_name(
                self.true_model_name)
        ]
        self.true_model_num_params = self.qmla_controls.true_model_class.num_constituents
        self.true_param_list = self.growth_class.true_params_list
        self.true_param_dict = self.growth_class.true_params_dict
        self.true_model_branch = -1  # overwrite if true model is added to database
        self.true_model_considered = False
        self.true_model_found = False
        self.true_model_id = -1
        self.true_model_on_branhces = []
        self.true_model_hamiltonian = self.growth_class.true_hamiltonian
        self.log_print([
            "True model:", self.true_model_name
        ])

    def _setup_tree_and_growth_rules(
        self,
    ):
        r""" Set up infrastructure."""

        # TODO most of this can probably go inside run_complete_qmla
        # Infrastructure
        self.model_database = pd.DataFrame({
            'model_id': [],
            'model_name': [],
            'branch_id': [],
            'model_storage_instance': [],
        })
        self.model_lists = {
            # assumes maxmium 13 qubit-models considered
            # to be checked when checking model_lists
            # TODO generalise to max dim of Growth Rule
            j: []
            for j in range(1, 13)
        }
        self.all_bayes_factors = {}
        self.bayes_factor_pair_computed = []

        # Growth rule setup
        self.growth_rules_list = self.qmla_controls.generator_list
        self.growth_rule_of_true_model = self.qmla_controls.growth_generation_rule
        self.unique_growth_rule_instances = self.qmla_controls.unique_growth_rule_instances

        # Keep track of models/branches
        self.model_count = 0
        self.highest_model_id = 0  # so first created model gets model_id=0
        self.models_branches = {}
        self.branch_highest_id = 0
        self.model_name_id_map = {}
        self.ghost_branches = {}

        # Tree object for each growth rule
        self.trees = {
            gen: qmla.growth_rule_tree.GrowthRuleTree(
                growth_class=self.unique_growth_rule_instances[gen]
            )
            for gen in self.unique_growth_rule_instances
        }
        self.branches = {}
        self.tree_count = len(self.trees)
        self.tree_count_completed = np.sum(
            [tree.is_tree_complete() for tree in self.trees.values()]
        )

    def _set_learning_and_comparison_parameters(
        self,
        model_priors,
        experimental_measurements,
    ):
        r""" Parameters related to learning/comparing models."""

        # Miscellaneous
        self.model_priors = model_priors

        # Learning parameters, used by QInfer updates
        self.num_particles = self.qmla_controls.num_particles
        self.num_experiments = self.qmla_controls.num_experiments
        self.num_experiments_for_bayes_updates = self.qmla_controls.num_times_bayes
        self.bayes_threshold_lower = self.qmla_controls.bayes_lower
        self.bayes_threshold_upper = self.qmla_controls.bayes_upper
        self.qinfer_resample_threshold = self.qmla_controls.resample_threshold
        self.qinfer_resampler_a = self.qmla_controls.resample_a
        self.qinfer_PGH_heuristic_factor = self.qmla_controls.pgh_factor
        self.qinfer_PGH_heuristic_exponent = self.qmla_controls.pgh_exponent

        # Tracking for analysis
        self.model_f_scores = {}
        self.model_precisions = {}
        self.model_sensitivities = {}
        self.bayes_factors_df = pd.DataFrame()

        # Get probes used for learning
        self.growth_class.generate_probes(
            noise_level=self.qmla_controls.probe_noise_level,
            minimum_tolerable_noise=0.0,
        )
        self.probes_system = self.growth_class.probes_system
        self.probes_simulator = self.growth_class.probes_simulator
        self.probe_number = self.growth_class.num_probes

        # Measurements of true model
        self.experimental_measurements = experimental_measurements
        self.experimental_measurement_times = (
            sorted(list(self.experimental_measurements.keys()))
        )

        # Used for consistent plotting
        self.times_to_plot = self.experimental_measurement_times
        self.times_to_plot_reduced_set = self.times_to_plot[0::10]
        self.probes_plot_file = self.qmla_controls.probes_plot_file
        try:
            self.probes_for_plots = pickle.load(
                open(self.probes_plot_file, 'rb')
            )
        except BaseException:
            self.log_print(
                [
                    "Could not load plot probes from {}".format(
                        self.probes_plot_file
                    )
                ]
            )

    def _potentially_redundant_setup(
        self,
    ):
        r"""
        Graveyard for deprecated ifnrastructure.

        Attributes etc stored here which are not functionally used
        within QMLA, but which are called somewhere,
        and cause errors when omitted.
        Should be stored here temporarily during development,
        and removed entirely when sure they are not needed.

        """

        # Some functionality towards time dependent models
        self.use_time_dependent_true_model = False
        self.num_time_dependent_true_params = 0
        self.time_dependent_params = None

        # Plotting data about pairwise comparisons
        self.instance_learning_and_comparisons_path = os.path.join(
            # self.results_directory,
            # 'instance_learning_and_comparisons', 
            # str(self.qmla_id)
            self.qmla_controls.plots_directory, 
            'comparisons'
        )
        if not os.path.exists(self.instance_learning_and_comparisons_path):
            try:
                os.makedirs(self.instance_learning_and_comparisons_path)
            except BaseException:
                # reached at exact same time as another process; don't crash
                pass
        self.bayes_factors_store_times_file = str(
            self.instance_learning_and_comparisons_path
            + 'BayesFactorsPairsTimes_'
            + str(self.qmla_controls.long_id)
            + '.txt'
        )

    def _setup_parallel_requirements(self):
        r""" Infrastructure for use when QMLA run in parallel. """

        self.use_rq = self.qmla_controls.use_rq
        self.rq_timeout = self.qmla_controls.rq_timeout
        self.rq_log_file = self.log_file
        # writeable file object to use for logging
        self.write_log_file = open(self.log_file, 'a')

        try:
            self.redis_conn = redis.Redis(
                host=self.redis_host_name,
                port=self.redis_port_number
            )
            parallel_enabled = True
        except BaseException:
            self.log_print("Importing rq failed: enforcing serial.")
            parallel_enabled = False
        self.run_in_parallel = parallel_enabled

    def _compute_base_resources(self):
        r"""
        Compute the set of minimal resources for models to learn on.

        In the case self.reallocate_resources==True,
        models will receive resources (epochs, particles)
        scaled by how complicated they are.
        For instance, models with 4 parameters will receive
        twice as many particles as a model with
        2 parameters.
        """

        # Decide if reallocating resources based on true GR; 
        # TODO In models, check base resources to decide whether to reallocate
        if self.growth_class.reallocate_resources:
            base_num_qubits = 3
            base_num_terms = 3
            for op in self.growth_class.initial_models:
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
                'reallocate' : True
            }
        else:
            self.base_resources = {
                'num_qubits': 1,
                'num_terms': 1,
                'reallocate' : False
            }

    def _compile_and_store_qmla_info_summary(
        self
    ):
        r"""
        Gather info needed to run QMLA tasks and store remotely.

        QMLA issues jobs to run remotely, namely for model (parameter) 
        learning and model comparisons (Bayes factors).
        These jobs don't need access to all QMLA data, but do need
        some common info, e.g. number of particles and epochs.
        This function gathers all relevant information in a single dict, 
        and stores it on the redis server which all worker nodes have access to.
        It also stores the probe sets required for the same tasks.

        """

        number_hamiltonians_to_exponentiate = (
            self.num_particles *
            (self.num_experiments + self.num_experiments_for_bayes_updates)
        )
        self.latex_config = str(
            '$P_{' + str(self.num_particles) +
            '}E_{' + str(self.num_experiments) +
            '}B_{' + str(self.num_experiments_for_bayes_updates) +
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
            'true_param_dict': self.true_param_dict,
            'num_particles': self.num_particles,
            'num_experiments': self.num_experiments,
            'results_directory': self.results_directory,
            'plots_directory': self.qmla_controls.plots_directory,
            'long_id': self.qmla_controls.long_id,
            'model_priors': self.model_priors, # could be path to unpickle within model?
            'experimental_measurements': self.experimental_measurements, # could be path to unpickle within model?
            'base_resources': self.base_resources, 
            'store_particles_weights': False,  # TODO from growth rule or unneeded
            'qhl_plots': False,  # TODO get from growth rule
            'experimental_measurement_times': self.experimental_measurement_times,
            'num_probes': self.probe_number,  # from growth rule or unneeded,
            'true_params_pickle_file': self.qmla_controls.true_params_pickle_file,
        }

        # Store qmla_settings and probe dictionaries on the redis database, 
        # accessible by all workers.
        # These are retrieved by workers to set
        # parameters to use when learning/comparing models.
        compressed_qmla_core_info = pickle.dumps(
            self.qmla_settings, protocol=4)
        compressed_probe_dict = pickle.dumps(self.probes_system, protocol=4)
        compressed_sim_probe_dict = pickle.dumps(
            self.probes_simulator, protocol=4)
        qmla_core_info_database = self.redis_databases['qmla_core_info_database']
        qmla_core_info_database.set('qmla_settings', compressed_qmla_core_info)
        qmla_core_info_database.set('ProbeDict', compressed_probe_dict)
        qmla_core_info_database.set('SimProbeDict', compressed_sim_probe_dict)

        self.qmla_core_info_database = {
            'qmla_settings': self.qmla_settings,
            'ProbeDict': self.probes_system,
            'SimProbeDict': self.probes_simulator
        }
        self.log_print(
            ["Saved QMLA instance info to ", qmla_core_info_database])

    ##########
    # Section: Calculation of models parameters and Bayes factors
    ##########

    def learn_models_on_given_branch(
        self,
        branch_id,
        blocking=False
    ):
        r"""
        Launches jobs to learn all models on the specified branch.

        Models which are on the branch but have already been learned are not re-learned.
        For each remaining model on the branch,
        :meth:`~qmla.QuantumModelLearningAgent.learn_model` is called.
        The branch is added to the redis database `active_branches_learning_models`,
        indicating that branch_id has currently got models in the learning phase.
        This redis database is monitored by the :meth:`~qmla.QuantumModelLearningAgent.learn_models_until_trees_complete`.
        When all models registered on the branch have completed, it is recorded, allowing QMLA
        to perform the next stage: either spawning a new branch from this branch, or
        continuing to the final stage of QMLA.
        This method can block, meaning it waits for a model's learning to complete
        before proceeding. If in parallel, do not block as model learning
        won't be launched until the previous model has completed.

        :param int branch_id: unique QMLA branch ID to learn models of.
        :param bool use_rq: whether to implement learning via RQ workers.
            Argument only used when passed to :meth:`QuantumModelLearningAgent.learn_model`.
        :param bool blocking: whether to wait on all models' learning before proceeding.
        """

        model_list = self.branches[branch_id].resident_models
        num_models_already_set_this_branch = self.branches[branch_id].num_precomputed_models
        unlearned_models_this_branch = self.branches[branch_id].unlearned_models

        # Update redis database
        active_branches_learning_models = (
            self.redis_databases['active_branches_learning_models']
        )
        active_branches_learning_models.set(
            int(branch_id),
            num_models_already_set_this_branch
        )

        # Learn models
        self.log_print([
            "Branch {} has models: \nprecomputed: {} \nunlearned: {}".format(
                branch_id,
                self.branches[branch_id].precomputed_models,
                unlearned_models_this_branch
            )
        ])

        for model_name in unlearned_models_this_branch:
            self.learn_model(
                model_name=model_name,
                branch_id=branch_id,
                blocking=blocking
            )
        self.log_print(
            [
                'Learning models from branch {} finished.'.format(branch_id)
            ]
        )

    def learn_model(
        self,
        model_name,
        branch_id,
        blocking=False
    ):
        r"""
        Learn a given model by calling the standalone model learning functionality.

        The model is learned by launching a job either locally or to the job queue.
        Model learning is implemented by :func:`remote_learn_model_parameters`,
        which takes a unique model name (string) and distills the terms to learn.
        If running locally, QMLA core info is passed.
        Else if RQ workers are being used, it retrieves QMLA info from the shared redis
        database, and the function is launched via rq's `Queue.enqueue` function.
        This puts a task on the redis `Queue` - the task is the implementation of
        :func:`remote_learn_model_parameters`.
        The effect is either to learn the model here, or else to have launched a job
        where it will be learned remotely, so nothing is returned.

        :param str model_name: string uniquely representing a model
        :param int branch_id: unique branch ID within QMLA environment
        :param bool use_rq: whether to use RQ workers, or implement locally
        :param bool blocking: whether to wait on model to finish learning before proceeding.
        """

        model_already_exists = database_framework.check_model_exists(
            model_name=model_name,
            model_lists=self.model_lists,
            db=self.model_database
        )

        if not model_already_exists:
            self.log_print([
                "Model {} not yet in database: can not be learned.".format(
                    model_name
                )
            ])
        else:
            model_id = database_framework.model_id_from_name(
                self.model_database,
                name=model_name
            )
            if model_id not in self.models_learned:
                self.models_learned.append(model_id)

            if self.run_in_parallel and self.use_rq:
                # get access to the RQ queue
                queue = rq.Queue(
                    self.qmla_id,
                    connection=self.redis_conn,
                    async=self.use_rq,
                    default_timeout=self.rq_timeout
                )
                self.log_print(["Redis queue object:", queue, "has job waiting IDs:", queue.job_ids])
                # send model learning as task to job queue
                queued_model = queue.enqueue(
                    remote_learn_model_parameters,
                    model_name,
                    model_id,
                    growth_generator=self.branches[branch_id].growth_rule,
                    branch_id=branch_id,
                    remote=True,
                    host_name=self.redis_host_name,
                    port_number=self.redis_port_number,
                    qid=self.qmla_id,
                    log_file=self.rq_log_file,
                    result_ttl=-1,
                    timeout=self.rq_timeout
                )
                self.log_print(["Model {} on rq job {}".format(model_id, queued_model)])
                if blocking:
                    # wait for result when called.
                    self.log_print([
                        "Blocking: waiting for {} to finish on redis queue".format(
                            model_name
                        )
                    ])
                    while not queued_model.is_finished:
                        t_init = time.time()
                        some_job_failed = queued_model.is_failed
                        self.timings['jobs_finished'] += time.time() - t_init
                        self.call_counter['jobs_finished'] += 1
                        if some_job_failed:
                            self.log_print([
                                "Model", model_name,
                                "has failed on remote worker."
                            ])
                            raise NameError("Remote QML failure")
                            break
                        time.sleep(self.sleep_duration)
                    self.log_print(
                        ['Blocking RQ - model learned:', model_name]
                    )
            else:
                # run model learning fnc locally
                self.log_print([
                    "Locally calling learn model function.",
                    "model:", model_name,
                    " ID:", model_id
                ])
                # pass probes directly instead of unpickling from redis database
                self.qmla_settings['probe_dict'] = self.probes_system

                remote_learn_model_parameters(
                    name=model_name,
                    model_id=model_id,
                    growth_generator=self.branches[branch_id].growth_rule,
                    branch_id=branch_id,
                    qmla_core_info_dict=self.qmla_settings,
                    remote=True,
                    host_name=self.redis_host_name,
                    port_number=self.redis_port_number,
                    qid=self.qmla_id,
                    log_file=self.rq_log_file
                )

    def compare_model_pair(
        self,
        model_a_id,
        model_b_id,
        return_job=False,
        branch_id=None,
        remote=True,
        wait_on_result=False
    ):
        r"""
        Launch the comparison between two models.

        Either locally or by passing to a job queue,
        run :func:`remote_bayes_factor_calculation`
        for a pair of models specified by their IDs.

        :param int model_a_id: unique ID of one model of the pair
        :param int model_b_id: unique ID of other model of the pair
        :param bool return_job:
            True - return the rq job object from this function call.
            False (default) - return nothing.
        :param int branch_id: unique branch ID, if this model pair
            are on the same branch
        :param bool remote: whether to run the job remotely or locally
            True - job is placed on queue for RQ worker
            False - function is computed locally immediately
        :param bool wait_on_result: whether to wait for the outcome
            or proceed after sending the job to the queue.
        :returns bayes_factor: the Bayes factor calculated between the two models,
            i.e. BF(m1,m2) where m1 is the lower model id. Only returned when
            `wait_on_result==True`.
        """

        unique_id = database_framework.unique_model_pair_identifier(
            model_a_id,
            model_b_id
        )
        if (
            unique_id not in self.bayes_factor_pair_computed
        ):
            self.bayes_factor_pair_computed.append(
                unique_id
            )

        # Launch comparison, either remotely or locally
        if self.use_rq:
            # launch remotely
            from rq import Connection, Queue, Worker
            queue = Queue(self.qmla_id, connection=self.redis_conn,
                          async=self.use_rq, default_timeout=self.rq_timeout
                          )

            # the function object is the first argument to RQ enqueue function
            job = queue.enqueue(
                remote_bayes_factor_calculation,
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                branch_id=branch_id,
                times_record=self.bayes_factors_store_times_file,
                bf_data_folder=self.instance_learning_and_comparisons_path,
                num_times_to_use=self.num_experiments_for_bayes_updates,
                bayes_threshold=self.bayes_threshold_lower,
                host_name=self.redis_host_name,
                port_number=self.redis_port_number,
                qid=self.qmla_id,
                log_file=self.rq_log_file,
                result_ttl=-1,
                timeout=self.rq_timeout
            )
            self.log_print([
                "Bayes factor calculation queued. Models {}/{}".format(
                    model_a_id, model_b_id
                )
            ])
            if wait_on_result == True:
                while not job.is_finished:
                    if job.is_failed:
                        raise("Remote BF failure")
                    sleep(self.sleep_duration)
            elif return_job == True:
                return job
        else:
            # run comparison locally 
            remote_bayes_factor_calculation(
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                bf_data_folder=self.instance_learning_and_comparisons_path,
                times_record=self.bayes_factors_store_times_file,
                num_times_to_use=self.num_experiments_for_bayes_updates,
                branch_id=branch_id,
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

    def compare_model_set(
        self,
        model_id_list=None,
        pair_list=None,
        remote=True,
        wait_on_result=False,
        recompute=False,
    ):
        r"""
        Launch pairwise model comparison for a set of models.

        If `pair_list` is specified, those pairs are compared;
        otherwise all pairs within `model_id_list` are compared.

        Pairs are sent to :meth:`~qmla.QuantumModelLearningAgent.compare_model_pair`
        to be computed either locally or on a job queue.

        :param list model_id_list: list of model names to compute comparisons between
        :param list pair_list: list of tuples specifying model IDs to compare
        :param bool remote:
            passed directly to :meth:`~qmla.QuantumModelLearningAgent.compare_model_pair`
        :param bool wait_on_results:
            passed directly to :meth:`~qmla.QuantumModelLearningAgent.compare_model_pair`
        :param bool recompute: whether to force comparison even if a pair has
            been compared previously

        """

        if pair_list is None:
            pair_list = list(itertools.combinations(
                model_id_list, 2
            ))
            self.log_print(
                [
                    "BF Pair list not provided; generated from model list:",
                    pair_list
                ]
            )

        remote_jobs = []
        for pair in pair_list:
            unique_id = database_framework.unique_model_pair_identifier(
                pair[0], pair[1]
            )
            if (
                unique_id not in self.bayes_factor_pair_computed
                or recompute == True
            ):
                # ie not yet considered
                remote_jobs.append(
                    self.compare_model_pair(
                        pair[0],
                        pair[1],
                        remote=remote,
                        return_job=wait_on_result,
                    )
                )

        if wait_on_result and self.use_rq:
            self.log_print([
                "Waiting on result of ",
                "Bayes comparisons from given model list:",
                model_id_list,
                "\n pair list:", pair_list
            ])
            for job in remote_jobs:
                while not job.is_finished:
                    if job.is_failed:
                        raise NameError("Remote QML failure")
                    time.sleep(self.sleep_duration)
        else:
            self.log_print([
                "Not waiting on results of BF calculations",
                "since we're not using RQ workers here."
            ])

    def compare_models_within_branch(
        self,
        branch_id,
        pair_list=None,
        remote=True,
        recompute=False
    ):
        r"""
        Launch pairwise model comparison for all models on a branch.

        If `pair_list` is specified, those pairs are compared;
        otherwise pairs are retrieved from the `pairs_to_compare`
        attribute of the branch, which is usually all-to-all.

        Pairs are sent to :meth:`~qmla.QuantumModelLearningAgent.compare_model_pair`
        to be computed either locally or on a job queue.

        :param branch_id: unique ID of the branch within the QMLA environment
        :param list pair_list: list of tuples specifying model IDs to compare
        :param bool remote:
            passed directly to :meth:`~qmla.QuantumModelLearningAgent.compare_model_pair`
        :param bool wait_on_results:
            passed directly to :meth:`~qmla.QuantumModelLearningAgent.compare_model_pair`
        :param bool recompute: whether to force comparison even if a pair has
            been compared previously
        """

        if pair_list is None:
            pair_list = self.branches[branch_id].pairs_to_compare
        active_branches_bayes = self.redis_databases['active_branches_bayes']
        active_branches_bayes.set(int(branch_id), 0)  # set up branch 0
        self.log_print([
            'compare_models_within_branch. branch {} has pairs {}'.format(
                branch_id,
                pair_list
            )
        ])

        for pair in pair_list:
            a = pair[0]
            b = pair[1]
            if a != b:
                unique_id = database_framework.unique_model_pair_identifier(
                    a, b
                )
                if (
                    unique_id not in self.bayes_factor_pair_computed
                    or recompute == True
                ):
                    # ie not yet considered or recomputing
                    self.compare_model_pair(
                        a,
                        b,
                        remote=remote,
                        branch_id=branch_id,
                    )
                elif unique_id in self.bayes_factor_pair_computed:
                    # if this is already computed,
                    # tell this branch not to wait on it.
                    active_branches_bayes.incr(
                        int(branch_id),
                        1
                    )

    def process_model_pair_comparison(
        self,
        a=None,
        b=None,
        pair=None,
    ):
        r"""
        Process a comparison between two models.

        The comparison (Bayes factor) result is retrieved from the
        redis database and used to update data on the models.

        :param int a: one of the model's unique ID
        :param int b: one of the model's unique ID
        :param tuple pair: alternative mechanism to provide the model IDs,
            effectively (a,b)
        :return: ID of the model which is deemed superior
            from this pair
        """

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
            self.log_print([
                "Must pass either two model ids, or a \
                pair name string, to process Bayes factors."
            ])
        try:
            bayes_factor = float(
                bayes_factors_db.get(pair)
            )
        except TypeError:
            self.log_print([
                "On bayes_factors_db for pair {} = {}".format(
                    pair,
                    bayes_factors_db.get(pair)
                )
            ])

        # Bayes_factor refers to calculation BF(pair), where pair
        # is always defined (lower, higher) for consistency
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

        # Tell growth rule's rating system about this comparison
        # TODO more general fnc to tell GR about this comparison
        self.growth_class.ratings_class.compute_new_ratings(
            model_a_id=mod_low.model_id,
            model_b_id=mod_high.model_id,
            winner_id=champ,
            bayes_factor = bayes_factor,
            spawn_step = self.growth_class.spawn_step, 
        )   
        # self.growth_class.record_comparison(
        #     model_a_id=mod_low.model_id,
        #     model_b_id=mod_high.model_id,
        #     winner_id=champ,
        #     bayes_factor = bayes_factor,
        #     spawn_step = self.growth_class.spawn_step, 
        # )

        return champ

    def process_model_set_comparisons(
        self,
        model_list,
    ):
        r"""
        Process comparisons between a set of models.

        Pairwise comparisons are retrieved and processed by
        :meth:`~qmla.QuantumModelLearningAgent.process_model_pair_comparison`,
        which informs the superior model.

        For each pairwise comparison a given model wins, it receives a single point.

        All comparisons are weighted evenly.
        Model points are gathered; the model with most points is
        deemed the champion of the set.

        If a subset of models have the same (highest) number of points,
        that subset is compared directly, with the nominated champion
        deemed the champion of the wider set.

        :param list model_list: list of model names to compete
        :return: unique model ID of the champion model within the set

        """

        # Establish pairs to check comparisons between
        pair_list = list(itertools.combinations(
            model_list, 2
        ))

        # Process result for each pair
        models_points = {
            mod: 0
            for mod in model_list
        }
        for pair in pair_list:
            mod1, mod2 = pair
            if mod1 != mod2:
                res = self.process_model_pair_comparison(a=mod1, b=mod2)
                models_points[res] += 1
                self.log_print(
                    [
                        "[process_model_set_comparisons]",
                        "Point to", res,
                        "(comparison {}/{})".format(mod1, mod2)
                    ]
                )

        # Analyse pairwise competition
        max_points = max(models_points.values())
        models_with_max_points = [key for key, val in models_points.items()
                                  if val == max_points]
        if len(models_with_max_points) > 1:
            self.log_print([
                "Multiple models \
                have same number of points in process_model_set_comparisons:",
                models_with_max_points
            ])
            for i in models_with_max_points:
                self.log_print(
                    [
                        database_framework.model_name_from_id(
                            self.model_database, i)
                    ]
                )
            self.log_print(["After re-comparison, points:\n", models_points])
            self.compare_model_set(
                model_id_list=models_with_max_points,
                remote=True,
                recompute=True,  # recompute here b/c deadlock last time
                wait_on_result=True
            )
            champ_id = self.process_model_set_comparisons(
                models_with_max_points,
            )
        else:
            self.log_print(["After comparing list, points:\n", models_points])
            champ_id = max(models_points, key=models_points.get)

        return champ_id

    def process_comparisons_within_branch(
        self,
        branch_id,
        pair_list=None
    ):
        r"""
        Process comparisons between models on the same branch.

        (Similar functionality to 
        :meth:`~qmla.QuantumModelLearningAgent.process_model_set_comparisons`,
        but additionally updates some branch infrastructure, such as updating 
        the branch's `champion_id`, `bayes_points` attributes).
        Pairwise comparisons are retrieved and processed by
        :meth:`~qmla.QuantumModelLearningAgent.process_model_pair_comparison`, 
        which informs the superior model.
        For each pairwise comparison a given model wins, it receives a single point.
        All comparisons are weighted evenly.
        Model points are gathered; the model with most points is
        deemed the champion of the set.
        If a subset of models have the same (highest) number of points,
        that subset is compared directly, with the nominated champion
        deemed the champion of the wider set.

        :param int branch_id: unique ID of the branch whose models to compare
        :returns:
            -   models_points: the points (number of comparisons won)
                of each model on the branch
            -   champ_id: unique model ID of the champion model within the set

        """

        # Establish pairs to check comparisons between
        if pair_list is None:
            pair_list = self.branches[branch_id].pairs_to_compare
            self.log_print([
                "Pair list not given for branch {}, generated:{}".format(
                    branch_id,
                    pair_list
                ),
            ])
            active_models_in_branch = self.branches[branch_id].resident_model_ids

        # Process result for each pair
        models_points = {
            k: 0
            for k in active_models_in_branch
        }
        for pair in pair_list:
            mod1 = pair[0]
            mod2 = pair[1]
            if mod1 != mod2:
                res = self.process_model_pair_comparison(
                    a=mod1, b=mod2
                )
                try:
                    models_points[res] += 1
                except BaseException:
                    models_points[res] = 1
                self.log_print([
                    "[branch {} comparison {}/{}] ".format(
                        branch_id, mod1, mod2
                    ),
                    "Point to", res,
                ])
        self.log_print(["Comparisons complete on branch {}".format(branch_id)])
        
        # Analyse pairwise competition
        max_points = max(models_points.values())
        models_with_max_points = [
            key for key, val in models_points.items()
            if val == max_points
        ]

        if len(models_with_max_points) > 1:
            # if multiple models have same number of wins,
            # run competition beween subset
            self.log_print([
                "Multiple models have same number of points within \
                    branch.\n",
                models_points,
                "This may cause infinite loop if models can \
                    not be separated.",
            ])
            self.compare_model_set(
                model_id_list=models_with_max_points,
                remote=True,
                recompute=False,
                wait_on_result=True
            )
            champ_id = self.process_model_set_comparisons(
                models_with_max_points,
            )
        else:
            # champion is model with most points
            champ_id = max(models_points, key=models_points.get)

        # Update branch object with results of competition
        champ_id = int(champ_id)
        champ_name = self.model_name_id_map[champ_id]
        champ_num_qubits = database_framework.get_num_qubits(champ_name)
        ranked_model_list = sorted(
            models_points,
            key=models_points.get,
            reverse=True
        )

        # Update the branch with comparison data
        # TODO give these data to GR but let it decide chamion etc, in case decided otherwise than BF contest
        self.branches[branch_id].champion_id = champ_id
        self.branches[branch_id].champion_name = champ_name
        self.branches[branch_id].rankings = ranked_model_list
        self.branches[branch_id].bayes_points = models_points
        self.log_print(["Setting eval log like on branch"])
        self.branches[branch_id].evaluation_log_likelihoods = {
            k: self.get_model_storage_instance_by_id(k).evaluation_log_likelihood
            for k in self.branches[branch_id].resident_model_ids
        }
        # self.growth_class.record_comparisons(comparisons = models_points)

        self.log_print([
            "Model points for branch {}: {}".format(
                branch_id,
                models_points,
            ),
            "\nChampion of branch {} is model {}: {}".format(
                branch_id,
                champ_id,
                champ_name
            )
        ])

        # Return the winning model's ID
        return champ_id

    ##########
    # Section: routines to implement tree-based QMLA
    ##########

    def learn_models_until_trees_complete(
        self,
    ):
        r"""
        Iteratively learn/compare/generate models on growth rule trees.

        Each :class:`~qmla.growth_rules.GrowthRule` has a unique :class:`~qmla.QMLATree``.
        Trees hold sets of models on :class:`~qmla.BranchTree` objects.

        Models on a each branch are learned through :meth:`learn_models_on_given_branch`.
        Any model which has previously been considered defaults to the earlier
        instance of that model, rather than repeating the calculation.
        When all models on a branch are learned, they are all compared
        through :meth:`compare_models_within_branch`.

        When a branch has completed learning and comparisons of models,
        the corresponding tree is checked to see if it has finished proposing
        models, through :meth:`~qmla.GrowthRuleTree.is_tree_complete`.
        If the tree is not complete, the :meth:`~qmla.GrowthRuleTree.next_layer`
        method is called to generate the next branch on that tree.
        The next branch can correspond to `spawn` or `prune` stages of the
        tree's :class:`~qmla.growth_rules.GrowthRule`, but QMLA is ambivalent to the
        inner workings of the tree/growth rule: a branch is
        simply a set of models to learn and compare.

        When all trees have completed learning, this method terminates.
        """

        # Get redis databases
        active_branches_learning_models = (
            self.redis_databases['active_branches_learning_models']
        )
        active_branches_bayes = self.redis_databases['active_branches_bayes']

        # Launch learning on initial branches
        for b in self.branches:
            self.learn_models_on_given_branch(
                b,
                blocking=False,
            )
        self.log_print([
            "Starting learning for initial branches:",
            list(self.branches.keys())
        ])

        # Iteratively learn/compare/spawn until all trees declare completion
        self.log_print([
            "Entering while loop: learning/comparing/spawning models."
        ])
        while self.tree_count_completed < self.tree_count:
            # get most recent branches on redis database
            branch_ids_on_db = list(
                active_branches_learning_models.keys()
            )
            branch_ids_on_db = [
                int(b) for b in branch_ids_on_db
            ]

            # check if any job has crashed
            if self.run_in_parallel:
                self._inspect_remote_job_crashes()

            # loop through active branches
            for branch_id in branch_ids_on_db:

                # inspect if branch has finished learning
                num_models_learned_on_branch = int(
                    active_branches_learning_models.get(branch_id)
                )
                if (
                    not self.branches[branch_id].model_learning_complete
                    and
                    num_models_learned_on_branch == self.branches[branch_id].num_models
                ):
                    self.log_print([
                        "All models on branch {} learned".format(branch_id)
                    ])
                    self.branches[branch_id].model_learning_complete = True
                    for mod_id in self.branches[branch_id].resident_model_ids:
                        mod = self.get_model_storage_instance_by_id(mod_id)
                        mod.model_update_learned_values()
                    # launch comparisons
                    self.compare_models_within_branch(branch_id)

            for branchID_bytes in active_branches_bayes.keys():
                branch_id = int(branchID_bytes)
                num_comparisons_complete_on_branch = active_branches_bayes.get(
                    branchID_bytes
                )
                if (
                    not self.branches[branch_id].comparisons_complete
                    and
                    int(
                        num_comparisons_complete_on_branch) == self.branches[branch_id].num_model_pairs
                ):
                    self.branches[branch_id].comparisons_complete = True
                    # analyse resulting bayes factors
                    self.log_print([
                        "Branch {} comparisons starting".format(branch_id)
                    ])
                    self.process_comparisons_within_branch(branch_id)
                    self.log_print([
                        "Branch {} comparisons complete".format(branch_id)
                    ])

                    # check if tree is complete
                    if self.branches[branch_id].tree.is_tree_complete():
                        self.tree_count_completed += 1
                        self.log_print([
                            "Tree complete:",
                            self.branches[branch_id].growth_rule,
                            "Number of trees now completed:",
                            self.tree_count_completed,
                        ])
                    else:
                        # tree not complete -> launch next set of models
                        self.spawn_from_branch(
                            branch_id=branch_id,
                        )

        self.log_print([
            "{} trees have completed. Waiting on final comparisons".format(
                self.tree_count_completed
            )
        ])

        # Allow any branches which have just started to finish
        still_learning = True
        while still_learning:
            branch_ids_on_db = list(active_branches_learning_models.keys())
            for branchID_bytes in branch_ids_on_db:
                branch_id = int(branchID_bytes)
                if (
                    (
                        int(active_branches_learning_models.get(branch_id))
                        == self.branches[branch_id].num_models
                    )
                    and
                    self.branches[branch_id].model_learning_complete == False
                ):
                    self.branches[branch_id].model_learning_complete = True
                    self.compare_models_within_branch(branch_id)
                    for mod_id in self.branches[branch_id].resident_model_ids:
                        mod = self.get_model_storage_instance_by_id(mod_id)
                        mod.model_update_learned_values()

                if branchID_bytes in active_branches_bayes:
                    num_comparisons_complete_on_branch = (
                        active_branches_bayes.get(branchID_bytes)
                    )
                    if (
                        (
                            int(num_comparisons_complete_on_branch)
                            == self.branches[branch_id].num_model_pairs
                        )
                        and
                        (
                            self.branches[branch_id].comparisons_complete == False
                        )
                    ):
                        self.branches[branch_id].comparisons_complete = True
                        self.process_comparisons_within_branch(branch_id)

            if (
                np.all(np.array([
                    self.branches[b].model_learning_complete for b in self.branches
                ]))
                and
                np.all(np.array([
                    self.branches[b].comparisons_complete for b in self.branches
                ]))
            ):
                # break out of this while loop
                still_learning = False

        # Finalise all trees.
        for tree in self.trees.values():
            tree.finalise_tree(
                model_names_ids=self.model_name_id_map,
            )

        self.log_print(["Learning stage complete on all trees."])

    def spawn_from_branch(
        self,
        branch_id,
    ):
        r"""
        Retrieve the next set of models and place on a new branch.

        By checking the :class:`~qmla.tree.QMLATree`` associated with the `branch_id` used
        to call this method, call :meth:`GrowthRuleTree.next_layer`, which returns
        a set of models to place on a new branch, as well as which models therein
        to compare. These are passed to :meth:`new_branch`, constructing a new branch
        in the QMLA environment. The generated new branch then has all its models
        learned by calling :meth:`~qmla.QuantumModelLearningAgent.learn_models_on_given_branch`.
        :meth:`~qmla.GrowthRuleTree.next_layer` is in control of how to select the next set of models,
        usually either by calling the :class:`~qmla.growth_rules.GrowthRule`'s 
        :meth:`~qmla.growth_rules.GrowthRule.generate_models` or
        :meth:`~qmla.growth_rules.GrowthRule.tree_pruning` methods. 
        This allows the user to define how models are generated,
        given access to the comparisons of the previous branch, 
        or how the tree is pruned, e.g. by performing preliminary 
        parent/child branch champion comparisons.

        :param int branch_id: unique ID of the branch which has completed
        """

        model_list = self.branches[branch_id].rankings
        model_names = [
            self.model_name_id_map[mod_id]
            for mod_id in model_list
        ]

        new_models, pairs_to_compare = self.branches[branch_id].tree.next_layer(
            model_list=model_names, # can this be functionally replaced by info in branch_model_points?
            model_names_ids=self.model_name_id_map,
            called_by_branch=branch_id,
            branch_model_points=self.branches[branch_id].bayes_points, # can get from branch/tree
            evaluation_log_likelihoods=self.branches[branch_id].evaluation_log_likelihoods, # can get from branch/tree
            model_dict=self.model_lists, # is this used by any GR? TODO remove
        )

        self.log_print(
            [
                "After model generation for GR",
                self.branches[branch_id].growth_rule,
                "\nnew models:", new_models,
                "pairs to compare:", pairs_to_compare,
            ]
        )

        # Generate new QMLA level branch
        new_branch_id = self.new_branch(
            model_list=new_models,
            pairs_to_compare=pairs_to_compare,
            growth_rule=self.branches[branch_id].growth_rule,
            spawning_branch=branch_id,
        )

        # Learn models on the new branch
        self.learn_models_on_given_branch(
            new_branch_id,
            blocking=False,
        )

    def new_branch(
        self,
        model_list,
        pairs_to_compare='all',
        growth_rule=None,
        spawning_branch=0,
    ):
        r"""
        Add a set of models to a new QMLA branch.

        Branches have a unique id within QMLA, but belong to a single
        tree, where each tree corresponds to a single growth rule.

        :param list model_list: strings corresponding to models to
            place in the branch
        :param pairs_to_compare: set of model pairs to perform comparisons between.
            'all' (deafult) means  all models in `model_list` are set to compare.
            Otherwise a list of tuples of model IDs to compare
        :type pairs_to_compare: str or list
        :param str growth_rule: growth rule identifer;
            used to get the unique tree object corresponding to a growth rule,
            which is then used to host the branch.
        :param int spawning_branch: branch id which is the parent of the new branch.
        :return: branch id which uniquely identifies the new branch
            within the QMLA environment.
        """

        model_list = list(set(model_list))  # remove possible duplicates
        branch_id = int(self.branch_highest_id) + 1
        self.branch_highest_id = branch_id

        this_branch_models = {}
        model_id_list = []
        pre_computed_models = []
        for model in model_list:
            # add_model_to_database returns whether adding model was successful
            # if false, that's because it's already been computed
            add_model_info = self.add_model_to_database(
                model,
                branch_id=branch_id
            )
            already_computed = not(
                add_model_info['is_new_model']
            )
            model_id = add_model_info['model_id']
            this_branch_models[model_id] = model
            model_id_list.append(model_id)

            # register if new model
            if already_computed:
                pre_computed_models.append(model)
            self.log_print([
                'Model {} computed already: {} -> ID {}'.format(
                    model, already_computed, model_id,
                ),
            ])

        model_instances = {
            m: self.get_model_storage_instance_by_id(m)
            for m in list(this_branch_models.keys())
        }

        # Start new branch on corresponding growth rule tree
        if growth_rule is None:
            growth_rule = self.growth_rule_of_true_model
        growth_tree = self.trees[growth_rule]

        self.branches[branch_id] = growth_tree.new_branch_on_tree(
            branch_id=branch_id,
            models=this_branch_models,
            pairs_to_compare=pairs_to_compare,
            model_instances=model_instances,
            precomputed_models=pre_computed_models,
            spawning_branch=spawning_branch,
        )

        return branch_id

    def add_model_to_database(
        self,
        model,
        branch_id=-1,
        force_create_model=False
    ):
        r"""
        Considers adding a model to QMLA's database of models.

        Checks whether the nominated model is already present;
        if not generates a model instance and
        stores pertinent details in the model database.

        :param str model: name of model to consider
        :param float branch_id: branch id to associate this model with,
            if the model is new.
        :param bool force_create_model:
            True: add model even if the name is found already.
            False: (default) check if the model exists before adding
        :return dict add_model_output:
            `is_new_model` : bool, whether model is new (True) or has already been added (False)
            model_id: unique model ID for the model, whether new or existing
        """

        model_name = database_framework.alph(model) 
        self.log_print(["Trying to add model to DB:", model_name])

        # Add model if not yet considered or told to force create
        if (
            qmla.database_framework.consider_new_model(
                self.model_lists, model_name, self.model_database) == 'New'
            or force_create_model == True
        ):
            # create new model instance
            model_num_qubits = qmla.database_framework.get_num_qubits(
                model_name)
            model_id = self.highest_model_id + 1
            self.model_lists[model_num_qubits].append(model_name)

            self.log_print([
                "Model {} not previously considered -- adding with ID {}".format(
                    model_name, model_id
                )
            ])
            op = qmla.database_framework.Operator(
                name=model_name, undimensionalised_name=model_name
            )
            # generate model storage instance
            model_storage_instance = qmla.model_for_storage.ModelInstanceForStorage(
                model_name=model_name,
                model_id=int(model_id),
                model_terms_matrices=op.constituents_operators,
                true_oplist=self.true_model_constituent_operators,
                true_model_terms_params=self.true_param_list,
                qid=self.qmla_id,
                qmla_core_info_database=self.qmla_core_info_database,
                plot_probes=self.probes_for_plots,
                host_name=self.redis_host_name,
                port_number=self.redis_port_number,
                log_file=self.log_file
            )

            # add to the model database
            running_db_new_row = pd.Series({
                'model_id': int(model_id),
                'model_name': model_name,
                'branch_id': int(branch_id),
                'model_storage_instance': model_storage_instance,
            })
            num_rows = len(self.model_database)
            self.model_database.loc[num_rows] = running_db_new_row
            
            model_added = True
            if database_framework.alph(
                    model) == database_framework.alph(self.true_model_name):
                self.true_model_id = model_id
                self.true_model_considered = True
                self.true_model_branch = branch_id
                self.true_model_on_branhces = [branch_id]
                self.log_print(["True model has ID", model_id])
            self.highest_model_id = model_id
            self.model_name_id_map[model_id] = model_name
            self.model_count += 1
            self.models_branches[model_id] = int(branch_id)
        else:
            # do not create new model instance
            model_added = False
            self.log_print([
                "Model not added: {}".format(model_name)
            ])
            try:
                model_id = database_framework.model_id_from_name(
                    db=self.model_database,
                    name=model_name
                )
                self.log_print([
                    "Previously considered as model ", model_id
                ])
                if model_id == self.true_model_id:
                    self.true_model_on_branhces.append(model_id)
            except BaseException:
                self.log_print(
                    [
                        "Couldn't find model id for model:", model_name,
                        "model_names_ids:",
                        self.model_name_id_map
                    ]
                )
                raise

        add_model_output = {
            'is_new_model': model_added,
            'model_id': model_id,
        }
        return add_model_output

    def finalise_qmla(self):
        r"""
        Steps to end QMLA algorithm, such as storing analytics.

        """

        champ_model = self.get_model_storage_instance_by_id(
            self.champion_model_id)

        # Get metrics for all models tested
        for i in self.models_learned:
            # dict of all Bayes factors for each model considered.
            self.all_bayes_factors[i] = (
                self.get_model_storage_instance_by_id(i).model_bayes_factors
            )
            self.compute_model_f_score(i)

        self.bayes_factors_data()
        self.growth_class.growth_rule_specific_plots(
            save_directory = self.qmla_controls.plots_directory,
            qmla_id = self.qmla_controls.long_id
        )
        self.growth_class.growth_rule_finalise()
        self.get_statistical_metrics()

        # Prepare model/name maps
        self.model_id_to_name_map = {}
        for k in self.model_name_id_map:
            v = self.model_name_id_map[k]
            self.model_id_to_name_map[v] = k


    def bayes_factors_data(self):
        self.bayes_factors_df = pd.DataFrame(
            columns=[
                'model_a', 'id_a', 'f_score_a',
                'model_b', 'id_b', 'f_score_b',
                'bayes_factor', 'log10_bayes_factor'
                ]
        )

        for m in self.models_learned:
            mod = self.get_model_storage_instance_by_id(m)
            mod_name_a = mod.model_name
            mod_id_a = int(mod.model_id)
            f_score_a = qmla.utilities.round_nearest(
                self.model_f_scores[mod_id_a], 0.05)

            bayes_factors = mod.model_bayes_factors
            for b in bayes_factors:
                mod_name_b = self.model_name_id_map[b]
                mod_id_b = int(b)
                f_score_b = qmla.utilities.round_nearest(
                    self.model_f_scores[mod_id_b], 0.05)
                
                for bf in bayes_factors[b]:
                    d = pd.Series(
                        {      
                            'model_a' : mod_name_a,
                            'id_a' : mod_id_a,
                            'f_score_a' : f_score_a,
                            'model_b' : mod_name_b, 
                            'id_b' : mod_id_b, 
                            'f_score_b' : f_score_b, 
                            'bayes_factor' : bf,
                            'log10_bayes_factor' : np.round(np.log10(bf), 1)
                        }
                    )
                    new_idx = len(self.bayes_factors_df)
                    self.bayes_factors_df.loc[new_idx] = d
        self.plot_bayes_factors()


    def get_results_dict(
        self,
        model_id=None
    ):
        r"""
        Store the useful information of a given model, usually the champion.

        :param int model_id: unique ID of the model whose information to store
        :return dict results_dict: data which will be stored in the results_{ID}.p
            file following QMLA's completion.
        """

        if model_id is None:
            model_id = self.champion_model_id

        mod = self.get_model_storage_instance_by_id(model_id)
        model_name = mod.model_name

        # Get expectation values of this model
        n_qubits = database_framework.get_num_qubits(model_name)
        if n_qubits > 3:
            expec_val_plot_times = self.times_to_plot_reduced_set
        else:
            expec_val_plot_times = self.times_to_plot

        mod.compute_expectation_values(
            times=expec_val_plot_times,
        )

        # Perform final steps in GrowthRule
        # self.growth_class.growth_rule_finalise()  # TODO put in main QMLA/QHL method

        # Evaluations of all models in this instance
        model_evaluation_log_likelihoods = {
            mod_id: self.get_model_storage_instance_by_id(mod_id).evaluation_log_likelihood
            for mod_id in self.models_learned
        }
        model_evaluation_median_likelihoods = {
            mod_id: self.get_model_storage_instance_by_id(mod_id).evaluation_median_likelihood
            for mod_id in self.models_learned
        }

        # Compare this model to the true model (only meaningful for simulated
        # cases)
        correct_model = misfit = underfit = overfit = 0
        num_params_champ_model = database_framework.Operator(
            model_name).num_constituents

        if model_name == self.true_model_name:
            correct_model = 1
        elif (
            num_params_champ_model == self.true_model_num_params
            and
            model_name != self.true_model_name
        ):
            misfit = 1
        elif num_params_champ_model > self.true_model_num_params:
            overfit = 1
        elif num_params_champ_model < self.true_model_num_params:
            underfit = 1
        num_params_difference = self.true_model_num_params - num_params_champ_model

        # Summarise the results of this model and instance in a dictionary
        time_taken = time.time() - self._start_time
        results_dict = {
            # details about QMLA instance:
            'QID': self.qmla_id,
            'NumParticles': self.num_particles,
            'NumExperiments': mod.num_experiments,
            'NumBayesTimes': self.num_experiments_for_bayes_updates,
            'ResampleThreshold': self.qinfer_resample_threshold,
            'ResamplerA': self.qinfer_resampler_a,
            'PHGPrefactor': self.qinfer_PGH_heuristic_factor,
            'ConfigLatex': self.latex_config,
            'Heuristic': mod.model_heuristic_class,
            'Time': time_taken,
            'Host': self.redis_host_name,
            'Port': self.redis_port_number,
            # details about true model:
            'TrueModel': self.true_model_name,
            'TrueModelConsidered': self.true_model_considered,
            'TrueModelFound': self.true_model_found,
            'TrueModelBranch': self.true_model_branch,
            'Truemodel_id': self.true_model_id,
            'TrueModelConstituentTerms': self.true_model_constituent_terms_latex,
            # details about this model
            'ChampID': model_id,
            'ChampLatex': mod.model_name_latex,
            'ConstituentTerms': mod.constituents_terms_latex,
            'LearnedHamiltonian': mod.learned_hamiltonian,
            'GrowthGenerator': mod.growth_rule_of_this_model,
            'NameAlphabetical': database_framework.alph(mod.model_name),
            'LearnedParameters': mod.qhl_final_param_estimates,
            'FinalSigmas': mod.qhl_final_param_uncertainties,
            'ExpectationValues': mod.expectation_values,
            'Trackplot_parameter_estimates': mod.track_parameter_estimates,
            'TrackVolume': mod.volume_by_epoch,
            'TrackTimesLearned': mod.times_learned_over,
            'QuadraticLosses': mod.quadratic_losses_record,
            'FinalRSquared': mod.r_squared(
                times=expec_val_plot_times,
            ),
            'Fscore': self.model_f_scores[model_id],
            'Precision': self.model_precisions[model_id],
            'Sensitivity': self.model_sensitivities[model_id],
            'PValue': mod.p_value,
            # comparison to true model (for simulated cases)
            'NumParamDifference': num_params_difference,
            'Underfit': underfit,
            'Overfit': overfit,
            'Misfit': misfit,
            'CorrectModel': correct_model,
            # about QMLA's learning procedure:
            'NumModels': len(self.models_learned),
            'StatisticalMetrics': self.generational_statistical_metrics,
            'GenerationalFscore': self.generational_f_score,
            'GenerationalLogLikelihoods': self.generational_log_likelihoods,
            'ModelEvaluationLogLikelihoods': model_evaluation_log_likelihoods,
            'ModelEvaluationMedianLikelihoods': model_evaluation_median_likelihoods,
            'AllModelFScores': self.model_f_scores,
            # data stored during GrowthRule.growth_rule_finalise():
            'GrowthRuleStorageData': self.growth_class.growth_rule_specific_data_to_store,
        }

        self.storage = qmla.utilities.StorageUnit()
        self.storage.qmla_id = self.qmla_id
        self.storage.bayes_factors_df = self.bayes_factors_df
        self.storage.growth_rule_storage = self.growth_class.storage        

        return results_dict

    def check_champion_reducibility(
        self,
    ):
        r"""
        Potentially remove negligible terms from the champion model.

        Consider whether the champion model has some terms whose parameters
        were found to be negligible (either within one standard
        deviation from 0, or very close to zero as determined by the growth rule's
        `learned_param_limit_for_negligibility` attribute).
        Construct a new model which is the same as the champion, less those negligible
        terms, named the reduced champion. The data of the champion model is inherited
        by the reduced candidate model, i.e. its parameter estimates, as well as
        its history of parameter learning for those which are not negligible.
        A new `normalization_record` is started, which is used in the comparison between
        the champion and the reduced champion.
        Compare the champion with the reduced champion; if the reduced champion
        is heavily favoured, directly select it as the global champion.
        This method is triggered if the growth rule's `check_champion_reducibility`
        attribute is set to True.

        """
        import qinfer

        champ_mod = self.get_model_storage_instance_by_id(
            self.global_champion_id
        )

        self.log_print(
            [
                "Checking reducibility of champ model:",
                self.global_champion_name,
                "\nParams:\n", champ_mod.qhl_final_param_estimates,
                "\nSigmas:\n", champ_mod.qhl_final_param_uncertainties
            ]
        )

        params = list(champ_mod.qhl_final_param_estimates.keys())
        to_remove = []
        removed_params = {}
        idx = 0
        for p in params:
            # if champ_mod.qhl_final_param_uncertainties[p] > champ_mod.qhl_final_param_estimates[p]:
            #     to_remove.append(p)
            #     removed_params[p] = np.round(
            #         champ_mod.qhl_final_param_estimates[p],
            #         2
            #     )

            if (
                np.abs(champ_mod.qhl_final_param_estimates[p])
                < self.growth_class.learned_param_limit_for_negligibility
            ):
                to_remove.append(p)
                removed_params[p] = np.round(
                    champ_mod.qhl_final_param_estimates[p], 2
                )

        if len(to_remove) >= len(params):
            self.log_print([
                "Attempted champion reduction failed due to",
                "all parameters found as neglibible.",
                "Check method of determining negligibility.",
                "(By default, parameter removed if sigma of that",
                "parameters final posterior > parameter.",
                "i.e. 0 within 1 sigma of distriubtion"
            ])
            return
        if len(to_remove) > 0:
            new_model_terms = list(
                set(params) - set(to_remove)
            )
            new_mod = '+'.join(new_model_terms)
            new_mod = database_framework.alph(new_mod)

            self.log_print([
                "Some neglibible parameters found:", removed_params,
                "\nReduced champion model suggested:", new_mod
            ])

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

            # get champion leared info
            reduced_champion_info = pickle.loads(
                self.redis_databases['learned_models_info_db'].get(
                    str(self.champion_model_id))
            )

            reduced_params = {}
            reduced_sigmas = {}
            for term in reduced_mod_terms:
                reduced_params[term] = champ_mod.qhl_final_param_estimates[term]
                reduced_sigmas[term] = champ_mod.qhl_final_param_uncertainties[term]

            learned_params = [reduced_params[t] for t in reduced_mod_terms]
            sigmas = np.array([reduced_sigmas[t] for t in reduced_mod_terms])
            final_params = np.array(list(zip(learned_params, sigmas)))

            new_cov_mat = np.diag(
                sigmas**2
            )

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
            # do not inherit normalization_record and times from original
            # champion
            reduced_champion_info['normalization_record'] = []
            reduced_champion_info['times'] = []

            compressed_reduced_champ_info = pickle.dumps(
                reduced_champion_info,
                protocol=4
            )

            # TODO generate new model for champion
            # - scratch normalization record;
            # - learn according to MPGH for both champion
            #   and suggested reduced champion,
            #   then take BF based on that
            self.redis_databases['learned_models_info_db'].set(
                str(float(reduced_mod_id)),
                compressed_reduced_champ_info
            )

            self.get_model_storage_instance_by_id(
                reduced_mod_id).model_update_learned_values()

            bayes_factor = self.compare_model_pair(
                model_a_id=int(self.champion_model_id),
                model_b_id=int(reduced_mod_id),
                wait_on_result=True
            )
            self.log_print([
                "BF b/w champ and reduced champ models:", bayes_factor                
            ])

            if (
                bayes_factor
                < (1.0 / self.growth_class.reduce_champ_bayes_factor_threshold)
            ):
                # overwrite champ id etc
                self.log_print([
                    "Replacing champion model ({}) with reduced champion model ({} - {})".format(
                        self.champion_model_id,
                        reduced_mod_id,
                        new_mod
                    ),
                    "\n i.e. removing negligible parameter terms:\n{}".format(
                        removed_params
                    )
                ])
                original_champ_id = self.champion_model_id
                self.champion_model_id = reduced_mod_id
                self.global_champion = new_mod
                # inherits BF of champion from which it derived (only really
                # used for plotting)
                new_champ = self.get_model_storage_instance_by_id(
                    self.champion_model_id
                )
                new_champ.model_bayes_factors = (
                    self.get_model_storage_instance_by_id(
                        original_champ_id).model_bayes_factors
                )
                new_champ.times_learned_over = champ_mod.times_learned_over
                self.models_learned.append(reduced_mod_id)

        else:
            self.log_print([
                "Parameters non-negligible; not replacing champion model."
            ])

    def compare_nominated_champions(self):
        r"""
        Compare the champions of all growth rule trees.

        Get the champions (usually one, but in general can be multiple)
        from each tree, where each tree is unique to a growth rule.
        Place the champions on a branch together and perform all-versus-all
        comparisons. The champion of that branch is deemed the global champion.

        """

        tree_champions = []
        for tree in self.trees.values():
            # extend in case multiple models nominated by tree
            tree_champions.extend(tree.nominate_champions())

        # Place tree champions on new QMLA branch, not tied to a growth rule
        global_champ_branch_id = self.new_branch(
            model_list=tree_champions
        )
        global_champ_branch = self.branches[
            global_champ_branch_id
        ]

        # Compare models (using this fnc so we can wait_on_result)
        self.compare_model_set(
            pair_list=global_champ_branch.pairs_to_compare,
            wait_on_result=True,
        )
        champ_id = self.process_comparisons_within_branch(
            branch_id=global_champ_branch_id
        )

        # Assign champion of set to be global champion
        self.global_champion_id = champ_id
        self.global_champion_model = self.get_model_storage_instance_by_id(
            self.global_champion_id
        )
        self.global_champion_name = self.global_champion_model.model_name
        self.log_print([
            "Global champion branch points:", global_champ_branch.bayes_points,
            "\nGlobal champion ID:", champ_id,
            "\nGlobal champion:", self.global_champion_name
        ])

    ##########
    # Section: Run available algorithms (QMLA, QHL or QHL with multiple models)
    ##########

    def run_quantum_hamiltonian_learning(
        self,
    ):
        r"""
        Run Quantum Hamiltonian Learning algorithm .

        The `true_model` of the :class:`~qmla.growth_rules.GrowthRule` is used to generate
        true data (in simulation) and have its parameters learned.

        """

        qhl_branch = self.new_branch(
            growth_rule=self.growth_rule_of_true_model,
            model_list=[self.true_model_name]
        )

        mod_to_learn = self.true_model_name
        self.log_print(
            [
                "QHL for true model:", mod_to_learn,
            ]
        )

        self.learn_model(
            model_name=mod_to_learn,
            branch_id=qhl_branch,
            blocking=True
        )

        mod_id = database_framework.model_id_from_name(
            db=self.model_database,
            name=mod_to_learn
        )

        # these don't really matter for QHL,
        # but are used in plots etc:
        self.true_model_id = mod_id
        self.champion_model_id = mod_id
        self.true_model_found = True
        self.true_model_considered = True
        self.log_print([
            "Learned model {}: {}".format(
                mod_id,
                mod_to_learn
            )
        ])
        self._update_database_model_info()
        self.compute_model_f_score(
            model_id=mod_id
        )
        self.growth_class.growth_rule_finalise()
        self.get_statistical_metrics()

    def run_quantum_hamiltonian_learning_multiple_models(
        self,
        model_names=None
    ):
        r"""
        Run Quantum Hamiltonian Learning algorithm with multiple simulated models.

        Numerous Hamiltonian models attempt to learn the dynamics of the true model.
        The underlying model is set in the :class:`~qmla.growth_rules.GrowthRule`'s `true_model` attribute.

        :param list model_names:
            list of strings of model names to learn the parameterisations of.
            None: taken from :class:`~qmla.growth_rules.GrowthRule` `qhl_models`.
        """

        # Choose models to perform QHL on
        if model_names is None:
            model_names = self.growth_class.qhl_models

        # Place models on a branch
        branch_id = self.new_branch(
            growth_rule=self.growth_rule_of_true_model,
            model_list=model_names
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
                'QHL for multiple models:', model_names,
            ]
        )
        learned_models_ids = self.redis_databases['learned_models_ids']

        # learn models
        for mod_name in model_names:
            mod_id = database_framework.model_id_from_name(
                db=self.model_database,
                name=mod_name
            )
            learned_models_ids.set(
                str(mod_id), 0
            )
            self.learn_model(
                model_name=mod_name,
                branch_id=branch_id,
                # use_rq=self.use_rq,
                blocking=False
            )

        running_models = learned_models_ids.keys()
        self.log_print(
            [
                'Running Models:', running_models,
            ]
        )
        for k in running_models:
            # waiting on all models to finish,
            # so we can wait on them in order.
            while int(learned_models_ids.get(k)) != 1:
                sleep(self.sleep_duration)
                self._inspect_remote_job_crashes()

        # Learning finished
        self.log_print(
            [
                'Finished learning for all:', running_models,
            ]
        )

        # Tidy up: store learned info, analyse, etc.
        for mod_name in model_names:
            mod_id = database_framework.model_id_from_name(
                db=self.model_database, name=mod_name
            )
            mod = self.get_model_storage_instance_by_id(mod_id)
            mod.model_update_learned_values()
            self.compute_model_f_score(model_id=mod_id)
        self.growth_class.growth_rule_finalise()
        self.get_statistical_metrics()
        self.model_id_to_name_map = {}
        for k in self.model_name_id_map:
            v = self.model_name_id_map[k]
            self.model_id_to_name_map[v] = k
        for k in self.timings:
            self.log_print([
                "QMLA Timing - {}: {}".format(k, np.round(self.timings[k], 2))
            ])

    def run_complete_qmla(
        self,
    ):
        r"""
        Run complete Quantum Model Learning Agent algorithm.

        Each :class:`~qmla.growth_rules.GrowthRule` is assigned a :class:`~qmla.tree.QMLATree`,
        which manages the growth rule. When new models are spawned by a growth rule, 
        they are placed on a :class:`~qmla.tree.BranchQMLA` of the corresponding tree.
        Models are learned/compared/spawned iteratively in 
        :meth:`learn_models_until_trees_complete`, until all
        trees declare that their growth rule has completed.
        Growth rules are complete when they have nominated one or more champions,
        which can follow spawning/pruning stages as required by the growth rule.
        Nominated champions are then compared with :meth:`compare_nominated_champions`,
        resulting in a single global champion selected.
        Some analysis then takes place, including possibly reducing the
        selected global champion if it is found that some of its terms are not impactful.

        """

        # Set up one tree per growth rule
        for tree in list(self.trees.values()):
            starting_models = tree.get_initial_models()
            self.log_print([
                "First branch for {} has starting models: {}".format(
                    tree.growth_rule, starting_models
                ),
            ])
            self.new_branch(
                model_list=starting_models,
                growth_rule=tree.growth_rule
            )

        # Iteratively learn models, compute bayes factors, spawn new models
        self.learn_models_until_trees_complete()
        self.log_print([
            "Growth rule trees completed."
        ])

        # Choose champion by comparing nominated champions of all trees.
        self.compare_nominated_champions()
        self.champion_model_id = self._get_model_data_by_field(
            name=self.global_champion_name,
            field='model_id'
        )
        self.log_print(["Champion selected."])

        # Internal analysis
        try:
            if self.global_champion_id == self.true_model_id:
                self.true_model_found = True
            else:
                self.true_model_found = False
        except BaseException:
            self.true_model_found = False
        self._update_database_model_info()
        if self.true_model_found:
            self.log_print([
                "True model found: {}".format(
                    database_framework.alph(self.true_model_name)
                )
            ])
        self.log_print([
            "True model considered: {}. on branch {}.".format(
                self.true_model_considered,
                self.true_model_branch
            )
        ])

        # Consider reducing champion if negligible parameters found
        if self.growth_class.check_champion_reducibility:
            self.check_champion_reducibility()

        # Tidy up and finish QMLA.
        self.finalise_qmla()

        self.log_print([
            "\nFinal winner:", self.global_champion_name,
            "has F-score ", np.round(self.model_f_scores[self.champion_model_id], 2)
        ])

    ##########
    # Section: Utilities
    ##########

    def log_print(self, to_print_list):
        r"""Wrapper for :func:`~qmla.print_to_log`"""
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier='QMLA {}'.format(self.qmla_id)

        )

    def get_model_storage_instance_by_id(self, model_id):
        r"""
        Get the unique :class:`~qmla.ModelInstanceForLearning` for the given model_id.

        :param int model_id: unique ID of desired model
        :return: storage class of the model
        :rtype: :class:`~qmla.ModelInstanceForLearning`

        """
        idx = self.model_database.loc[self.model_database['model_id']
                                      == model_id].index[0]
        model_instance = self.model_database.loc[idx]["model_storage_instance"]
        return model_instance

    def _update_database_model_info(self):
        r"""
        Calls :meth:`~qmla.ModelForStorage.model_update_learned_values` for all models learned in this instance.
        """

        self.log_print([
            "Updating info for all learned models"
        ])
        for mod_id in self.models_learned:
            try:
                mod = self.get_model_storage_instance_by_id(mod_id)
                mod.model_update_learned_values()
            except BaseException:
                pass

    def _inspect_remote_job_crashes(self):
        r"""Check if any job on redis queue has failed."""
        self.call_counter['job_crashes'] += 1
        t_init = time.time()
        if self.redis_databases['any_job_failed']['Status'] == b'1':
            # TODO better way to detect errors?
            self.log_print(
                [
                    "Failure on remote node. Terminating QMD."
                ]
            )
            raise NameError('Remote QML Failure')
        self.timings['inspect_job_crashes'] += time.time() - t_init

    def _delete_unpicklable_attributes(self):
        r"""Remove elements of QMLA which cannot be pickled, which cause errors if retained."""

        del self.redis_conn
        del self.redis_databases
        del self.write_log_file

    def _get_model_data_by_field(self, name, field):
        r"""
        Get information from the model database by model name.

        :param str name: model name to get data of
        :param str field: field name to get data corresponding to model
        """
        return database_framework.pull_field(
            self.model_database,
            name,
            field
        )

    ##########
    # Section: Analysis/plotting methods
    ##########
    def compute_model_f_score(
        self,
        model_id,
        beta=1  # beta=1 for F1-score. Beta is relative importance of sensitivity to precision
    ):
        r"""
        Compte and store f-score of given model.

        :param int model_id: model ID to compute f-score of
        :param float beta: for generalised F_beta score. (default) 1 for F1 score.
        :return float f_score: F-score of given model.

        """

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
            f_score = (
                (1 + beta**2) * (
                    (precision * sensitivity)
                    / (beta**2 * precision + sensitivity)
                )
            )
        except BaseException:
            # both precision and sensitivity=0 as true_positives=0
            f_score = 0

        self.model_f_scores[model_id] = f_score
        self.model_precisions[model_id] = precision
        self.model_sensitivities[model_id] = sensitivity
        return f_score

    def get_statistical_metrics(
        self,
        save_to_file=None
    ):
        r"""
        Compute, store and plot various statistical metrics of all studied models.

        :param str save_to_file: path to save the resultant figure in.
        """
        generations = sorted(set(self.branches.keys()))
        # generations = sorted()
        # [
        #     b for b in self.branches
        #     if not self.branches[b].prune_branch
        # ]
        self.log_print(
            [
                "[get_statistical_metrics",
                "generations: ", generations
            ]
        )

        generational_sensitivity = {
            b: []
            for b in generations
        }
        generational_f_score = {
            b: []
            for b in generations
        }
        generational_precision = {
            b: []
            for b in generations
        }
        self.generational_log_likelihoods = {
            b: []
            for b in generations
        }

        for b in generations:
            models_this_branch = sorted(self.branches[b].resident_model_ids)
            self.log_print(
                [
                    "Adding models to generational measures for Generation {}:{}".format(
                        b,
                        models_this_branch
                    )
                ]
            )
            for m in models_this_branch:
                generational_sensitivity[b].append(self.model_sensitivities[m])
                generational_precision[b].append(self.model_precisions[m])
                generational_f_score[b].append(self.model_f_scores[m])
                self.generational_log_likelihoods[b].append(
                    self.get_model_storage_instance_by_id(
                        m).evaluation_log_likelihood
                )

        include_plots = [
            {'name': 'F-score', 'data': generational_f_score, 'colour': 'red'},
            {'name': 'Precision', 'data': generational_precision, 'colour': 'blue'},
            {'name': 'Sensitivity',
             'data': generational_sensitivity,
             'colour': 'green'},
        ]
        self.generational_f_score = generational_f_score
        self.generational_statistical_metrics = {
            k['name']: k['data']
            for k in include_plots
        }
        self.alt_generational_statistical_metrics = {
            b: {
                'Precision': generational_precision[b],
                'Sensitivity': generational_sensitivity[b],
                'F-score': generational_f_score[b]
            }
            for b in generations
        }

        fig = plt.figure(
            figsize=(15, 5),
            tight_layout=True
        )
        gs = GridSpec(
            nrows=1,
            ncols=len(include_plots),
        )
        plot_col = 0

        for plotting_data in include_plots:

            ax = fig.add_subplot(gs[0, plot_col])
            data = plotting_data['data']
            ax.plot(
                generations,
                [np.median(data[b]) for b in generations],
                label="{} median".format(plotting_data['name']),
                color=plotting_data['colour'],
                marker='o'
            )
            ax.fill_between(
                generations,
                [np.min(data[b]) for b in generations],
                [np.max(data[b]) for b in generations],
                alpha=0.2,
                label="{} min/max".format(plotting_data['name']),
                color=plotting_data['colour']
            )
            ax.set_ylabel("{}".format(plotting_data['name']))
            ax.set_xlabel("Generation")
            ax.legend()
            ax.set_ylim(0, 1)
            plot_col += 1

        self.log_print(["getting statistical metrics complete"])
        if save_to_file is not None:
            plt.savefig(save_to_file)


    def plot_bayes_factors(
        self, 
    ):
        # Plot Bayes factors of this instance
        bayes_factor_by_id = pd.pivot_table(
            self.bayes_factors_df, 
            values='log10_bayes_factor', 
            index=['id_a'], 
            columns=['id_b'],
            aggfunc=np.median
        )
        mask = np.tri(bayes_factor_by_id.shape[0], k=-1).T
        plt.clf()
        s = sns.heatmap(
            bayes_factor_by_id,
            cmap='RdYlGn',
            mask=mask,
            annot=False
        )   
        s.get_figure().savefig(
            os.path.join(
                self.qmla_controls.plots_directory,
                'bayes_factors_{}.png'.format(self.qmla_controls.long_id)
            )
        )                         

        # # Heat map BF against F(A)/F(B)
        fig = qmla.analysis.bayes_factor_f_score_heatmap(bayes_factors_df = self.bayes_factors_df)
        fig.savefig(
            os.path.join(
                self.qmla_controls.plots_directory, "bayes_factors_by_f_score_{}".format(self.qmla_controls.long_id)
            )
        )
        

    def plot_branch_champs_quadratic_losses(
        self,
        save_to_file=None,
    ):
        r"""Wrapper for :func:`~qmla.analysis.plot_quadratic_loss`."""
        qmla.analysis.plot_quadratic_loss(
            qmd=self,
            champs_or_all='champs',
            save_to_file=save_to_file
        )

    def plot_branch_champs_volumes(
        self, model_id_list=None, branch_champions=False,
        branch_id=None, save_to_file=None
    ):
        r"""
        Plot the volume of each branch champion within this instance.

        :param list model_id_list: list of model IDs to plot volumes of,
            if None plot branch champions
        :param bool branch_champions: force plot only branch champions' volumes
        :param int branch_id: if provided, plot the volumes of all models within
            that branch
        :param str save_to_file: path at which to store the resultant figure.
        """

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
        r"""
        *deprecated* Store the pairwise comparisons computed during this instance.
        :func:`~qmla.analysis.model_bayes_factorsCSV` removed and is needed 
        TODO if wanted, find in old github commits and reimplement.

        Wrapper for :func:`~qmla.analysis.model_bayes_factorsCSV`.
        """
        qmla.analysis.model_bayes_factorsCSV(
            self, save_to_file, names_ids=names_ids)

    def store_bayes_factors_to_shared_csv(self, bayes_csv):
        r"""
        Store the pairwise comparisons computed during this instance in a CSV shared by all concurrent instances.
        """
        qmla.analysis.update_shared_bayes_factor_csv(self, bayes_csv)

    def plot_parameter_learning_single_model(
        self,
        model_id=0,
        true_model=False,
        save_to_file=None
    ):
        r"""
        Plot parameter estimates vs experiment number for a single model.

        Wrapper for :func:`~qmla.analysis.plot_parameter_estimates`
        :param bool true_model: whether to force only plotting the true
            model's parameter estimeates
        """
        if true_model:
            model_id = database_framework.model_id_from_name(
                db=self.model_database, name=self.true_model_name)

        qmla.analysis.plot_parameter_estimates(qmd=self,
                                               model_id=model_id,
                                               #    use_experimental_data=self.use_experimental_data,
                                               save_to_file=save_to_file
                                               )

    def plot_branch_champions_dynamics(
        self,
        all_models=False,
        model_ids=None,
        save_to_file=None,
    ):
        r"""
        Plot reproduced dynamics of all branch champions

        :param bool all_models: whether to plot all models in the instance
        :param list model_ids: list of model IDs to plot dynamics of
        :param str save_to_file: path at which to save the resultant figure
        """

        include_params = False
        include_bayes_factors = False
        if all_models:
            model_ids = list(sorted(self.model_name_id_map.keys()))
        elif self.qhl_mode:
            model_ids = [self.true_model_id]
            include_params = True
        elif self.qhl_mode_multiple_models:
            model_ids = list(self.qhl_mode_multiple_models_model_ids)
        elif self.growth_class.tree_completed_initially:
            model_ids = list(self.models_learned)
            include_bayes_factors = True
            include_params = True
        elif model_ids is None:
            model_ids = [
                self.branches[b].champion_id
                for b in self.branches
            ]
            include_bayes_factors = True
        self.log_print([
            "Plotting dynamics of models:",
            model_ids
        ])
        path_to_save = os.path.join(self.qmla_controls.plots_directory, 'dynamics.png')
        try:
            include_times_learned = False
            include_params = False
            qmla.analysis.plot_learned_models_dynamics(
                qmd=self,
                include_bayes_factors=include_bayes_factors,
                include_times_learned=include_times_learned,
                include_param_estimates=include_params,
                model_ids=model_ids,
                save_to_file=path_to_save,
            )
        except:
            self.log_print(["Failed to plot dynamics"])
            raise

    def plot_volume_after_qhl(self,
                              model_id=None,
                              true_model=True,
                              show_resamplings=True,
                              save_to_file=None
                              ):
        r"""
        Plot volume vs experiment number of a single model.
        Wrapper for :func:`~qmla.analysis.plot_volume_after_qhl`
        """
        qmla.analysis.plot_volume_after_qhl(
            qmd=self,
            model_id=model_id,
            true_model=true_model,
            show_resamplings=show_resamplings,
            save_to_file=save_to_file
        )

    def plot_GrowthRuleTree(
        self,
        modlist=None,
        only_adjacent_branches=True,
        save_to_file=None
    ):
        r"""Wrappter for :func:`~qmla.analysis.plot_qmla_single_instance_tree`"""
        qmla.analysis.plot_qmla_single_instance_tree(
            self,
            modlist=modlist,
            only_adjacent_branches=only_adjacent_branches,
            save_to_file=save_to_file
        )

    def plot_qmla_radar_scores(self, modlist=None, save_to_file=None):
        r"""*deprecated* Wrapper for :func:`~qmla.analysis.plotRadar`."""
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
        r"""
        Plot R^2 vs experiment number for given model list.
        """
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

    def plot_one_qubit_probes_bloch_sphere(self, save=False):
        r"""Show all one qubit probes on Bloch sphere."""
        import qutip as qt
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
        
        if save:
            bloch.save(
                os.path.join(
                    self.qmla_controls.plots_directory, 
                    'probes_bloch_sphere.png'
                )
            )
        else:
            bloch.show()

