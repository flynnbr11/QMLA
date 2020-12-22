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

try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure

# QMLA functionality
import qmla.analysis
import qmla.construct_models as construct_models
import qmla.get_exploration_strategy as get_exploration_strategy
import qmla.redis_settings as rds
import qmla.model_for_storage
from qmla.remote_bayes_factor import remote_bayes_factor_calculation
from qmla.remote_model_learning import remote_learn_model_parameters
import qmla.exploration_tree
import qmla.utilities

pickle.HIGHEST_PROTOCOL = 4
plt.switch_backend('agg')

__all__ = [
    'QuantumModelLearningAgent'
]


class QuantumModelLearningAgent():
    r"""
    QMLA manager class.

    Controls the infrastructure which determines which models are learned and compared.
    By interpreting user defined :class:`~qmla.exploration_strategies.ExplorationStrategy`,
    grows :class:`~qmla.ExplorationTree` objects which hold numerous models
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

        self._start_time = time.time()  # to measure run-time

        # Configure this QMLA instance
        if qmla_controls is None:
            self.qmla_controls = qmla.controls_qmla.parse_cmd_line_args(
                args={}
            )
        else:
            self.qmla_controls = qmla_controls
        self.exploration_class = self.qmla_controls.exploration_class
        

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

        # Set up infrastructure related to exploration strategies and tree management
        self._setup_tree_and_exploration_strategies()

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
        self.log_print(["\nwithin QMLA, ES's qmla id is {}. True model={}".format(self.exploration_class.qmla_id, self.exploration_class.true_model)])
        self.qhl_mode = self.qmla_controls.qhl_mode
        self.qhl_mode_multiple_models = self.qmla_controls.qhl_mode_multiple_models
        self.latex_name_map_file_path = self.qmla_controls.latex_mapping_file
        self.results_directory = self.qmla_controls.results_directory
        self.debug_mode = self.qmla_controls.debug_mode
        self.plot_level = self.qmla_controls.plot_level

        # Databases for storing learning/comparison data
        self.redis_databases = rds.get_redis_databases_by_qmla_id(
            self.redis_host_name,
            self.redis_port_number,
            self.qmla_id,
        )
        self.redis_databases['any_job_failed'].set('Status', 0)

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
        self.sleep_duration = 2

    def _true_model_definition(self):
        r""" Information related to true (target) model."""

        self.true_model_name = construct_models.alph(
            self.qmla_controls.true_model_name)
        self.true_model_dimension = construct_models.get_num_qubits(
            self.true_model_name)
        self.true_model_constituent_operators = self.qmla_controls.true_model_terms_matrices
        self.true_model_constituent_terms_latex = [
            self.exploration_class.latex_name(term)
            for term in
            qmla.construct_models.get_constituent_names_from_name(
                self.true_model_name)
        ]
        self.true_model_num_params = self.qmla_controls.true_model_class.num_constituents
        self.true_param_list = self.exploration_class.true_params_list
        self.true_param_dict = self.exploration_class.true_params_dict
        self.true_model_branch = -1  # overwrite if true model is added to database
        self.true_model_considered = False
        self.true_model_found = False
        self.true_model_id = -1
        self.true_model_on_branhces = []
        self.true_model_hamiltonian = self.exploration_class.true_hamiltonian
        self.log_print([
            "True model:", self.true_model_name
        ])

    def _setup_tree_and_exploration_strategies(
        self,
    ):
        r""" Set up infrastructure."""

        self.model_database = pd.DataFrame(
            {
                'model_id': [],
                'model_name': [],
                'latex_name': [],
                'branch_id': [],
                'f_score': [],
                'model_storage_instance': [],
                'branches_present_on' : [], 
                'terms' : [],
                'latex_terms' : []
            }
        )
        self.model_lists = {
            # assumes maxmium 13 qubit-models considered
            # to be checked when checking model_lists
            # TODO generalise to max dim of Exploration Strategy
            j: []
            for j in range(1, 13)
        }
        self.all_bayes_factors = {}
        self.bayes_factor_pair_computed = []

        # Exploration Strategy setup
        self.exploration_strategy_of_true_model = self.qmla_controls.exploration_rules
        self.unique_exploration_strategy_instances = self.qmla_controls.unique_exploration_strategy_instances

        # Keep track of models/branches
        self.model_count = 0
        self.highest_model_id = 0  # so first created model gets model_id=0
        self.models_branches = {}
        self.branch_highest_id = 0
        self.model_name_id_map = {}
        self.ghost_branches = {}

        # Tree object for each exploration strategy
        self.trees = {
            gen: qmla.exploration_tree.ExplorationTree(
                exploration_class=self.unique_exploration_strategy_instances[gen]
            )
            for gen in self.unique_exploration_strategy_instances
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
        # self.fraction_experiments_for_bf = self.exploration_class.fraction_experiments_for_bf
        self.num_experiments_for_bayes_updates = self.num_experiments # TODO remove


        self.bayes_threshold_lower = 1
        self.bayes_threshold_upper = 100 # TODO get from ES


        # Analysis infrastructure
        self.model_f_scores = {}
        self.model_precisions = {}
        self.model_sensitivities = {}
        self.bayes_factors_df = pd.DataFrame()

        # Get probes used for learning
        self.exploration_class.generate_probes(
            # noise_level=self.exploration_class.probe_noise_level,
            # minimum_tolerable_noise=0.0,
            # tell it the max number of qubits required by any ES under consideration
            probe_maximum_number_qubits = max(
                [gr.max_num_probe_qubits for gr in self.qmla_controls.unique_exploration_strategy_instances.values()]
            )
        )
        self.probes_system = self.exploration_class.probes_system
        self.probes_simulator = self.exploration_class.probes_simulator
        self.probe_number = self.exploration_class.num_probes
        sim_probe_keys = list(self.probes_simulator.keys())
        self.log_print(["Simulator probe keys (len {}):{}".format(len(sim_probe_keys), sim_probe_keys) ])

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
            self.log_print([
                "Could not load plot probes from {}".format(
                    self.probes_plot_file
                )
            ])

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
        # writeable file object to use for logging:
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

        # Decide if reallocating resources based on true ES.
        if self.exploration_class.reallocate_resources:
            base_num_qubits = 3
            base_num_terms = 3
            for op in self.exploration_class.initial_models:
                if construct_models.get_num_qubits(op) < base_num_qubits:
                    base_num_qubits = construct_models.get_num_qubits(op)
                num_terms = len(
                    construct_models.get_constituent_names_from_name(op))
                if (
                    num_terms < base_num_terms
                ):
                    base_num_terms = num_terms

            self.base_resources = {
                'num_qubits': base_num_qubits,
                'num_terms': base_num_terms,
                'reallocate': True
            }
        else:
            self.base_resources = {
                'num_qubits': 1,
                'num_terms': 1,
                'reallocate': False
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
            (2*self.num_experiments)
        )
        self.latex_config = str(
            '$P_{' + str(self.num_particles) +
            '}E_{' + str(self.num_experiments) +
            # '}B_{' + str(self.num_experiments_for_bayes_updates) +
            '}H_{' + str(number_hamiltonians_to_exponentiate) +
            r'}|\psi>_{' + str(self.probe_number) +
            '}PN_{' + str(self.exploration_class.probe_noise_level) +
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
            'debug_mode' : self.debug_mode, 
            'plot_level' : self.plot_level, 
            'long_id': self.qmla_controls.long_id,
            'model_priors': self.model_priors,  # could be path to unpickle within model?
            'experimental_measurements': self.experimental_measurements,
            'base_resources': self.base_resources,
            'store_particles_weights': False,  # TODO from exploration strategy or unneeded
            'qhl_plots': False,  # TODO get from exploration strategy
            'experimental_measurement_times': self.experimental_measurement_times,
            'num_probes': self.probe_number,  # from exploration strategy or unneeded,
            'run_info_file': self.qmla_controls.run_info_file,
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
        qmla_core_info_database.set('probes_system', compressed_probe_dict)
        qmla_core_info_database.set('probes_simulator', compressed_sim_probe_dict)

        self.qmla_core_info_database = {
            'qmla_settings': self.qmla_settings,
            'probes_system': self.probes_system,
            'probes_simulator': self.probes_simulator
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
        self.log_print([
            'Learning models from branch {} finished.'.format(branch_id)
        ])

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

        model_already_exists = self._check_model_exists(
            model_name=model_name,
        )

        if not model_already_exists:
            self.log_print([
                "Model {} not yet in database: can not be learned.".format(
                    model_name
                )
            ])
        else:
            model_id = self._get_model_id_from_name(
                model_name=model_name
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
                self.log_print(["Redis queue object:", queue,
                                "has job waiting IDs:", queue.job_ids])
                # send model-learning, as task to job queue
                queued_model = queue.enqueue(
                    remote_learn_model_parameters,
                    model_name,
                    model_id,
                    exploration_rule=self.branches[branch_id].exploration_strategy,
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
                    ["Model {} on rq job {}".format(model_id, queued_model)])
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
                # pass probes directly instead of unpickling from redis
                # database
                self.qmla_settings['probe_dict'] = self.probes_system

                remote_learn_model_parameters(
                    name=model_name,
                    model_id=model_id,
                    exploration_rule=self.branches[branch_id].exploration_strategy,
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

        unique_id = construct_models.unique_model_pair_identifier(
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
                # num_times_to_use=self.num_experiments_for_bayes_updates,
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
                # num_times_to_use=self.num_experiments_for_bayes_updates,
                branch_id=branch_id,
                bayes_threshold=self.bayes_threshold_lower,
                host_name=self.redis_host_name,
                port_number=self.redis_port_number,
                qid=self.qmla_id,
                log_file=self.rq_log_file
            )
        if wait_on_result == True:
            pair_id = construct_models.unique_model_pair_identifier(
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
        self.log_print([
            "compare_model_set with BF pair list:",
            pair_list
        ])

        remote_jobs = []
        for pair in pair_list:
            unique_id = construct_models.unique_model_pair_identifier(
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
                self.log_print([
                    "Monitoring job {}".format(job)
                ])
                while not job.is_finished:
                    if job.is_failed:
                        self.log_print([
                            "Model comparison job failed:", job
                        ])
                        raise NameError("Remote job failure")
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
        self.log_print([
            'compare_models_within_branch for branch {} has {} pairs: {}'.format(
                branch_id,
                len(pair_list),
                pair_list
            )
        ])
        
        # Set branch as active on redis db
        active_branches_bayes = self.redis_databases['active_branches_bayes']
        active_branches_bayes.set(int(branch_id), 0) 
        
        # Compare model pairs
        for a,b in pair_list:
            if a != b:
                unique_id = construct_models.unique_model_pair_identifier(
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
            pair = construct_models.unique_model_pair_identifier(a, b)
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

        # bayes_factor refers to calculation BF(pair), where pair
        # is always defined (lower, higher) for consistency
        lower_id = min(a, b)
        higher_id = max(a, b)
        self.log_print(["processing BF {}/{}".format(lower_id, higher_id)])

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
        else:
            champ = None
            self.log_print([
                "Neither model sufficiently better to earn point between {}/{}. BF={}".format(
                    mod_low.model_id, mod_high.model_id, bayes_factor
                )
            ])

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
                if res is not None: 
                    models_points[res] += 1
                self.log_print([
                    "[process_model_set_comparisons]",
                    "Point to", res,
                    "(comparison {}/{})".format(mod1, mod2)
                ])

        # Analyse pairwise competition
        self.log_print([
            "Models points: \n{}".format(models_points)
        ])
        max_points = max(models_points.values())
        models_with_max_points = [key for key, val in models_points.items()
                                  if val == max_points]
        if len(models_with_max_points) > 1:
            self.log_print([
                "Multiple models \
                have same number of points in process_model_set_comparisons:",
                models_with_max_points,
                "\n Model points:\n", models_points
            ])
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

        branch = self.branches[branch_id]
        active_models_in_branch = branch.resident_model_ids

        # Establish pairs to check comparisons between
        if pair_list is None:
            pair_list = branch.pairs_to_compare
            self.log_print([
                "Pair list not given for branch {}, generated:{}".format(
                    branch_id,
                    pair_list
                ),
            ])
        else:
            self.log_print([
                "pair list given to branch processing:", pair_list
            ])

        # Process result for each pair
        models_points = {
            k: 0
            for k in active_models_in_branch
        }
        for mod1, mod2 in pair_list:
            if mod1 != mod2:
                res = self.process_model_pair_comparison(
                    a=mod1, b=mod2
                )
                if res is not None: 
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

        # Update branch with these results to determine branch champion
        branch.update_branch(
            pair_list=pair_list, 
            models_points=models_points
        )

        # If the given results are not sufficient for the ES to determine a branch champion,
        # reconsider a subset of models
        while not branch.is_branch_champion_set:
            reduced_model_set = branch.joint_branch_champions
            self.log_print([
                "Branch champion not determined.",
                "Reconsidering:", reduced_model_set
            ])
            self.compare_model_set(
                model_id_list=reduced_model_set,
                remote=True,
                recompute=False,
                wait_on_result=True
            )
            # Pass result of compare_model_set to branch to decide if sufficient to choose champion
            models_to_recompare = list(itertools.combinations(
                reduced_model_set, 2
            ))
            self.process_comparisons_within_branch(
                branch_id=branch_id,
                pair_list=models_to_recompare
            )

        return branch.champion_id

    ##########
    # Section: routines to implement tree-based QMLA
    ##########

    def learn_models_until_trees_complete(
        self,
    ):
        r"""
        Iteratively learn/compare/generate models on exploration strategy trees.

        Each :class:`~qmla.exploration_strategies.ExplorationStrategy` has a unique :class:`~qmla.QMLATree``.
        Trees hold sets of models on :class:`~qmla.BranchTree` objects.

        Models on a each branch are learned through :meth:`learn_models_on_given_branch`.
        Any model which has previously been considered defaults to the earlier
        instance of that model, rather than repeating the calculation.
        When all models on a branch are learned, they are all compared
        through :meth:`compare_models_within_branch`.

        When a branch has completed learning and comparisons of models,
        the corresponding tree is checked to see if it has finished proposing
        models, through :meth:`~qmla.ExplorationTree.is_tree_complete`.
        If the tree is not complete, the :meth:`~qmla.ExplorationTree.next_layer`
        method is called to generate the next branch on that tree.
        The next branch can correspond to `spawn` or `prune` stages of the
        tree's :class:`~qmla.exploration_strategies.ExplorationStrategy`, but QMLA is ambivalent to the
        inner workings of the tree/exploration strategy: a branch is
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
        ctr = 0
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
                sleep(self.sleep_duration)
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
                elif ctr % 100 == 0:
                    self.log_print([
                        "Ctr {} branch {} has {} of {} models learned; model_learning_complete: {}".format(
                        ctr, 
                        branch_id,
                        int(num_models_learned_on_branch),
                        self.branches[branch_id].num_models,
                        self.branches[branch_id].model_learning_complete
                    )])

            for branchID_bytes in active_branches_bayes.keys():
                branch_id = int(branchID_bytes)
                num_comparisons_complete_on_branch = active_branches_bayes.get(
                    branchID_bytes
                )
                if (
                    not self.branches[branch_id].comparisons_complete
                    and (
                        int( num_comparisons_complete_on_branch) 
                        == self.branches[branch_id].num_model_pairs
                    )
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
                            self.branches[branch_id].exploration_strategy,
                            "Number of trees now completed:",
                            self.tree_count_completed,
                        ])
                    else:
                        # tree not complete -> launch next set of models
                        self.spawn_from_branch(
                            branch_id=branch_id,
                        )
                elif ctr % 100 == 0:
                    self.log_print([
                        "Ctr {} branch {} has {} out of {} comparisons complete; comparisons_complete: {}".format(
                        ctr,
                        branch_id, 
                        int(num_comparisons_complete_on_branch),
                        self.branches[branch_id].num_model_pairs,
                        self.branches[branch_id].comparisons_complete
                    )])
            ctr += 1

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
        to call this method, call :meth:`ExplorationTree.next_layer`, which returns
        a set of models to place on a new branch, as well as which models therein
        to compare. These are passed to :meth:`new_branch`, constructing a new branch
        in the QMLA environment. The generated new branch then has all its models
        learned by calling :meth:`~qmla.QuantumModelLearningAgent.learn_models_on_given_branch`.
        :meth:`~qmla.ExplorationTree.next_layer` is in control of how to select the next set of models,
        usually either by calling the :class:`~qmla.exploration_strategies.ExplorationStrategy`'s
        :meth:`~qmla.exploration_strategies.ExplorationStrategy.generate_models` or
        :meth:`~qmla.exploration_strategies.ExplorationStrategy.tree_pruning` methods.
        This allows the user to define how models are generated,
        given access to the comparisons of the previous branch,
        or how the tree is pruned, e.g. by performing preliminary
        parent/child branch champion comparisons.

        :param int branch_id: unique ID of the branch which has completed
        """

        model_list = self.branches[branch_id].ranked_models
        model_names = [
            self.model_name_id_map[mod_id]
            for mod_id in model_list
        ]

        new_models, models_to_compare = self.branches[branch_id].tree.next_layer(
            model_list=model_names,
            # can model_list be functionally replaced by info in branch_model_points?
            model_names_ids=self.model_name_id_map,
            called_by_branch=branch_id,
            branch_model_points=self.branches[branch_id].bayes_points,
            evaluation_log_likelihoods=self.branches[branch_id].evaluation_log_likelihoods,
            # model_dict=self.model_lists,  # is this used by any ES? TODO remove
        )

        self.log_print([
            "After model generation for ES",
            self.branches[branch_id].exploration_strategy,
            "\nnew models:", new_models,
        ])

        # Generate new QMLA level branch
        new_branch_id = self.new_branch(
            model_list=new_models,
            pairs_to_compare_by_names=models_to_compare,
            exploration_strategy=self.branches[branch_id].exploration_strategy,
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
        pairs_to_compare_by_names=None,
        exploration_strategy=None,
        spawning_branch=0,
    ):
        r"""
        Add a set of models to a new QMLA branch.

        Branches have a unique id within QMLA, but belong to a single
        tree, where each tree corresponds to a single exploration strategy.

        :param list model_list: strings corresponding to models to
            place in the branch
        :param pairs_to_compare: set of model pairs to perform comparisons between.
            'all' (deafult) means  all models in `model_list` are set to compare.
            Otherwise a list of tuples of model IDs to compare
        :type pairs_to_compare: str or list
        :param str exploration_strategy: exploration strategy identifer;
            used to get the unique tree object corresponding to an exploration strategy,
            which is then used to host the branch.
        :param int spawning_branch: branch id which is the parent of the new branch.
        :return: branch id which uniquely identifies the new branch
            within the QMLA environment.
        """

        model_list = list(set(model_list))  # remove possible duplicates
        branch_id = int(self.branch_highest_id) + 1
        self.branch_highest_id = branch_id

        if exploration_strategy is None:
            exploration_strategy = self.exploration_strategy_of_true_model
        exploration_tree = self.trees[exploration_strategy]

        this_branch_models = {}
        model_id_list = []
        pre_computed_models = []
        for model in model_list:
            # add_model_to_database returns whether adding model was successful
            # if false, that's because it's already been computed
            add_model_info = self.add_model_to_database(
                model,
                branch_id=branch_id,
                exploration_tree=exploration_tree,
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

        model_storage_instances = {
            m: self.get_model_storage_instance_by_id(m)
            for m in list(this_branch_models.keys())
        }

        # Start new branch on corresponding exploration strategy tree

        if pairs_to_compare_by_names is not None:
            if pairs_to_compare_by_names == 'all':
                pairs_to_compare = 'all'
            else:
                self.log_print(["Getting model IDs to set comparison subset"])
                try:
                    pairs_to_compare = [(
                        self.model_database[self.model_database.model_name == m1].model_id.item(),
                        self.model_database[self.model_database.model_name == m2].model_id.item()
                        ) for m1, m2 in pairs_to_compare_by_names 
                    ]
                    self.log_print(["IDs:", pairs_to_compare])
                except BaseException:
                    self.log_print(
                        ["Failed to unpack pairs_to_compare_by_names:\n", pairs_to_compare_by_names])
                    raise

        self.branches[branch_id] = exploration_tree.new_branch_on_tree(
            branch_id=branch_id,
            models=this_branch_models,
            pairs_to_compare=pairs_to_compare,
            model_storage_instances=model_storage_instances,
            precomputed_models=pre_computed_models,
            spawning_branch=spawning_branch,
        )

        return branch_id

    def add_model_to_database(
        self,
        model,
        exploration_tree,
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

        model_name = construct_models.alph(model)
        self.log_print(["Trying to add model to DB:", model_name])

        # Add model if not yet considered or told to force create
        if (
            self._consider_new_model(model_name) == 'New'
            or force_create_model == True
        ):
            # create new model instance
            model_num_qubits = qmla.construct_models.get_num_qubits(
                model_name)
            model_id = self.highest_model_id + 1
            self.model_lists[model_num_qubits].append(model_name)

            self.log_print([
                "Model {} not previously considered -- adding with ID {}".format(
                    model_name, model_id
                )
            ])
            op = qmla.construct_models.Operator(
                name=model_name
            )
            # Generate model storage instance
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

            # Add to the model database
            f_score = np.round(self.compute_model_f_score(
                model_id=model_id,
                model_name=model_name,
                exploration_class=exploration_tree.exploration_class
            ), 2)
            terms = qmla.construct_models.get_constituent_names_from_name(model_name)

            running_db_new_row = pd.Series({
                'model_id': int(model_id),
                'model_name': model_name,
                'latex_name': exploration_tree.exploration_class.latex_name(model_name),
                'branch_id': int(branch_id),
                'f_score': f_score,
                'model_storage_instance': model_storage_instance,
                'branches_present_on' : [int(branch_id)], 
                'terms' : terms,
                'latex_terms' : [exploration_tree.exploration_class.latex_name(t) for t in terms] # need to get latex name by the ES which spawned this model
            })
            num_rows = len(self.model_database)
            self.model_database.loc[num_rows] = running_db_new_row

            model_added = True
            if construct_models.alph(
                    model) == construct_models.alph(self.true_model_name):
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
                model_id = self._get_model_id_from_name(
                    model_name=model_name
                )
                self.log_print([
                    "Previously considered as model ID ", model_id
                ])
                self.model_database[ 
                    self.model_database.model_id == model_id 
                ].branches_present_on.item().append(int(branch_id))
                
                if model_id == self.true_model_id:
                    self.true_model_on_branhces.append(model_id)
            except BaseException:
                self.log_print([
                    "Couldn't find model id for model:", model_name,
                    "model_names_ids:",
                    self.model_name_id_map
                ])
                raise

        add_model_output = {
            'is_new_model': model_added,
            'model_id': model_id,
        }
        return add_model_output

    def finalise_instance(self):
        self.compute_statistical_metrics_by_generation()
        self.exploration_class.exploration_strategy_finalise()

        if self.qhl_mode_multiple_models:
            self.log_print(["No special analysis for this mode"])
        elif self.qhl_mode:
            self.log_print(["No special analysis for this mode"])
        else:
            self.finalise_qmla()

    def finalise_qmla(self):
        r"""
        Steps to end QMLA algorithm, such as storing analytics.

        """

        champ_model = self.get_model_storage_instance_by_id(
            self.champion_model_id)

        # compute full dynamics for branch champions
        champ_model.compute_expectation_values(
            times=self.times_to_plot,
        )
        self.branch_champions = [
            self.branches[b].champion_id for b in self.branches
        ]
        self.log_print([
            "Branch champions:", self.branch_champions 
        ])
        for bc in self.branch_champions:
            bc_mod = self.get_model_storage_instance_by_id(bc)
            bc_mod.compute_expectation_values(
                times = self.times_to_plot
            )

        # Get metrics for all models tested
        for i in self.models_learned:
            # dict of all Bayes factors for each model considered.
            self.all_bayes_factors[i] = (
                self.get_model_storage_instance_by_id(i).model_bayes_factors
            )

        self.bayes_factors_data()

        # Prepare model/name maps
        self.model_id_to_name_map = {}
        for k in self.model_name_id_map:
            v = self.model_name_id_map[k]
            self.model_id_to_name_map[v] = k

        try:
            self.branch_graphs = qmla.analysis.branch_graphs.plot_qmla_branches(
                q=self, return_graphs=False
            )
        except:
            self.log_print(["Failed to plot branch graphs."])

        # Store model IDs and names
        model_data = self.model_database[
            # subset of columns to store
            ['model_id', 'model_name', 'latex_name', 'branch_id', 'f_score',] # TODO add log_likelihood here
        ]
        model_data.to_csv(
            os.path.join(
                self.qmla_controls.plots_directory,
                'model_directory.csv')
        )

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
                    d = pd.Series({
                        'model_a': mod_name_a,
                        'id_a': mod_id_a,
                        'f_score_a': f_score_a,
                        'model_b': mod_name_b,
                        'id_b': mod_id_b,
                        'f_score_b': f_score_b,
                        'bayes_factor': bf,
                        'log10_bayes_factor': np.round(np.log10(bf), 1)
                    })
                    new_idx = len(self.bayes_factors_df)
                    self.bayes_factors_df.loc[new_idx] = d
        

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
        n_qubits = construct_models.get_num_qubits(model_name)
        if n_qubits > 5:
            expec_val_plot_times = self.times_to_plot_reduced_set
        else:
            expec_val_plot_times = self.times_to_plot

        mod.compute_expectation_values(
            times=expec_val_plot_times,
        )

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
        num_params_champ_model = construct_models.Operator(
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
            # Details about QMLA instance:
            'QID': self.qmla_id,
            'NumParticles': self.num_particles,
            'NumExperiments': mod.num_experiments,
            # 'NumBayesTimes': self.num_experiments_for_bayes_updates,
            'ConfigLatex': self.latex_config,
            'Heuristic': mod.model_heuristic_class,
            'Time': time_taken,
            'Host': self.redis_host_name,
            'Port': self.redis_port_number,
            'ResampleThreshold': self.exploration_class.qinfer_resampler_threshold,
            'ResamplerA': self.exploration_class.qinfer_resampler_a,
            # Details about true model:
            'TrueModel': self.true_model_name,
            'TrueModelConsidered': self.true_model_considered,
            'TrueModelFound': self.true_model_found,
            'TrueModelBranch': self.true_model_branch,
            'Truemodel_id': self.true_model_id,
            'TrueModelConstituentTerms': self.true_model_constituent_terms_latex,
            # Details about this model
            'ChampID': model_id,
            'ChampLatex': mod.model_name_latex,
            'ConstituentTerms': mod.constituents_terms_latex,
            'LearnedHamiltonian': mod.learned_hamiltonian,
            'ExplorationRule': mod.exploration_strategy_of_this_model,
            'NameAlphabetical': construct_models.alph(mod.model_name),
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
            # Comparison to true model (for simulated cases)
            'NumParamDifference': num_params_difference,
            'Underfit': underfit,
            'Overfit': overfit,
            'Misfit': misfit,
            'CorrectModel': correct_model,
            # About QMLA's learning procedure:
            'NumModels': len(self.models_learned),
            'StatisticalMetrics': self.generational_statistical_metrics,
            'GenerationalFscore': self.generational_f_score,
            'GenerationalLogLikelihoods': self.generational_log_likelihoods,
            'ModelEvaluationLogLikelihoods': model_evaluation_log_likelihoods,
            'ModelEvaluationMedianLikelihoods': model_evaluation_median_likelihoods,
            'AllModelFScores': self.model_f_scores,
        }

        self.storage = qmla.utilities.StorageUnit()
        self.storage.qmla_id = self.qmla_id
        self.storage.bayes_factors_df = self.bayes_factors_df
        self.storage.model_f_scores = self.model_f_scores
        self.storage.exploration_strategy_storage = self.exploration_class.storage

        # store expectation values of all models

        df_cols = ['time', 'exp_val', 'model_id', 'qmla_id']
        expectation_values_df = pd.DataFrame(columns=df_cols)

        for m in self.models_learned:
            mod = self.get_model_storage_instance_by_id(m)
            times = list(sorted(mod.expectation_values.keys()))
            ev = [mod.expectation_values[t] for t in times]    
            d = pd.DataFrame(
                columns = df_cols,
            )
            d['time'] = times
            d['exp_val'] = ev
            d['model_id'] = m
            d['qmla_id'] = self.qmla_id
            
            expectation_values_df = expectation_values_df.append(d)
        
        self.storage.expectation_values = expectation_values_df            
        try:
            # TODO this fails for QHL mode since champion not assigned -- fix
            self.storage.branch_champions = {
                b : self.branches[b].champion_id
                for b in self.branches
            }
        except:
            pass

        models_generated = self.model_database[
            ['model_name', 'model_id', 'latex_name', 'f_score', 'terms']
        ]

        models_generated['champion'] = False
        models_generated.loc[
            (models_generated.model_id == self.champion_model_id), 
            'champion'
        ] = True
        self.storage.models_generated = models_generated

        for r in results_dict:
            # TODO: get rid of results_dict; use storage class instead to achieve the same things
            self.storage.__setattr__(r, results_dict[r])

        return results_dict

    def check_champion_reducibility(
        self,
    ):
        r"""
        Potentially remove negligible terms from the champion model.

        Consider whether the champion model has some terms whose parameters
        were found to be negligible (either within one standard
        deviation from 0, or very close to zero as determined by the exploration strategy's
        `learned_param_limit_for_negligibility` attribute).
        Construct a new model which is the same as the champion, less those negligible
        terms, named the reduced champion. The data of the champion model is inherited
        by the reduced candidate model, i.e. its parameter estimates, as well as
        its history of parameter learning for those which are not negligible.
        A new `normalization_record` is started, which is used in the comparison between
        the champion and the reduced champion.
        Compare the champion with the reduced champion; if the reduced champion
        is heavily favoured, directly select it as the global champion.
        This method is triggered if the exploration strategy's `check_champion_reducibility`
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
                < self.exploration_class.learned_param_limit_for_negligibility
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
            new_mod = construct_models.alph(new_mod)

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
                construct_models.get_constituent_names_from_name(
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
                < (1.0 / self.exploration_class.reduce_champ_bayes_factor_threshold)
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
        Compare the champions of all exploration strategy trees.

        Get the champions (usually one, but in general can be multiple)
        from each tree, where each tree is unique to an exploration strategy.
        Place the champions on a branch together and perform all-versus-all
        comparisons. The champion of that branch is deemed the global champion.

        """

        tree_champions = []
        for tree in self.trees.values():
            # extend in case multiple models nominated by tree
            tree_champions.extend(tree.nominate_champions())

        # Place tree champions on new QMLA branch, not tied to an exploration strategy
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

        The `true_model` of the :class:`~qmla.exploration_strategies.ExplorationStrategy` is used to generate
        true data (in simulation) and have its parameters learned.

        """

        qhl_branch = self.new_branch(
            exploration_strategy=self.exploration_strategy_of_true_model,
            model_list=[self.true_model_name]
        )

        mod_to_learn = self.true_model_name
        self.log_print([
            "QHL for true model:", mod_to_learn,
        ])

        self.learn_model(
            model_name=mod_to_learn,
            branch_id=qhl_branch,
            blocking=True
        )
        mod_id = self._get_model_id_from_name(
            model_name=mod_to_learn
        )

        # These don't really matter for QHL,
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
        self.exploration_class.exploration_strategy_finalise()
        self.finalise_instance()
        # self._plot_statistical_metrics()

    def run_quantum_hamiltonian_learning_multiple_models(
        self,
        model_names=None
    ):
        r"""
        Run Quantum Hamiltonian Learning algorithm with multiple simulated models.

        Numerous Hamiltonian models attempt to learn the dynamics of the true model.
        The underlying model is set in the :class:`~qmla.exploration_strategies.ExplorationStrategy`'s `true_model` attribute.

        :param list model_names:
            list of strings of model names to learn the parameterisations of.
            None: taken from :class:`~qmla.exploration_strategies.ExplorationStrategy` `qhl_models`.
        """

        # Choose models to perform QHL on
        if model_names is None:
            model_names = self.exploration_class.qhl_models

        # Place models on a branch
        branch_id = self.new_branch(
            exploration_strategy=self.exploration_strategy_of_true_model,
            model_list=model_names
        )
        self.qhl_mode_multiple_models = True
        self.champion_model_id = -1,  # TODO just so not to crash during dynamics plot
        self.qhl_mode_multiple_models_model_ids = [
            self._get_model_id_from_name(
                model_name=mod_name
            ) for mod_name in model_names
        ]
        self.log_print([
            'QHL for multiple models:', model_names,
        ])
        learned_models_ids = self.redis_databases['learned_models_ids']

        # learn models
        for mod_name in model_names:
            mod_id = self._get_model_id_from_name(
                model_name=mod_name
            )
            learned_models_ids.set(
                str(mod_id), 0
            )
            self.learn_model(
                model_name=mod_name,
                branch_id=branch_id,
                blocking=False
            )

        running_models = learned_models_ids.keys()
        self.log_print([
            'Running Models:', running_models,
        ])
        for k in running_models:
            # waiting on all models to finish,
            while int(learned_models_ids.get(k)) != 1:
                sleep(self.sleep_duration)
                self._inspect_remote_job_crashes()

        # Learning finished
        self.log_print([
            'Finished learning for all:', running_models,
        ])

        # Tidy up: store learned info, analyse, etc.
        for mod_name in model_names:
            mod_id = self._get_model_id_from_name(
                model_name=mod_name
            )
            mod = self.get_model_storage_instance_by_id(mod_id)
            mod.model_update_learned_values()

        self.exploration_class.exploration_strategy_finalise()
        self.model_id_to_name_map = {}
        for k in self.model_name_id_map:
            v = self.model_name_id_map[k]
            self.model_id_to_name_map[v] = k
        for k in self.timings:
            self.log_print([
                "QMLA Timing - {}: {}".format(k, np.round(self.timings[k], 2))
            ])
        self.finalise_instance()

    def run_complete_qmla(
        self,
    ):
        r"""
        Run complete Quantum Model Learning Agent algorithm.

        Each :class:`~qmla.exploration_strategies.ExplorationStrategy` is assigned a :class:`~qmla.tree.QMLATree`,
        which manages the exploration strategy. When new models are spawned by an exploration strategy,
        they are placed on a :class:`~qmla.tree.BranchQMLA` of the corresponding tree.
        Models are learned/compared/spawned iteratively in
        :meth:`learn_models_until_trees_complete`, until all
        trees declare that their exploration strategy has completed.
        Exploration Strategies are complete when they have nominated one or more champions,
        which can follow spawning/pruning stages as required by the exploration strategy.
        Nominated champions are then compared with :meth:`compare_nominated_champions`,
        resulting in a single global champion selected.
        Some analysis then takes place, including possibly reducing the
        selected global champion if it is found that some of its terms are not impactful.

        """

        # Set up one tree per exploration strategy
        for tree in list(self.trees.values()):
            starting_models, models_to_compare = tree.get_initial_models()
            # TODO genetic alg giving some non-unique initial model sets
            self.log_print([
                "First branch for {} has ( {}/{} unique ) starting models: {}".format(
                    tree.exploration_strategy, 
                    len(set(starting_models)), 
                    len(starting_models),
                    starting_models
                ),
                # "models_to_compare:", models_to_compare
            ])
            self.new_branch(
                model_list=starting_models,
                exploration_strategy=tree.exploration_strategy,
                pairs_to_compare_by_names=models_to_compare
            )

        # Iteratively learn models, compute bayes factors, spawn new models
        self.learn_models_until_trees_complete()
        self.log_print([
            "Exploration Strategy trees completed."
        ])

        # Choose champion by comparing nominated champions of all trees.
        self.compare_nominated_champions()
        self.champion_model_id = self._get_model_data_by_field(
            name=self.global_champion_name,
            field='model_id'
        )
        self.log_print(
            ["Champion selected. ID={}".format(self.champion_model_id)])

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
                    construct_models.alph(self.true_model_name)
                )
            ])
        self.log_print([
            "True model considered: {}. on branch {}.".format(
                self.true_model_considered,
                self.true_model_branch
            )
        ])

        # Consider reducing champion if negligible parameters found
        if self.exploration_class.check_champion_reducibility:
            self.check_champion_reducibility()

        # Tidy up and finish QMLA.
        self.finalise_instance()

        self.log_print([
            "\nFinal winner:", self.global_champion_name,
            "(ID {}) has F-score {}".format(
                self.champion_model_id, 
                np.round(
                    self.model_f_scores[self.champion_model_id], 2)
                )
        ])

    ##########
    # Section: Database interface
    ##########

    def _get_model_data_by_field(self, name, field):
        r"""
        Get any data from the model database corresponding to a given model name.

        :param str name: model name to get data of
        :param str field: field name to get data corresponding to model
        """

        d = self.model_database[self.model_database['model_name'] == name][field].item()
        return d

    def _get_model_id_from_name(self, model_name):
        model_id = self._get_model_data_by_field(
            name=model_name,
            field='model_id'
        )
        return model_id

    def _consider_new_model(self, model_name):
        r"""
        Check whether a proposed model already exists.

        Check whether the new model `name`, exists in
        all previously considered models, held in `model_lists`, organised
        by dimension of models.
        If name has not been previously considered, 'New' is returned.
        If name has been previously considered, the corresponding location
            in db is returned.

        :param dict model_lists: lists of models already considered, organised
            by the number of qubits of those models
        :param str name: model for consideration
        """
        # Return true indicates it has not been considered and so can be added
        al_name = qmla.construct_models.alph(model_name)
        n_qub = qmla.construct_models.get_num_qubits(model_name)
        if al_name in self.model_lists[n_qub]:
            return 'Previously Considered'  # todo -- make clear if in legacy or running db
        else:
            return 'New'

    def _check_model_exists(self, model_name):
        r"""
        True if model already exists; False if not.
        """
        if self._consider_new_model(model_name) == 'New':
            return False
        else:
            return True

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
            self.log_print([
                "Failure on remote job. Terminating QMLA."
            ])
            raise NameError('Remote model learning failure')
        self.timings['inspect_job_crashes'] += time.time() - t_init

    def _delete_unpicklable_attributes(self):
        r"""Remove elements of QMLA which cannot be pickled, which cause errors if retained."""

        del self.redis_conn
        del self.redis_databases
        del self.write_log_file

    ##########
    # Section: Analysis/plotting methods
    ##########

    def analyse_instance(self):
        r""" Basic analysis of this instance"""
        
        pickle.dump(
            self.get_results_dict(),
            open(self.qmla_controls.results_file, "wb"),
            protocol=4
        )
        storage_location = os.path.join(
            self.qmla_controls.results_directory, 
            'storage_{}.p'.format(self.qmla_controls.long_id), 
        )
        pickle.dump(
            self.storage, 
            open(storage_location, 'wb'),
            protocol = 4, 
        )

        if self.qhl_mode: 
            self._analyse_qhl()

        elif self.qhl_mode_multiple_models:
            self._analyse_multiple_model_qhl()
        
        else: 
            self._analyse_qmla()

    def _analyse_qhl(self):
        return

    def _analyse_multiple_model_qhl(self):
        model_ids = [
            self._get_model_id_from_name(
                model_name=mod
            ) for mod in self.exploration_class.qhl_models
        ]
        
        for mid in model_ids:
            mod = self.get_model_storage_instance_by_id(mid)
            name = mod.model_name
            results_file = str(
                self.qmla_controls.results_directory +
                output_prefix +
                'results_' +
                str("m{}_q{}.p".format(
                    int(mid), self.qmla_controls.long_id)
                )
            )

            pickle.dump(
                self.get_results_dict(model_id = mid),
                open(results_file, "wb"),
                protocol=4
            )
    
    def _analyse_qmla(self):
        expec_value_mods_to_plot = []
        try:
            expec_value_mods_to_plot = [self.true_model_id]
        except BaseException:
            pass

        expec_value_mods_to_plot.append(self.champion_model_id)
        champ_mod = self.get_model_storage_instance_by_id(
            self.champion_model_id
        )

        try:
            self.store_bayes_factors_to_csv(
                save_to_file=str(
                    self.qmla_controls.results_directory +
                    'bayes_factors_' + str(self.qmla_controls.long_id) + '.csv'
                ),
                names_ids='latex'
            )
        except Exception as e:
            self.log_print([
                "failed to store_bayes_factors_to_csv with error {}".format(e)
            ])

    def store_bayes_factors_to_csv(
        self, 
        save_to_file, 
        names_ids='latex'
    ):
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
        # TODO this doesn't get used anywhere useful any more; remove
        qmla.analysis.update_shared_bayes_factor_csv(self, self.qmla_controls.cumulative_csv)

    def compute_model_f_score(
        self,
        model_id,
        model_name=None,
        exploration_class=None,
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
        true_set = self.exploration_class.true_model_terms
        if exploration_class is None:
            exploration_class = self.get_model_storage_instance_by_id(
                model_id).exploration_class
            model_name = self.model_name_id_map[model_id]
        terms = [
            exploration_class.latex_name(
                term
            )
            for term in
            construct_models.get_constituent_names_from_name(
                model_name
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

    def plot_instance_outcomes(
        self, 
    ):
        self.log_print([
            "Plotting instance outcomes"
        ])

        plot_methods = [
            self._plot_model_terms,
            self._plot_dynamics_all_models_on_branches,
            self._plot_one_qubit_probes_bloch_sphere, 
            self._plot_evaluation_normalisation_records,
            self._plot_bayes_factors,
            self._plot_branch_champs_quadratic_losses,
            self._plot_branch_champs_volumes,
            self._plot_exploration_tree,
            self._plot_r_squared_by_epoch_for_model_list,
            self._plot_statistical_metrics
        ]

        for method in plot_methods:
            try:
                method()
            except Exception as e:
                self.log_print([
                    "plot failed {} with exception: {}".format(method.__name__, e)
                ])

        self.log_print([
            "Plotting exploration strategy analysis"
        ])
        self.exploration_class.exploration_strategy_specific_plots(
            save_directory = self.qmla_controls.plots_directory,
            qmla_id = self.qmla_controls.long_id,
            true_model_id = self.true_model_id, 
            champion_model_id = self.champion_model_id, 
            plot_level = self.plot_level
        )

    def compute_statistical_metrics_by_generation(self):
        r"""
        Compute, store and plot various statistical metrics of all studied models.

        :param str save_to_file: path to save the resultant figure in.
        """
        generations = sorted(set(self.branches.keys()))
        self.log_print([
            "[compute_statistical_metrics_by_generation]",
            "generations: ", generations
        ])

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
            self.log_print([
                "Adding models to generational measures for Generation {}:{}".format(
                    b,
                    models_this_branch
                )
            ])
            for m in models_this_branch:
                generational_sensitivity[b].append(self.model_sensitivities[m])
                generational_precision[b].append(self.model_precisions[m])
                generational_f_score[b].append(self.model_f_scores[m])
                self.generational_log_likelihoods[b].append(
                    self.get_model_storage_instance_by_id(
                        m).evaluation_log_likelihood
                )
        self.generational_f_score = generational_f_score
        self.generational_sensitivity = generational_sensitivity
        self.generational_precision = generational_precision

        self.stat_data = [
            {'name': 'F-score', 'data': self.generational_f_score, 'colour': 'red'},
            {'name': 'Precision', 'data': self.generational_precision, 'colour': 'blue'},
            {'name': 'Sensitivity',
             'data': self.generational_sensitivity,
             'colour': 'green'},
        ]
        self.generational_statistical_metrics = {
            k['name']: k['data']
            for k in self.stat_data
        }

    
    def _plot_statistical_metrics(
        self,
        save_to_file=None
    ):
        generations = sorted(set(self.branches.keys()))
        self.alt_generational_statistical_metrics = {
            b: {
                'Precision': self.generational_precision[b],
                'Sensitivity': self.generational_sensitivity[b],
                'F-score': self.generational_f_score[b]
            }
            for b in generations
        }
        include_plots = self.stat_data
        lf = LatexFigure(
            gridspec_layout=(1, len(include_plots))
        )
        
        plot_col = 0
        for plotting_data in include_plots:

            # ax = fig.add_subplot(gs[0, plot_col])
            ax = lf.new_axis()
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
            # plot_col += 1

        self.log_print(["getting statistical metrics complete"])
        if save_to_file is not None:
            plt.savefig(save_to_file)

    def _plot_bayes_factors(
        self,
    ):
        r"""
        Plot Bayes factors between pairs of models, both by model IDs and by their F-scores.
        """

        # Plot Bayes factors of this instance
        bayes_factor_by_id = pd.pivot_table(
            self.bayes_factors_df,
            values='log10_bayes_factor',
            index=['id_a'],
            columns=['id_b'],
            aggfunc=np.median
        )
        mask = np.tri(bayes_factor_by_id.shape[0], k=-1).T
        
        lf = LatexFigure()
        ax = lf.new_axis()
        plt.clf()
        sns.heatmap(
            bayes_factor_by_id,
            # cmap='RdYlGn',
            cmap=self.exploration_class.bf_cmap, 
            mask=mask,
            ax=ax, 
            annot=False
        )
        # s.get_figure().savefig(
        lf.save(
            os.path.join(
                self.qmla_controls.plots_directory,
                'bayes_factors.png'.format(self.qmla_controls.long_id)
            )
        )

        # Heat map BF against F(A)/F(B)
        fig = qmla.analysis.bayes_factor_f_score_heatmap(
            bayes_factors_df=self.bayes_factors_df)
        fig.savefig(
            os.path.join(
                self.qmla_controls.plots_directory, "bayes_factors_by_f_score"
            )
        )

    def _plot_branch_champs_quadratic_losses(
        self,
    ):
        r"""Wrapper for :func:`~qmla.analysis.plot_quadratic_loss`."""
        qmla.analysis.plot_quadratic_loss(
            qmd=self,
            champs_or_all='champs',
            save_to_file=os.path.join(
                self.qmla_controls.plots_directory,
                "quadratic_losses_branch_champs.pdf"
            )
        )

    def _plot_branch_champs_volumes(
        self, 
        model_id_list=None, 
        branch_champions=True,
        branch_id=None, 
        save_to_file=None
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
            model_id_list = list(
                self.model_database[
                    self.model_database['branch_id'] == branch_id]['model_id']
            )
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

    def _plot_parameter_learning_champion(
        self,
    ):
        r"""
        Plot parameter estimates vs experiment number for a single model.

        Wrapper for :func:`~qmla.analysis.plot_parameter_estimates`
        :param bool true_model: whether to force only plotting the true
            model's parameter estimeates
        """

        qmla.analysis.plot_parameter_estimates(
            qmd=self,
            model_id=self.champion_model_id,
            save_to_file=os.path.join(
                self.qmla_controls.plots_directory, 
                "champion_parameters.png"
            )
        )

    def _plot_parameter_learning_true(
        self,
    ):
        r"""
        Plot parameter estimates vs experiment number for a single model.

        Wrapper for :func:`~qmla.analysis.plot_parameter_estimates`
        :param bool true_model: whether to force only plotting the true
            model's parameter estimeates
        """
        if self.true_model_id == -1:
            return 

        qmla.analysis.plot_parameter_estimates(
            qmd=self,
            model_id=self.true_model_id,
            save_to_file=os.path.join(
                self.qmla_controls.plots_directory, 
                "champion_parameters.png"
            )
        )

    def _plot_parameter_learning_single_model(
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
            model_id = self._get_model_id_from_name(name=self.true_model_name)

        qmla.analysis.plot_parameter_estimates(qmd=self,
                                               model_id=model_id,
                                               save_to_file=save_to_file
                                               )

    def _plot_branch_champions_dynamics(
        self,
        all_models=False,
        model_ids=None,
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
        elif self.exploration_class.tree_completed_initially:
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
        path_to_save = os.path.join(
            self.qmla_controls.plots_directory,
            'dynamics.png')
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
        except BaseException:
            self.log_print(["Failed to plot dynamics"])
            # raise

    def _plot_volume_after_qhl(self,
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

    def _plot_exploration_tree(
        self,
        modlist=None,
        only_adjacent_branches=True,
        save_to_file=None
    ):
        r"""Wrapper for :func:`~qmla.analysis.plot_qmla_single_instance_tree`"""
        if save_to_file is None: 
            save_to_file = os.path.join(
                self.qmla_controls.plots_directory, 
                "exploration_tree.png"
            )

        qmla.analysis.plot_qmla_single_instance_tree(
            self,
            modlist=modlist,
            only_adjacent_branches=only_adjacent_branches,
            save_to_file=save_to_file
        )

    def _plot_qmla_radar_scores(self, modlist=None, save_to_file=None):
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

    def _plot_r_squared_by_epoch_for_model_list(
        self,
        modlist=None,
        save_to_file=None
    ):
        r"""
        Plot $R^2$ vs experiment number for given model list.
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
        
        if save_to_file is None: 
            save_to_file = os.path.join(
                self.qmla_controls.plots_directory, 
                "r_squareds.png"
            )

        qmla.analysis.r_squared_from_epoch_list(
            qmd=self,
            model_ids=modlist,
            save_to_file=save_to_file
        )

    def _plot_one_qubit_probes_bloch_sphere(
        self, 
        save=False
    ):
        r"""Show all one qubit probes on Bloch sphere."""

        qmla.utilities.plot_probes_on_bloch_sphere(
            probe_dict = self.probes_system, 
            num_probes = self.probe_number, 
            save_to_file=os.path.join(
                self.qmla_controls.plots_directory,
                'probes_bloch_sphere.png'
            )
        )

    def _plot_model_terms(self, colour_by='binary'):
        """
        Plot the terms of each model by model ID. 

        :param colour_by: defaults to 'binary' for black/white; alternatively colour by f_score of model
        :type colour_by: str, optional
        """        
        plt.rcParams.update({"text.usetex" : False})
        if self.plot_level < 1:
            return
        
        # Prepare dataframes
        unique_terms = list(set(qmla.utilities.flatten(list(self.model_database.latex_terms))))

        unique_branches = list(set(qmla.utilities.flatten(list(self.model_database.branches_present_on))))
        unique_branches = [
            "branch_{}".format(int(b)) for b in unique_branches
        ]

        database_columns = ['model_id', 'f_score'] + unique_terms
        model_reference_database = pd.DataFrame(
            columns = database_columns
        )

        branch_cols = ['model_id', 'f_score'] + unique_branches
        models_branches = pd.DataFrame(columns=branch_cols)

        for model_id in self.model_database.model_id:

            model_data = self.model_database[
                 self.model_database.model_id == model_id]
            model_id = int(model_id)
            f_score = model_data['f_score'].item()

            if colour_by == 'binary':
                terms_in_model = {
                    term : int(1) # for binary representation
                    for term in model_data.latex_terms.item()
                }
            elif colour_by == 'f_score' : 
                terms_in_model = {
                    term : f_score # to colour by f_score
                    for term in model_data.latex_terms.item()
                }
                
            terms_in_model['model_id'] = int(model_id)
            terms_in_model['f_score'] = model_data.f_score.item()
            model_reference_database.loc[ len(model_reference_database) ] = pd.Series(terms_in_model)

            branches = {
                "branch_{}".format(int(b)) : 1
                for b in model_data.branches_present_on.item()
            }
            branches['model_id'] = int(model_id)

            models_branches.loc[len(models_branches)] = pd.Series(branches)

        if colour_by == 'binary':
            models_branches.fillna(0, inplace=True)
            model_reference_database.fillna(0, inplace=True)

        piv_table = pd.pivot_table(    
            columns = ['model_id'], 
            values = unique_terms, 
            data = model_reference_database    
        ).transpose()

        # Plot as heatmap
        # fig, ax = plt.subplots(figsize=(15,10))
        lf = LatexFigure()
        ax = lf.new_axis()

        if colour_by == 'f_score':
            sns.heatmap(
                piv_table,
                cmap=self.exploration_class.f_score_cmap, 
                ax = ax,
                cbar_kws={'label': 'F-score', }
            )
        elif colour_by == 'binary':
            sns.heatmap(
                piv_table,
                linewidths=.5,
                cmap='binary',
                cbar=False,
                ax = ax,
            )

        ax.tick_params(which='y', rotation=0)
        fontsize = 20
        ax.tick_params(
            top=True, 
            bottom=False,
            labeltop=True,
            labelbottom=False,
            labelrotation=0,
            labelsize=fontsize
        )
        ax.set_ylabel('Model ID', fontsize=2*fontsize)
        ax.set_xlabel('Term')

        # fig.savefig(
        #     os.path.join(
        #         self.qmla_controls.plots_directory, "composition_of_models.png"
        #     )
        # )
        lf.save(
            os.path.join(
                self.qmla_controls.plots_directory, "composition_of_models.png"
            )
        )

    def _plot_dynamics_all_models_on_branches(self, branches=None):
        """Plot the dynamics of all models on given branches.

        :param branches: list of branches to draw dynamics for, defaults to None, in which case all branches are drawn. 
        :type branches: list, optional
        """        
        self.branch_results_dir = os.path.join(
            self.qmla_controls.plots_directory, 
            'branches'
        )
        try:
            os.makedirs(self.branch_results_dir)
        except:
            pass

        if not self.plot_level >= 3:
            return


        if branches is None:
            branches = sorted(list(self.branches.keys()))

        colours = itertools.cycle(
            ['blue', 'orange', 'green', 'cyan', 'purple', 'olive', 'grey']
        )
        linestyles = itertools.cycle(
            ['solid','dashed', 'dotted', 'dashdot']
        )
        max_models_per_subplot = 5

        for branch_id in branches:
            models = self.branches[branch_id].resident_model_ids
            times = sorted(self.experimental_measurements.keys())

            plt.clf()
            fig = plt.figure(
                figsize=(15, 10),
                tight_layout=True
            )
            num_rows = math.ceil( len(models) / max_models_per_subplot )
            self.log_print([
                "plotting branch dynamics. On branch {} there are {} rows".format(
                    branch_id, num_rows
                )
            ])
            gs = GridSpec(
                nrows=num_rows,
                ncols=1,
            )

            col = 0
            row = 0
            n_models_this_row = 0
            ax = fig.add_subplot(gs[row, col])

            for m in models:
                
                mod = self.get_model_storage_instance_by_id(m)
                computed_expec_val_times = sorted(mod.expectation_values.keys())
                try:
                    exp_vals = [
                        mod.expectation_values[t] 
                        for t in computed_expec_val_times
                    ]
                except:
                    self.log_print([
                        "Failed to get expectation values for model id {}".format(m)
                    ])
                    raise
                ax.plot(
                    computed_expec_val_times, 
                    exp_vals, 
                    label="{} (ID={}, $LL$={})".format(mod.model_name_latex, m, mod.evaluation_log_likelihood),
                    color=next(colours), 
                    ls=next(linestyles)
                )

                n_models_this_row += 1
                if (
                    n_models_this_row == max_models_per_subplot
                ):
                    row += 1
                    n_models_this_row = 0 
                    ax = fig.add_subplot(gs[row, col])
                    
            for row in range(num_rows):
                # Finish each subplot
                ax = fig.add_subplot(gs[row, col])

                ax.scatter(
                    times, 
                    [self.experimental_measurements[t] for t in times],
                    c = 'red',
                    label = 'System',
                    s = 5, 
                )

                ax.set_xlim(0, max(times))
                ax.set_ylabel('Expectation Value')
                ax.set_xlabel('Time ($s$)')
                ax.legend(
                    bbox_to_anchor=(1.2, 1.05),
                    fontsize=12, 
                )

            path = os.path.join(
                self.branch_results_dir, 
                'dynamics_branch_{}.png'.format(branch_id)
            )

            fig.savefig(
                path
            )


    def _plot_evaluation_normalisation_records(self):
        """Plot the normalisation record of all models grouped by the branch they are on.
        """        
        if self.plot_level < 3 : 
            return

        for branch_id in list(self.branches.keys()):

            fig, ax = plt.subplots(
                figsize=(15, 10),
                tight_layout=True
            )
            for m in self.branches[branch_id].resident_model_ids:
                mod = self.get_model_storage_instance_by_id(m)
                
                ax.hist(
                    qmla.utilities.flatten(mod.evaluation_normalization_record),
                    bins = np.arange(0, 1, 0.05), 
                    label = "{} ($LL={}$)".format(
                        mod.model_name_latex,
                        # TODO use ES of branch to get latex name
                        mod.evaluation_log_likelihood
                    ),
                    histtype='step'
                )
            ax.legend(
                bbox_to_anchor=(1.1, 1.05),
                fontsize=12, 
            )

            ax.set_ylabel('Frequency')
            ax.set_xlabel('Likelihood')
            ax.set_title('Normalisation record for evaluating models on branch {}'.format(branch_id))

            fig.savefig(
                os.path.join(
                    self.branch_results_dir, 
                    'normalisation_record_branch_{}.png'.format(branch_id)
                )
            )
