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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import redis

# QMLA functionality
import qmla.analysis
import qmla.database_framework as database_framework
import qmla.get_growth_rule as get_growth_rule
import qmla.redis_settings as rds
from qmla.remote_bayes_factor import remote_bayes_factor_calculation
from qmla.remote_model_learning import remote_learn_model_parameters
import qmla.tree

pickle.HIGHEST_PROTOCOL = 4  # if <python3, must use lower protocol
plt.switch_backend('agg')

__all__ = [
    'QuantumModelLearningAgent'
]

class QuantumModelLearningAgent():
    r"""
    QMLA manager class.
    
    Controls the infrastructure which determines which models are learned and compared. 

    :param ControlsQMLA qmla_controls: Storage for configuration of a QMLA instance. 
    :param dict model_priors: values of means/widths to enfore on given models, 
        specifically for further_qhl mode. 
    :param dict experimental_measurements: expectation values by time of the 
        underlying true/target model. 
    :param float sigma_threshold: volume threshold at which to terminate QHL. 
        (Not used by default)

    """

    def __init__(self,
                 qmla_controls=None, 
                 model_priors=None, 
                 experimental_measurements=None,
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
            experimental_measurements=experimental_measurements,
        )

        # Resources potentially reallocated 
        self._compute_base_resources()

        # Redundant attributes, retained for legacy; to be removed
        self._potentially_redundant_setup(
            sigma_threshold=sigma_threshold,
        )

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

        self.qmla_id = self.qmla_controls.qmla_id
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
        self.redis_databases = rds.get_redis_databases_by_qmla_id(
            self.redis_host_name,
            self.redis_port_number,
            self.qmla_id,
        )
        self.redis_databases['any_job_failed'].set('Status', 0)
        self.timings = {
            'inspect_job_crashes' : 0,
            'jobs_finished' : 0
        }
        self.call_counter = {
            'job_crashes' : 0,
            'jobs_finished' : 0, 
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
            qmla.database_framework.get_constituent_names_from_name(self.true_model_name)
        ]
        self.true_model_num_params = self.qmla_controls.true_model_class.num_constituents
        self.true_param_list = self.growth_class.true_params_list
        self.true_param_dict = self.growth_class.true_params_dict
        self.true_model_branch = -1 # overwrite if considered
        self.true_model_considered = False
        self.true_model_found = False
        self.true_model_id = -1
        self.true_model_on_branhces = []
        self.log_print(
            [
                "True model:", self.true_model_name
            ]
        )

    def _setup_tree_and_growth_rules(
        self,
    ):
        r""" Set up infrastructure."""

        # TODO most of this can probably go inside run_complete_qmla
        self.model_database = pd.DataFrame({
            '<Name>': [],
            'Status': [],  
            'Completed': [], 
            'branch_id': [],  
            'Reduced_Model_Class_Instance': [],
            'Operator_Instance': [],
            'Epoch_Start': [],
            'ModelID': [],
        })
        self.model_lists = { 
            # assumes maxmium 13 qubit-models considered
            # to be checked when checking model_lists
            # TODO generalise to max dim of Growth Rule
            j : []
            for j in range(1,13)
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
        self.ghost_branch_list = []
        self.ghost_branches = {}

        # Tree object for each growth rule
        self.trees = {
            gen : qmla.tree.qmla_tree(
                growth_class = self.unique_growth_rule_instances[gen],
                log_file = self.log_file
            )
            for gen in self.unique_growth_rule_instances
        }
        self.branches = {}
        self.tree_count = len(self.trees)
        self.tree_count_completed = np.sum(
            [ tree.is_tree_complete() for tree in self.trees.values()]
        )

    def _set_learning_and_comparison_parameters(
        self,
        model_priors,
        experimental_measurements,
    ):
        r""" Parameters related to learning/comparing models."""

        self.true_model_hamiltonian = self.growth_class.true_hamiltonian
        self.model_priors = model_priors
        self.reallocate_resources = self.qmla_controls.reallocate_resources

        # learning parameters, used by QInfer updates
        self.num_particles = self.qmla_controls.num_particles
        self.num_experiments = self.qmla_controls.num_experiments
        self.num_experiments_for_bayes_updates = self.qmla_controls.num_times_bayes
        self.bayes_threshold_lower = self.qmla_controls.bayes_lower
        self.bayes_threshold_upper = self.qmla_controls.bayes_upper
        self.qinfer_resample_threshold = self.qmla_controls.resample_threshold
        self.qinfer_resampler_a = self.qmla_controls.resample_a
        self.qinfer_PGH_heuristic_factor = self.qmla_controls.pgh_factor
        self.qinfer_PGH_heuristic_exponent = self.qmla_controls.pgh_exponent
        
        # tracking for analysis
        self.model_f_scores = {}
        self.model_precisions = {}
        self.model_sensitivities = {}

        # get probes used for learning
        self.growth_class.generate_probes(
            noise_level=self.qmla_controls.probe_noise_level,
            minimum_tolerable_noise=0.0,
        )
        self.probes_system = self.growth_class.probes_system
        self.probes_simulator = self.growth_class.probes_simulator
        self.probe_number = self.growth_class.num_probes

        self.experimental_measurements = experimental_measurements
        self.experimental_measurement_times = (
            sorted(list(self.experimental_measurements.keys()))
        )
        self.times_to_plot = self.experimental_measurement_times
        self.times_to_plot_reduced_set = self.times_to_plot[0::10]
        self.probes_plot_file = self.qmla_controls.probes_plot_file
        try:
            self.probes_for_plots = pickle.load(
                open(self.probes_plot_file, 'rb')
            )
        except: 
            self.log_print(
                [
                    "Could not load plot probes from {}".format(
                        self.probes_plot_file
                    )
                ]
            )

    def _potentially_redundant_setup(
        self,
        sigma_threshold,
    ):
        r"""
        Graveyard for deprecated ifnrastructure. 

        Attributes etc stored here which are not functionally used
            within QMLA, but which are called somewhere, 
            and cause errors when omitted. 
        Should be stored here temporarily during development, 
            and removed entirely when sure they are not needed. 

        """

        self.sigma_threshold = sigma_threshold
        self.model_fitness_scores = {}
        self.use_time_dependent_true_model = False
        self.num_time_dependent_true_params = 0
        self.time_dependent_params = None
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
        r""" Infrastructure for use when QMLA run in parallel. """

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
        r"""
        Compute the set of minimal resources for models to learn on. 

        In the case self.reallocate_resources==True, 
            models will receive resources (epochs, particles)
            scaled by how complicated they are. 
        For instance, models with 4 parameters will receive
            twice as many particles as a model with 
            2 parameters. 
        """
        # TODO remove base_num_qubits stuff?
        # TODO put option inside growth rule
        # and functionality as method of ModelForLearning
        if self.reallocate_resources:
            base_num_qubits = 3
            base_num_terms = 3
            for op in self.growth_rules.initial_models:
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
        r""" 
        Gather info needed to run QMLA tasks and store remotely. 

        QMLA issues jobs to run remotely, 
            namely for model (parameter) learning 
            and model comparisons (Bayes factors). 
        These jobs don't need access to all QMLA data, 
            but do need some common info,
            e.g. number of particles and epochs. 
        This function gathers all relevant information
            in a single dict, and stores it on the redis server
            which all worker nodes have access to. 
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
            'true_param_dict' : self.true_param_dict, 
            'num_particles': self.num_particles,
            'num_experiments': self.num_experiments,
            'results_directory': self.results_directory,
            'plots_directory': self.qmla_controls.plots_directory,
            'long_id': self.qmla_controls.long_id,
            'prior_specific_terms': self.growth_class.gaussian_prior_means_and_widths,
            'model_priors': self.model_priors,
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
            'num_probes': self.probe_number, # from growth rule or unneeded,
            'true_params_pickle_file' : self.qmla_controls.true_params_pickle_file,
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

    ##########
    # Section: Calculation of models parameters and Bayes factors
    ##########

    def learn_models_on_given_branch(
        self,
        branch_id,
        use_rq=True,
        blocking=False
    ):
        r"""
        Launches jobs to learn all models on the specified branch. 
        
        Models which are on the branch but have already been learned are not re-learned. 
        For each remaining model on the branch, 
            :meth:`QuantumModelLearningAgent.learn_model` is called. 
        The branch is added to the redis database `active_branches_learning_models`, 
            indicating that branch_id has currently got models in the learning phase. 
            This redis database is monitored by the :meth:`QuantumModelLearningAgent.learn_models_until_trees_complete`.
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
        active_branches_learning_models = (
            self.redis_databases['active_branches_learning_models']
        )

        self.log_print(
            [
                "Setting active branch learning rdb for ", branch_id
            ]
        )
        active_branches_learning_models.set(
            int(branch_id),
            num_models_already_set_this_branch
        )
        unlearned_models_this_branch = self.branches[branch_id].unlearned_models

        self.log_print([
            "branch {} has models: \nprecomputed: {} \nunlearned: {}".format(
                branch_id,
                self.branches[branch_id].precomputed_models,
                unlearned_models_this_branch
            )
        ])
        if len(unlearned_models_this_branch) == 0:
            self.ghost_branch_list.append(branch_id)

        for model_name in unlearned_models_this_branch:
            self.log_print(
                [
                    "Model {} being passed to learnModel function".format(
                        model_name
                    )
                ]
            )
            self.learn_model(
                model_name=model_name,
                branch_id = branch_id,
                use_rq=self.use_rq,
                blocking=blocking
            )
            if blocking:
                self.log_print(
                    [
                        "Blocking on; model finished:",
                        model_name
                    ]
                )
            self._update_model_record(
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
        branch_id,
        use_rq=True,
        blocking=False
    ):
        r"""
        Learn a given model by calling the standalone model learning functionality. 

        The model is learned by launching a job either locally or to the job queue. 
        Model learning is implemented by 
            :func:`remote_learn_model_parameters`,
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
        if model_already_exists:
            model_id = database_framework.model_id_from_name(
                self.model_database,
                name=model_name
            )
            if model_id not in self.models_learned: 
                self.models_learned.append(model_id)

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

                self.log_print(
                    [
                        "Model",
                        model_name,
                        "added to queue."
                    ]
                )
                if blocking:  # i.e. wait for result when called.
                    self.log_print(
                        [
                            "Blocking, ie waiting for",
                            model_name,
                            "to finish on redis queue."
                        ]
                    )
                    while not queued_model.is_finished:
                        t_init = time.time()
                        some_job_failed = queued_model.is_failed
                        self.timings['jobs_finished'] += time.time() - t_init
                        self.call_counter['jobs_finished'] += 1
                        if some_job_failed:
                            self.log_print(
                                [
                                    "Model", model_name,
                                    "has failed on remote worker."
                                ]
                            )
                            raise NameError("Remote QML failure")
                            break
                        time.sleep(self.sleep_duration)
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
                    growth_generator = self.branches[branch_id].growth_rule, 
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
        wait_on_result=False
    ):

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
                times_record=self.bayes_factors_store_times_file,
                bf_data_folder=self.bayes_factors_store_directory,
                num_times_to_use=self.num_experiments_for_bayes_updates,
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
                while not job.is_finished:
                    if job.is_failed:
                        raise("Remote BF failure")
                    sleep(self.sleep_duration)
            elif return_job == True:
                return job
        else:
            remote_bayes_factor_calculation(
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                bf_data_folder=self.bayes_factors_store_directory,
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

    def get_bayes_factors_from_list(
        self,
        model_id_list=None,
        pair_list=None,
        remote=True,
        wait_on_result=False,
        recompute=False,
    ):
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
                pair[0], pair[1])
            if (
                unique_id not in self.bayes_factor_pair_computed
                or recompute == True
            ):
                # ie not yet considered
                self.log_print(
                    [
                        "Getting BF from list"
                    ]
                )
                remote_jobs.append(
                    self.get_pairwise_bayes_factor(
                        pair[0],
                        pair[1],
                        remote=remote,
                        return_job=wait_on_result,
                    )
                )

        if wait_on_result and self.use_rq:
            self.log_print(
                [
                    "Waiting on result of ",
                    "Bayes comparisons from given model list:",
                    model_id_list,
                    "\n pair list:", pair_list

                ]
            )
            for job in remote_jobs:
                while not job.is_finished:
                    if job.is_failed:
                        raise NameError("Remote QML failure")
                    time.sleep(self.sleep_duration)
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
        pair_list=None,
        remote=True,
        recompute=False
    ):
        if pair_list is None:
            pair_list = self.branches[branch_id].pairs_to_compare
        active_branches_bayes = self.redis_databases['active_branches_bayes']
        self.log_print(
            [
                'get_bayes_factors_by_branch_id',
                branch_id,
                'pair_list:',
                pair_list
            ]
        )

        active_branches_bayes.set(int(branch_id), 0)  # set up branch 0
        for pair in pair_list:
            a = pair[0]
            b = pair[1]
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
                            unique_id,
                            " on branch ", branch_id
                        ]
                    )
                    self.get_pairwise_bayes_factor(
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

    def process_remote_bayes_factor(
        self,
        a=None,
        b=None,
        pair=None,
    ):
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

        # Tell growth rule's rating system about this comparison
        self.growth_class.ratings_class.compute_new_ratings(
            model_a_id = mod_low.model_id, 
            model_b_id = mod_high.model_id, 
            winner_id = champ
        )

        return champ

    def compare_all_models_in_branch(
        self,
        branch_id,
        pair_list = None
    ):      
        if pair_list is None:
            pair_list = self.branches[branch_id].pairs_to_compare
            self.log_print([
                "Pair list not given for branch {}, generated:{}".format(
                    branch_id, 
                    pair_list
                ),
            ])
            active_models_in_branch = self.branches[branch_id].resident_model_ids

        models_points = {
            k : 0
            for k in active_models_in_branch
        }
        for pair in pair_list: 
            mod1 = pair[0]
            mod2 = pair[1]
            if mod1 != mod2:
                res = self.process_remote_bayes_factor(a=mod1, b=mod2)
                try:
                    models_points[res] += 1
                except:
                    models_points[res] = 1
                self.log_print(
                    [
                        "[branch {} comparison {}/{}] ".format(branch_id, mod1, mod2),
                        "Point to", res,
                    ]
                )

        max_points = max(models_points.values())
        max_points_branches = [
            key for key, val in models_points.items()
            if val == max_points
        ]

        if len(max_points_branches) > 1:
            # TODO ensure multiple models with same # wins
            # doesn't crash
            # e.g. take top two, and select the pairwise winner
            # as champ_id
            self.log_print(
                [
                    "This may cause it to crash:",
                    "Multiple models have same number of points within \
                    branch.\n",
                    models_points
                ]
            )
            self.get_bayes_factors_from_list(
                model_id_list=max_points_branches,
                remote=True,
                recompute=False,
                wait_on_result=True
            )

            champ_id = self.compare_models_from_list(
                max_points_branches,
                models_points_dict=models_points
            )
        else:
            champ_id = max(models_points, key=models_points.get)

        champ_id = int(champ_id)
        champ_name = self.model_name_id_map[champ_id]
        champ_num_qubits = database_framework.get_num_qubits(champ_name)
        # self.branches[branch_id].update_comparisons(
        #     models_points = models_points,
        # )

        for model_id in active_models_in_branch:
            self._update_model_record(
                model_id=model_id,
                field='Status',
                new_value='Deactivated'
            )

        self._update_model_record(
            name=champ_name,
            field='Status',
            new_value='Active'
        )
        ranked_model_list = sorted(
            models_points,
            key=models_points.get,
            reverse=True
        )

        self.branches[branch_id].champion_id = champ_id
        self.branches[branch_id].champion_name = champ_name
        self.branches[branch_id].rankings = ranked_model_list
        self.branches[branch_id].bayes_points = models_points
        self.log_print(
            [
                "Model points for branch {}: {}".format(
                    branch_id,
                    models_points,
                ),
                "\nChampion of branch {} is model {}: {}".format(
                    branch_id, 
                    champ_id, 
                    champ_name
                )
            ]
        )

        if self.branches[branch_id].is_ghost_branch:
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
                    self._update_model_record(
                        model_id=losing_model_id,
                        field='Status',
                        new_value='Deactivated'
                    )
                except BaseException:
                    self.log_print(
                        [
                            "not deactivating",
                            losing_model_id,
                        ]
                    )
        return models_points, champ_id

    def compare_models_from_list(
        self,
        model_list,
        models_points_dict=None,
        num_times_to_use='all'
    ):
        r"""
        Never called in practice; legacy.

        """
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
                recompute=True, # recompute here b/c deadlock last time
                wait_on_result=True
            )
            champ_id = self.compare_models_from_list(
                max_points_branches,
            )
        else:
            self.log_print(["After comparing list:", models_points])
            champ_id = max(models_points, key=models_points.get)
        
        champ_name = self.model_name_id_map[champ_id]
        return champ_id


    ##########
    # Section: routines to implement tree-based QMLA 
    ##########

    def learn_models_until_trees_complete(
        self, 
    ):
        active_branches_learning_models = (
            self.redis_databases['active_branches_learning_models']
        )
        active_branches_bayes = self.redis_databases['active_branches_bayes']

        self.log_print(
            [
                "Starting learning for initial branches:", 
                list(self.branches.keys())
            ]
        )
        for b in self.branches:
            self.learn_models_on_given_branch(
                b,
                blocking=False,
                use_rq=True
            )

        self.log_print(
            [
                "Entering while loop; spawning model layers and comparing."
            ]
        )
        while self.tree_count_completed < self.tree_count:
            branch_ids_on_db = list(
                active_branches_learning_models.keys()
            )
            branch_ids_on_db = [
                int(b) for b in branch_ids_on_db
            ]
            self._inspect_remote_job_crashes()
            for branch_id in branch_ids_on_db:
                if (
                    int(
                        active_branches_learning_models.get(
                            branch_id)
                    ) == self.branches[branch_id].num_models
                    and
                    self.branches[branch_id].model_learning_complete == False
                ):
                    self.log_print([
                        "All models on branch {} learned".format(branch_id)
                    ])
                    self.branches[branch_id].model_learning_complete = True
                    for mod_id in self.branches[branch_id].resident_model_ids:
                        mod = self.get_model_storage_instance_by_id(mod_id)
                        mod.model_update_learned_values()
                    self.get_bayes_factors_by_branch_id(branch_id)

            for branchID_bytes in active_branches_bayes.keys():
                branch_id = int(branchID_bytes)
                bayes_calculated = active_branches_bayes.get(
                    branchID_bytes
                ) # how many completed and stored on redis db

                if (
                    int(bayes_calculated) == self.branches[branch_id].num_model_pairs
                    and
                    self.branches[branch_id].comparisons_complete == False
                ):
                    self.branches[branch_id].comparisons_complete = True
                    self.compare_all_models_in_branch(branch_id)
                    self.log_print(
                        [
                            "Branch {} comparisons complete".format(
                                branch_id
                            )
                        ]
                    )
                    if self.branches[branch_id].tree.is_tree_complete():
                        self.tree_count_completed += 1
                        self.log_print(
                            [
                                "Tree complete:", 
                                self.branches[branch_id].growth_rule, 
                                "Number of trees now completed:",
                                self.tree_count_completed,
                            ]
                        )
                    else: 
                        self.spawn_from_branch(
                            branch_id=branch_id,
                            growth_rule=self.branches[branch_id].tree.growth_rule,
                            num_models=1
                        )

        self.log_print([
                "{} trees have completed. Waiting on final comparisons".format(
                    self.tree_count_completed
                )
        ])

        # allow any branches which have just started to finish 
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
                    self.get_bayes_factors_by_branch_id(branch_id)

                if branchID_bytes in active_branches_bayes:
                    num_bayes_done_on_branch = (
                        active_branches_bayes.get(branchID_bytes)
                    )
                    if (
                        (
                            int(num_bayes_done_on_branch)
                            == self.branches[branch_id].num_model_pairs
                        )
                        and
                        (
                            self.branches[branch_id].comparisons_complete == False    
                        )
                    ):
                        self.branches[branch_id].comparisons_complete =  True
                        self.compare_all_models_in_branch(branch_id)

            if (
                np.all(
                    np.array(
                        [
                            self.branches[b].model_learning_complete
                            for b in self.branches
                        ]
                    ) == True
                )
                and
                np.all(
                    np.array(
                        [self.branches[b].comparisons_complete for b in self.branches]
                    ) == True
                )
            ):
                still_learning = False  # i.e. break out of this while loop
        self.log_print([
            "Learning stage complete on all trees."
        ])

    def spawn_from_branch(
        self,
        branch_id,
        growth_rule,
        num_models=1
    ):

        all_models_this_branch = self.branches[branch_id].rankings
        best_models = all_models_this_branch[:num_models]

        self.log_print([
            "Model rankings on branch {}: {}".format(branch_id, all_models_this_branch),
            "Best models:", best_models
        ])

        best_model_names = [
            self.model_name_id_map[mod_id]
            for mod_id in best_models
        ]
        evaluation_log_likelihoods = {
            mod : 
            self.get_model_storage_instance_by_id(mod).evaluation_log_likelihood
            for mod in all_models_this_branch
        }
        self.log_print(
            [
                "Before model generation, evaluation log likelihoods:", 
                evaluation_log_likelihoods
            ]            
        )
        self.log_print([
            "Generating from top models:", 
            best_model_names,
            "\nAll:", all_models_this_branch
        ])
        new_models, pairs_to_compare = self.branches[branch_id].tree.next_layer(
            model_list=best_model_names,
            log_file=self.log_file,
            branch_model_points = self.branches[branch_id].bayes_points,
            model_names_ids=self.model_name_id_map,
            evaluation_log_likelihoods = evaluation_log_likelihoods, 
            model_dict=self.model_lists,
            called_by_branch=branch_id, 
        )

        self.log_print(
            [
                "After model generation for GR",
                self.branches[branch_id].growth_rule,
                "\nnew models:",
                new_models,
                "pairs to compare:", pairs_to_compare,
            ]
        )

        # Generate new QMLA level branch, and launch learning for those models
        new_branch_id = self.new_branch(
            model_list = new_models,
            pairs_to_compare = pairs_to_compare, 
            growth_rule = growth_rule,
            spawning_branch = branch_id, 
        )
        self.log_print([
            "Models to add to new branch {}: ".format(
                new_branch_id,
                new_models
            )
        ])

        # Learn models on the new branch
        self.learn_models_on_given_branch(
            new_branch_id,
            blocking=False,
            use_rq=True
        )

    def new_branch(
        self,
        model_list,
        pairs_to_compare='all', 
        growth_rule=None,
        spawning_branch = 0,
    ):
        r"""
        Add a set of models to a new QMLA branch. 

        Branches have a unique id within QMLA, but belong to a single 
            tree, where each tree corresponds to a single growth rule. 

        :param list model_list: strings corresponding to models to 
            place in the branch
        :param pairs_to_compare: set of model pairs to perform Bayes comparisons on
            by default all pairs are compared, but for pruning branches a list of 
            model pairs can be specified instead, by tuples of model ids. 
        :type pairs_to_compare: str or list
        :param str growth_rule: growth rule identifer; 
            used to get the unique tree object corresponding to a growth rule, 
            which is then used to host the branch. 
        :param int spawning_branch: branch id which is the parent of the new branch. 
        :return int branch_id: branch id which uniquely identifies the new branch 
            within the QMLA environment. 

        """

        model_list = list(set(model_list))  # remove possible duplicates
        self.branch_highest_id += 1
        branch_id = int(self.branch_highest_id)
        self.log_print(
            [
                "NEW BRANCH {}. growth rule= {}".format(branch_id, growth_rule)
            ]
        )

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
            if already_computed == False:  # first instance of this model
                self.models_branches[model_id] = branch_id

            self.log_print(
                [
                    'Model {} computed already: {} -> ID {}'.format(
                        model, 
                        already_computed,
                        model_id, 
                        ),
                ]
            )
            if bool(already_computed):
                pre_computed_models.append(model)
        
        model_instances = {
            m: self.get_model_storage_instance_by_id(m)
            for m in list(this_branch_models.keys())
        }

        if growth_rule is None:
            # placing on true growth rule's tree...
            growth_rule = self.growth_rule_of_true_model
        growth_tree = self.trees[growth_rule]
        self.branches[branch_id] = growth_tree.new_branch_on_tree(
            branch_id = branch_id, 
            models = this_branch_models, 
            pairs_to_compare = pairs_to_compare, 
            model_instances = model_instances, 
            precomputed_models = pre_computed_models,
            spawning_branch = spawning_branch, 
        )
        return branch_id

    def add_model_to_database(
        self,
        model,
        branch_id=-1,
        force_create_model=False
    ):
        r"""
        Considers adding a model to QMLA's database of model.

        Checks whether the nominated model is already present; 
            if not generates a model instance; 
            stores pertinent details in the running database. 

        :param str model: name of model to consider
        :param float branch_id: branch id to associate this model with, 
            if the model is new. 
        :param bool force_create_model: 
            True: add model even if the name is found already. 
        """

        model_name = database_framework.alph(model)
        self.log_print([
            "Trying to add model to DB:", model_name
        ])

        if (
            qmla.database_framework.consider_new_model(
                self.model_lists, model_name, self.model_database) == 'New'
            or
            force_create_model == True
        ):
            model_num_qubits = qmla.database_framework.get_num_qubits(model_name)
            model_id = self.highest_model_id + 1
            self.model_lists[model_num_qubits].append(model_name)

            self.log_print(
                [
                    "Model ", model_name,
                    "not previously considered -- adding.",
                    "ID:", model_id
                ]
            )
            op = qmla.database_framework.Operator(
                name=model_name, undimensionalised_name=model_name
            )
            reduced_qml_instance = qmla.model_instances.ModelInstanceForStorage(
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
            running_db_new_row = pd.Series({
                '<Name>': model_name,
                'Status': 'Active',
                'Completed': False,
                'branch_id': int(branch_id),
                'Reduced_Model_Class_Instance': reduced_qml_instance,
                'Operator_Instance': op,
                'Epoch_Start': 0, 
                'ModelID': int(model_id),
            })
            # add to the database
            num_rows = len(self.model_database)
            self.model_database.loc[num_rows] = running_db_new_row
            model_added = True
            if database_framework.alph(
                    model) == database_framework.alph(self.true_model_name):
                self.true_model_id = model_id
                self.true_model_considered = True
                self.true_model_branch = branch_id
                self.true_model_on_branhces = [branch_id]
            self.highest_model_id = model_id
            self.model_name_id_map[model_id] = model_name
            self.model_count += 1

        else:
            model_added = False
            self.log_print([
                "Model not added: {}".format(
                    model_name
                )
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
        # Final functions at end of QMLA instance

        champ_model = self.get_model_storage_instance_by_id(
            self.champion_model_id)
        
        # Get metrics for all models tested
        for i in self.models_learned:
            # Dict of all Bayes factors for each model considered.
            self.all_bayes_factors[i] = (
                self.get_model_storage_instance_by_id(i).model_bayes_factors
            )
            self.compute_model_f_score(i)
        self.get_statistical_metrics()

        # Prepare model/name maps
        self.model_id_to_name_map = {}
        for k in self.model_name_id_map:
            v = self.model_name_id_map[k]
            self.model_id_to_name_map[v] = k

    def get_results_dict(
        self, 
        model_id=None
    ):
        if model_id is None:
            model_id = self.champion_model_id

        mod = self.get_model_storage_instance_by_id(model_id)
        model_name = mod.model_name

        # Get expectation values of this model
        n_qubits = database_framework.get_num_qubits(model_name)
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

        # Perform final steps in GrowthRule
        self.growth_class.growth_rule_finalise() # TODO put in main QMLA/QHL method

        # Evaluate all models in this instance
        model_evaluation_log_likelihoods = {
            mod_id : self.get_model_storage_instance_by_id(mod_id).evaluation_log_likelihood
            for mod_id in self.models_learned
        }
        model_evaluation_median_likelihoods = {
            mod_id : self.get_model_storage_instance_by_id(mod_id).evaluation_median_likelihood
            for mod_id in self.models_learned
        }

        # Compare this model to the true model (only meaningful for simulated cases)
        correct_model = misfit = underfit = overfit = 0
        model_operator = database_framework.Operator(model_name)
        num_params_champ_model = model_operator.num_constituents

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
            'Host' : self.redis_host_name, 
            'Port' : self.redis_port_number, 
            # details about true model:
            'TrueModel' : self.true_model_name,
            'TrueModelConsidered' : self.true_model_considered, 
            'TrueModelFound' : self.true_model_found,
            'TrueModelBranch' : self.true_model_branch,
            'TrueModelID' : self.true_model_id, 
            'TrueModelConstituentTerms' : self.true_model_constituent_terms_latex, 
            # details about this model 
            'ChampID': model_id,
            'ChampLatex': mod.model_name_latex,
            'ConstituentTerms' : mod.constituents_terms_latex,
            'LearnedHamiltonian': mod.learned_hamiltonian,
            'GrowthGenerator': mod.growth_rule_of_this_model,
            'NameAlphabetical': database_framework.alph(mod.model_name),
            'LearnedParameters': mod.learned_parameters_qhl,
            'FinalSigmas': mod.final_sigmas_qhl,
            'ExpectationValues': mod.expectation_values,
            'Trackplot_parameter_estimates': mod.track_parameter_estimates,
            'TrackVolume': mod.volume_by_epoch,
            'TrackTimesLearned': mod.times_learned_over,
            'QuadraticLosses': mod.quadratic_losses_record,
            'FinalRSquared': mod.r_squared(
                times=expec_val_plot_times,
                plot_probes=self.probes_for_plots
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
            'NumModels' : len(self.models_learned),
            'StatisticalMetrics' : self.generational_statistical_metrics,
            'GenerationalFscore'  : self.generational_f_score,
            'GenerationalLogLikelihoods' : self.generational_log_likelihoods, 
            'ModelEvaluationLogLikelihoods' : model_evaluation_log_likelihoods,
            'ModelEvaluationMedianLikelihoods' : model_evaluation_median_likelihoods,
            'AllModelFScores' : self.model_f_scores, 
            # data stored during GrowthRule.growth_rule_finalise():
            'GrowthRuleStorageData' : self.growth_class.growth_rule_specific_data_to_store,
        }
        return results_dict

    def check_champion_reducibility(
        self,
    ):
        champ_mod = self.get_model_storage_instance_by_id(
            # self.champion_model_id
            self.global_champion_id
        )

        self.log_print(
            [
                "Checking reducibility of champ model:",
                self.global_champion_name,
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
            p_str = 'P' * dim # TODO  generalise this to tie together terms via GR instead of assuming P strings work
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
            # so that champion does not train further on times it already learned
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

            bayes_factor = self.get_pairwise_bayes_factor(
                model_a_id=int(self.champion_model_id),
                model_b_id=int(reduced_mod_id),
                wait_on_result=True
            )
            self.log_print(
                [
                    "BF b/w champ and reduced champ models:",
                    bayes_factor
                ]
            )

            if (
                (
                    bayes_factor
                    < (1.0 / self.growth_class.reduce_champ_bayes_factor_threshold)
                )
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
                self.global_champion = new_mod
                # inherits BF of champion from which it derived (only for plotting really)
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
            self.log_print(
                [
                    "Parameters non-negligible; not replacing champion model."
                ]
            )

    def compare_nominated_champions(self):
        
        tree_champions = []
        for tree in self.trees.values():
            tree_champions.extend(tree.nominate_champions())

        global_champ_branch_id = self.new_branch(
            model_list = tree_champions
        )
        global_champ_branch = self.branches[
            global_champ_branch_id
        ]

        self.get_bayes_factors_from_list(
            pair_list = global_champ_branch.pairs_to_compare,
            wait_on_result = True, 
        )
        model_points, champ_id = self.compare_all_models_in_branch(
            branch_id = global_champ_branch_id
        )
        self.global_champion_id = champ_id
        self.global_champion_model = self.get_model_storage_instance_by_id(
            self.global_champion_id
        )
        self.global_champion_name = self.global_champion_model.model_name
        self.log_print([
            "Global champion branch points:", model_points, 
            "\nGlobal champion ID:", champ_id,
            "\nGlobal champion:", self.global_champion_name
        ])

    ##########
    # Section: Run available algorithms (QMLA, QHL or QHL with multiple models)
    ##########

    def run_quantum_hamiltonian_learning(
        self,
        force_qhl=False
    ):

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
            branch_id = qhl_branch,
            use_rq=self.use_rq,
            blocking=True
        )

        mod_id = database_framework.model_id_from_name(
            db=self.model_database,
            name=mod_to_learn
        )

        # these don't really matter for QHL
        # but are used in plots etc:
        self.true_model_id = mod_id
        self.champion_model_id = mod_id
        self.true_model_found = True
        self.true_model_considered = True 
        self.log_print(
            [
                "Learned model {}: {}".format(
                    mod_id, 
                    mod_to_learn
                )
            ]
        )
        self._update_database_model_info()
        self.compute_model_f_score(
            model_id=mod_id
        )
        self.get_statistical_metrics()
        for k in self.timings:
            self.log_print([
                "QMLA Timing - {}: {}".format(k, np.round(self.timings[k], 2))
            ])


    def run_quantum_hamiltonian_learning_multiple_models(
            self, model_names=None):
        if model_names is None:
            model_names = self.growth_class.qhl_models

        branch_id = self.new_branch(
            growth_rule=self.growth_rule_of_true_model,
            model_list= model_names
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
            mod_id = database_framework.model_id_from_name(
                db=self.model_database,
                name=mod_name
            )
            learned_models_ids.set(
                str(mod_id), 0
            )
            self.learn_model(
                model_name=mod_name,
                branch_id = branch_id,
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
            # need to wait on all models to finish anyway, 
            # so can just wait on them in order.
            while int(learned_models_ids.get(k)) != 1:
                sleep(self.sleep_duration)
                self._inspect_remote_job_crashes()

        # Learning finished
        self.log_print(
            [
                'Finished learning, for all:', running_models,
            ]
        )

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
        self.get_statistical_metrics()
        self.growth_class.growth_rule_finalise()
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
        # set up one tree per growth rule
        for tree in list(self.trees.values()):
            starting_models = tree.get_initial_models() 
            self.log_print([
                "First branch for {} has starting models: {}".format(
                    tree.growth_rule, starting_models
                ),
            ])
            self.new_branch(
                model_list = starting_models, 
                growth_rule = tree.growth_rule
            )

        # Iteratively learn models, compute bayes factors, spawn new models
        self.learn_models_until_trees_complete()
        self.log_print([
            "Growth rule trees completed."
        ])

        # Choose champion by comparing nominated models (champions) of all trees.
        self.compare_nominated_champions()
        self.champion_model_id = self._get_model_data_by_field(
            name=self.global_champion_name,
            field='ModelID'
        )
        self.log_print(["Champion selected."])

        # internal analysis
        try:            
            if self.global_champion_id == self.true_model_id:
                self.true_model_found = True
            else:
                self.true_model_found = False
        except:
            self.true_model_found = False
        self._update_database_model_info()
        if self.true_model_found:
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

        # Consider reducing champion if negligible parameters found
        if self.growth_class.check_champion_reducibility:
            self.check_champion_reducibility()

        # tidy up and finish QMLA. 
        self.finalise_qmla()
        for k in self.timings:
            self.log_print([
                "QMLA Timing - {}: {}".format(k, np.round(self.timings[k], 2))
            ])
        self.log_print(
            [
                "\nFinal winner:", self.global_champion_name, 
                "has F-score ", self.model_f_scores[self.champion_model_id]
            ]
        )

    ##########
    # Section: Utilities
    ##########

    def log_print(self, to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'QMLA {}'.format(self.qmla_id)

        )

    def get_model_storage_instance_by_id(self, model_id):
        r"""
        Get the unique :class:`qmla.ModelInstanceForLearning` for the given model_id. 

        :param int model_id: unique ID of desired model
        :return model_instance: storage class of the model 
        :rtype: :class:`qmla.ModelInstanceForLearning`

        """
        # return database_framework.reduced_model_instance_from_id(
        #     self.model_database, model_id)
        idx = self.model_database.loc[self.model_database['ModelID'] == model_id].index[0]
        model_instance = self.model_database.loc[idx]["Reduced_Model_Class_Instance"]
        return model_instance

    def _update_database_model_info(self):
        r"""

        """

        self.log_print([
            "Updating info for all learned models"
        ])
        for mod_id in self.models_learned:
            try:
                # TODO remove this try/except when reduced-champ-model instance
                # is update-able
                mod = self.get_model_storage_instance_by_id(mod_id)
                mod.model_update_learned_values(
                    fitness_parameters=self.model_fitness_scores
                )
            except BaseException:
                pass

    def _update_model_record(
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

    def _inspect_remote_job_crashes(self):
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
        r"""
        Remove elements of QMLA which cannot be pickled, which cause errors if retained. 
        """
        
        del self.redis_conn
        del self.rq_queue
        del self.redis_databases
        del self.write_log_file

    def _get_model_data_by_field(self, name, field):
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
        generations = sorted(set(self.branches.keys()))
        generations = [ 
            b for b in self.branches 
            if not self.branches[b].prune_branch
        ]
        self.log_print(
            [
                "[get_statistical_metrics",
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
        self.generational_log_likelihoods = {
            b : []
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
                    self.get_model_storage_instance_by_id(m).evaluation_log_likelihood
                )

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
        qmla.analysis.update_shared_bayes_factor_csv(self, bayes_csv)

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
                                            #    use_experimental_data=self.use_experimental_data,
                                               save_to_file=save_to_file
                                               )

    def plot_branch_champions_dynamics(
        self,
        all_models=False,
        model_ids=None,
        save_to_file=None,
    ):
        include_params = False
        include_bayes_factors = False
        if all_models == True:
            model_ids = list(sorted(self.model_name_id_map.keys()))
        elif self.qhl_mode:
            model_ids = [self.true_model_id]
            include_params = True
        elif self.qhl_mode_multiple_models:
            model_ids = list(self.qhl_mode_multiple_models_model_ids)
        elif self.growth_class.tree_completed_initially:
            model_ids = list(self.models_learned)
            include_bayes_factors=True
            include_params=True
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

        qmla.analysis.plot_learned_models_dynamics(
            qmd=self,
            include_bayes_factors=include_bayes_factors,
            include_times_learned=True,
            include_param_estimates=include_params,
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

    def plot_one_qubit_probes_bloch_sphere(self):
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
        bloch.show()
