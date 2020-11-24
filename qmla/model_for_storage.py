import numpy as np
import scipy as sp
import os
import time
import copy
import qinfer as qi

import redis
import pickle

import qmla.redis_settings
import qmla.memory_tests
import qmla.logging
import qmla.get_exploration_strategy
import qmla.shared_functionality.experimental_data_processing
import qmla.construct_models
import qmla.analysis
import qmla.process_string_to_matrix

pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    'ModelInstanceForStorage'
]


class ModelInstanceForStorage():
    r"""
    Model stored in QMLA environment.

    Retrieves data after model is trained remotely, so that
    :class:`qmla.QuantumModelLearningAgent` can access that data.

    :param str model_name: name of model under study
    :param int model_id: ID of model which is unique to QMLA instance
    :param np.array() model_terms_matrices:
        list of matrices corresponding to the operators which compose the model
    :param dict plot_probes: probes used in all plots for consistency
    :param dict qmla_core_info_database: essential details about the QMLA
        instance needed to learn/compare models.
        If None, this is retrieved instead from the redis database.
    :param str host_name: name of host server on which redis database exists.
    :param int port_number: port number unique to this QMLA instance on redis database
    :param str log_file: path of QMLA instance's log file.
    """

    def __init__(
        self,
        model_name,
        model_id,
        model_terms_matrices,
        qid,
        plot_probes=None,
        qmla_core_info_database=None,
        host_name='localhost',
        port_number=6379,
        log_file='QMD_log.log',
        **kwargs
    ):
        # Basic info about this QMLA instance and model
        self.qmla_id = qid
        self.model_name = qmla.construct_models.alph(model_name)
        self.model_id = model_id
        self.model_terms_matrices = model_terms_matrices
        self.num_terms = len(self.model_terms_matrices)
        self.log_file = log_file

        # Redis database settings
        self.redis_host_name = host_name
        self.redis_port_number = port_number

        # Get data from redis database
        if qmla_core_info_database is None:
            redis_databases = qmla.redis_settings.get_redis_databases_by_qmla_id(
                self.redis_host_name,
                self.redis_port_number,
                self.qmla_id
            )
            qmla_core_info_database = redis_databases['qmla_core_info_database']
            self.probes_system = pickle.loads(
                qmla_core_info_database['probes_system'])
            self.probes_simulator = pickle.loads(
                qmla_core_info_database['probes_simulator'])
            qmla_core_info_dict = pickle.loads(
                qmla_core_info_database.get('qmla_settings'))

        else:
            self.log_print(
                [
                    'QMLA core info provided to model storage class w/ keys:',
                    list(qmla_core_info_database.keys())
                ]
            )
            self.probes_system = qmla_core_info_database['probes_system']
            self.probes_simulator = qmla_core_info_database['probes_simulator']
            qmla_core_info_dict = qmla_core_info_database.get('qmla_settings')

        # Extract data from core QMLA database
        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        self.true_model_name = qmla_core_info_dict['true_name']
        if self.model_name == self.true_model_name:
            self.is_true_model = True
            self.log_print(["This is the true model for storage."])
        else:
            self.is_true_model = False
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.probe_number = qmla_core_info_dict['num_probes']
        self.experimental_measurement_times = qmla_core_info_dict['experimental_measurement_times']
        if plot_probes is None:
            self.probes_for_plots = pickle.load(
                open(qmla_core_info_dict['probes_plot_file'], 'rb')
            )
        else:
            self.probes_for_plots = plot_probes

        # Parameters used by QMLA manager class
        self.model_bayes_factors = {}
        # self.model_num_qubits = qmla.construct_models.get_num_qubits(
        #     self.model_name)
        # self.probe_num_qubits = self.model_num_qubits
        self.model_num_qubits = np.log2( np.shape(self.model_terms_matrices[0])[0] )
        self.probe_num_qubits = self.model_num_qubits
        self.expectation_values = {}
        self.values_updated = False

    ##########
    # Section: Update model based on learned values
    ##########

    def model_update_learned_values(
        self,
        learned_info=None,
        **kwargs
    ):
        """
        Get result of model learning and store within this object.

        Every element stored by :meth:`~qmla.ModelInstanceForLearning.learned_info_dict`
        is stored as an attribute here.

        :param dict learned_info: results of remote model learning
            if None, retrieved from the redis database
            if not None, computed locally and passed
        """

        if self.values_updated:
            return

        self.values_updated = True
        num_redis_retries = 5
        for k in range(num_redis_retries):
            try:
                redis_databases = qmla.redis_settings.get_redis_databases_by_qmla_id(
                    self.redis_host_name,
                    self.redis_port_number,
                    self.qmla_id
                )
                break
            except Exception as e:
                if k == num_redis_retries-1:
                    log_print([
                        "Failed to retrieve redis databases. Error: ", e
                    ])
                    any_job_failed_db.set('Status', 1)
                    raise

        for k in range(num_redis_retries):
            try:
                learned_models_info_db = redis_databases['learned_models_info_db']
                break
            except Exception as e:
                if k == num_redis_retries-1:
                    log_print([
                        "Failed to retrieve model stored data. Error: ", e
                    ])
                    any_job_failed_db.set('Status', 1)
                    raise

        self.log_print([
            "Updating learned info for model {}".format(self.model_id)
        ])

        if learned_info is None:
            # TODO put unloading redis inside this if statement
            # everything can be done locally if learned_info is provided
            model_id_float = float(self.model_id)
            model_id_str = str(model_id_float)
            for k in range(num_redis_retries):
                try:
                    learned_info = pickle.loads(
                        learned_models_info_db.get(model_id_str),
                        encoding='latin1'
                    )
                    break
                except Exception as e:
                    if k  == num_redis_retries -1 :
                        self.log_print([
                            "Unable to load learned info",
                            "model_id_str: ", model_id_str,
                            "model id: ", self.model_id,
                            "learned info keys:, ", learned_models_info_db.keys(),
                            # "learned info:, ", learned_models_info_db.get(
                            #     model_id_str)
                        ])

        # Load results: assign attribute of this class for everything stored
        # in learned_info_dict() of ModelInstanceForStorage.
        for k in learned_info:
            self.__setattr__(k, learned_info[k])

        # Process the learned info
        self.track_covariance_matrices = np.array(
            self.track_covariance_matrices)
        self.volume_by_epoch = {}
        for i in range(len(self.raw_volume_list)):
            self.volume_by_epoch[i] = self.raw_volume_list[i]

        # Instantiate exploration strategy instance (passive - not used to generate
        # models)
        self.exploration_class = qmla.get_exploration_strategy.get_exploration_class(
            exploration_rules=self.exploration_strategy_of_this_model,
            log_file=self.log_file,
            qmla_id=self.qmla_id, 
        )

        # Compile some attributes
        self.model_name_latex = self.exploration_class.latex_name(
            name=self.model_name
        )
        model_constituent_terms = qmla.construct_models.get_constituent_names_from_name(
            self.model_name
        )
        self.constituents_terms_latex = [
            self.exploration_class.latex_name(term)
            for term in model_constituent_terms
        ]
        self.track_parameter_estimates = self.track_param_estimate_v_epoch

        # Learned model loaded
        self.log_print([
            "Updated learned info for model {}".format(self.model_id),
        ])

    ##########
    # Section: Evaluation
    ##########

    def compute_expectation_values(
        self,
        times=[],
    ):
        r"""
        Get the expectation values using the learned Hamiltonian.

        Construct Hamiltonian from estimated learned parameters,
        and compute the expectation values, using the same input
        state as used for plotting.
        Stores a dictionary of { t : expectation value }.

        :param list times: times to use
        """

        # Choose probe to compute expectation value with
        probe = self.probes_for_plots[self.probe_num_qubits]

        # Find which times are not yet in self.expectation_values
        present_expec_val_times = sorted(
            list(self.expectation_values.keys())
        )
        required_times = sorted(
            list(set(times) - set(present_expec_val_times))
        )

        # Compute and store results.
        for t in required_times:
            self.expectation_values[t] = self.exploration_class.get_expectation_value(
                ham=self.learned_hamiltonian,
                t=t,
                state=probe,
                log_file=self.log_file,
                log_identifier='[QML - compute expectation values]'
            )

    def r_squared(
        self,
        times=None,
        min_time=0,
        max_time=None
    ):
        r"""
        Compute and store r squared for given times.

        :param list times: times to use for calculation
        :param float min_time: minimum time to use for calculation
        :param float min_time: maximum time to use for calculation
        :return float final_r_squared: r squared of the learned model against the times given
        """

        self.log_print([
            "R squared function for", self.model_name
        ])

        # Choose times to get r squared for
        if times is None:
            exp_times = sorted(list(self.experimental_measurements.keys()))
        else:
            exp_times = times

        if max_time is None:
            max_time = max(exp_times)

        min_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(
            exp_times, min_time)
        max_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(
            exp_times, max_time)
        min_data_idx = exp_times.index(min_time)
        max_data_idx = exp_times.index(max_time)
        exp_times = exp_times[min_data_idx:max_data_idx]

        # Get expectation values for system
        exp_data = [
            self.experimental_measurements[t] for t in exp_times
        ]

        # Compute r squared
        probe = self.probes_for_plots[self.probe_num_qubits]

        datamean = np.mean(exp_data[0:max_data_idx])
        total_sum_of_squares = 0
        for d in exp_data:
            total_sum_of_squares += (d - datamean)**2
        self.true_exp_val_mean = datamean
        self.total_sum_of_squares = total_sum_of_squares

        sum_of_residuals = 0
        available_expectation_values = sorted(
            list(self.expectation_values.keys()))

        chi_squared = 0
        self.r_squared_of_t = {}
        for t in exp_times:
            if t in available_expectation_values:
                sim = self.expectation_values[t]
            else:
                sim = self.exploration_class.get_expectation_value(
                    ham=self.learned_hamiltonian,
                    t=t,
                    state=probe
                )
                self.expectation_values[t] = sim

            true = self.experimental_measurements[t]
            diff_squared = (true - sim)**2
            sum_of_residuals += diff_squared
            self.r_squared_of_t[t] = 1 - \
                (sum_of_residuals / total_sum_of_squares)
            chi_squared += diff_squared / true

        if total_sum_of_squares == 0:
            # calculation failed
            print(
                "[ModelForStorage - r_squared]",
                "Total sum of squares is 0",
                total_sum_of_squares,
                "\ndatamean=", datamean,
                "\nd=", d,
                "\nexp_data=", exp_data
            )

        try:
            self.final_r_squared = 1 - \
                (sum_of_residuals / total_sum_of_squares)
        except BaseException:
            self.final_r_squared = None
        self.p_value = 0
        # self.p_value = (
        #     1 -
        #     sp.stats.chi2.cdf(
        #         chi_squared,
        #         len(exp_times) - 1  # number of degrees of freedom
        #     )
        # )

        return self.final_r_squared

    def r_squared_by_epoch(
        self,
        times=None,
        min_time=0,
        max_time=None,
        num_points=10
    ):
        r"""
        Compute and store r squared up to all times.
        
        TODO incorporate as flag in r_squared() to store by epoch
        instead of separate fnc.
        """

        self.log_print([
            "R squared by epoch function for",
            self.model_name,
            "Times passed:",
            times
        ])

        # Choose times to get R-squared for
        if times is None:
            exp_times = sorted(list(self.experimental_measurements.keys()))
        else:
            exp_times = times

        if max_time is None:
            max_time = max(exp_times)
        min_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(
            exp_times,
            min_time
        )
        max_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(
            exp_times,
            max_time
        )
        min_data_idx = exp_times.index(min_time)
        max_data_idx = exp_times.index(max_time)
        exp_times = exp_times[min_data_idx:max_data_idx]

        # Get expectation values for system
        exp_data = [
            self.experimental_measurements[t]
            for t in exp_times
        ]

        # Compute r squared
        probe = self.probes_for_plots[self.probe_num_qubits]
        datamean = np.mean(exp_data[0:max_data_idx])
        datavar = np.sum(
            (exp_data[0:max_data_idx] - datamean)**2
        )
        r_squared_by_epoch = {}

        # only use subset of epochs in case there are a large
        # num experiments due to heavy computational overhead
        spaced_epochs = np.round(
            np.linspace(
                0,
                self.num_experiments - 1,
                min(self.num_experiments, num_points))
        )

        for e in spaced_epochs:
            sum_of_residuals = 0
            available_expectation_values = sorted(
                list(self.expectation_values.keys())
            )
            for t in exp_times:
                sim = self.exploration_class.get_expectation_value(
                    ham=self.learned_hamiltonian,
                    t=t,
                    state=probe
                )
                true = self.experimental_measurements[t]
                diff_squared = (sim - true)**2
                sum_of_residuals += diff_squared

            rsq = 1 - sum_of_residuals / datavar
            r_squared_by_epoch[e] = rqs
        self.r_squared_by_epoch = r_squared_by_epoch
        self.final_r_squared = rsq

        return r_squared_by_epoch

    ##########
    # Section: Utilities
    ##########

    def log_print(
        self,
        to_print_list
    ):
        r"""Wrapper for :func:`~qmla.print_to_log`"""
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier='ModelForStorage {}'.format(self.model_id)
        )
