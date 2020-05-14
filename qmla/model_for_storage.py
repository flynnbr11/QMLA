import numpy as np
import scipy as sp
import os
import time
import copy
import qinfer as qi

import redis
import pickle

import qmla.redis_settings as rds
# import qmla.qinfer_model_interface as qml_qi
import qmla.memory_tests
import qmla.logging
import qmla.get_growth_rule as get_growth_rule
import qmla.shared_functionality.experimental_data_processing
import qmla.database_framework
import qmla.analysis

pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    'ModelInstanceForStorage'
]

### Reduced class with only essential information saved ###
class ModelInstanceForStorage():
    """
    Class holds what is required for updates only.
    i.e.
        - times learned over
        - final parameters
        - oplist
        - true_oplist (?) needed to regenerate GenSimModel identically (necessary?)
        - true_model_terms_params (?)
        - resample_thresh
        - resample_a [are resampling params needed only for updates?]
        - Prior (specified by mean and std_dev?)

    Then initialises an updater and GenSimModel which are used for updates.
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
        self.redis_host_name = host_name
        self.redis_port_number = port_number
        self.qmla_id = qid
        self.log_file = log_file
        self.model_name = qmla.database_framework.alph(model_name)
        self.model_id = model_id
        self.model_terms_matrices = model_terms_matrices

        if qmla_core_info_database is None: 
            redis_databases = rds.get_redis_databases_by_qmla_id(
                self.redis_host_name,
                self.redis_port_number,
                self.qmla_id
            )
            qmla_core_info_database = redis_databases['qmla_core_info_database']
            # Get data from redis database which is needed to learn from
            self.probes_system = pickle.loads(qmla_core_info_database['ProbeDict'])
            self.probes_simulator = pickle.loads(qmla_core_info_database['SimProbeDict'])
            qmla_core_info_dict = pickle.loads(qmla_core_info_database.get('qmla_settings'))

        else: 
            self.log_print(
                [
                    'QMLA core info provided to model storage class w/ keys:',
                    list(qmla_core_info_database.keys())
                ]
            )
            self.probes_system = qmla_core_info_database['ProbeDict']
            self.probes_simulator = qmla_core_info_database['SimProbeDict']
            qmla_core_info_dict = qmla_core_info_database.get('qmla_settings')

        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        # self.use_experimental_data = qmla_core_info_dict['use_experimental_data']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        self.true_model_name = qmla_core_info_dict['true_name']
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.probe_number = qmla_core_info_dict['num_probes']
        self.experimental_measurement_times = qmla_core_info_dict['experimental_measurement_times']
        self.qinfer_resampler_threshold = qmla_core_info_dict['resampler_thresh']
        self.qinfer_resampler_a = qmla_core_info_dict['resampler_a']

        if plot_probes is None: 
            self.probes_for_plots = pickle.load(
                open(qmla_core_info_dict['probes_plot_file'], 'rb')
            )
        else: 
            self.probes_for_plots = plot_probes 

        self.store_particle_locations_and_weights = qmla_core_info_dict[
            'store_particles_weights'
        ]

        # define parameters used by qmla class
        self.model_bayes_factors = {}
        self.model_num_qubits = qmla.database_framework.get_num_qubits(
            self.model_name)
        self.probe_num_qubits = self.model_num_qubits
        
        self.expectation_values = {}
        self.values_updated = False

    def log_print(
        self,
        to_print_list
    ):
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier='ModelForStorage {}'.format(self.model_id)
        )

    def model_update_learned_values(
        self,
        learned_info=None,
        **kwargs
    ):
        """
        Pass a dict, learned_info, with essential info on
        reconstructing the state of the model, updater and GenSimModel

        """
        if not self.values_updated:
            self.values_updated = True
            redis_databases = rds.get_redis_databases_by_qmla_id(
                self.redis_host_name,
                self.redis_port_number,
                self.qmla_id
            )
            learned_models_info_db = redis_databases['learned_models_info_db']
            self.log_print(
                [
                    "Updating learned info for model {}".format(self.model_id),
                    "learned info:", learned_info
                ]
            )

            if learned_info is None:
                # TODO put unloading redis inside this if statement
                # everything can be done locally if learned_info is provided
                model_id_float = float(self.model_id)
                model_id_str = str(model_id_float)
                try:
                    learned_info = pickle.loads(
                        learned_models_info_db.get(model_id_str),
                        encoding='latin1'
                    )
                except BaseException:
                    self.log_print(
                        [
                            "Unable to load learned info",
                            "model_id_str: ", model_id_str,
                            "model id: ", self.model_id,
                            "learned info keys:, ", learned_models_info_db.keys(),
                            "learned info:, ", learned_models_info_db.get(
                                model_id_str)
                        ]
                    )


            for k in learned_info:
                self.__setattr__(k, learned_info[k])      
                self.log_print([
                    "Set attr {} ".format(k)
                ])      


            # process the learned info
            self.model_terms_parameters_final = np.array(
                self.final_learned_params
            )
            self.track_param_means = np.array(self.track_param_means)
            self.track_covariance_matrices = np.array(
                self.track_covariance_matrices)
            self.track_param_uncertainties = np.array(
                self.track_param_uncertainties)
            
            self.volume_by_epoch = {}
            for i in range(len(self.raw_volume_list)):
                self.volume_by_epoch[i] = self.raw_volume_list[i]
            
            try:
                self.growth_class = get_growth_rule.get_growth_generator_class(
                    growth_generation_rule=self.growth_rule_of_this_model,
                    # use_experimental_data=self.use_experimental_data,
                    log_file=self.log_file
                )
                self.log_print(["Loaded growth class."])
            except BaseException:
                self.log_print([
                    "Failed to load growth class {} for model".format(
                        self.growth_rule_of_this_model
                    )
                ])
                raise
            self.model_name_latex = self.growth_class.latex_name(
                name=self.model_name
            )
            model_constituent_terms = qmla.database_framework.get_constituent_names_from_name(
                self.model_name
            )
            self.constituents_terms_latex = [
                self.growth_class.latex_name(term)
                for term in model_constituent_terms
            ]

            # match the learned parameters by their name in a dict
            self.track_parameter_estimates = {}
            num_params = np.shape(self.track_param_means)[1]
            max_exp = np.shape(self.track_param_means)[0] - 1
            for i in range(num_params):
                some_final_param = self.track_param_means[max_exp][i]
                for term in self.qhl_final_param_estimates:
                    if self.qhl_final_param_estimates[term] == some_final_param:
                        param_estimate_v_experiments = self.track_param_means[:][i]
                        self.track_parameter_estimates[term] = param_estimate_v_experiments
            
            sim_params = list(self.final_learned_params[:, 0])
            try:
                self.learned_hamiltonian = np.tensordot(
                    sim_params,
                    self.model_terms_matrices,
                    axes=1
                )
            except:
                print(
                    "(failed) trying to build learned hamiltonian for ",
                    self.model_id, " : ",
                    self.model_name,
                    "\nsim_params:", sim_params,
                    "\nsim op list", self.model_terms_matrices
                )
                raise

            self.log_print([
                "Updated learned info for model {}".format(self.model_id),
            ])

    def compute_expectation_values(
        self,
        # plot_probes, 
        times=[],
    ):
        probe = self.probes_for_plots[self.probe_num_qubits]
        # probe = plot_probes[self.probe_num_qubits]
        present_expec_val_times = sorted(
            list(self.expectation_values.keys())
        )
        required_times = sorted(
            list(set(times) - set(present_expec_val_times))
        )

        for t in required_times:
            self.expectation_values[t] = self.growth_class.expectation_value(
                ham=self.learned_hamiltonian,
                t=t,
                state=probe,
                log_file=self.log_file,
                log_identifier='[QML - compute expectation values]'
            )

    def r_squared(
        self,
        plot_probes,
        times=None,
        min_time=0,
        max_time=None
    ):
        self.log_print(
            [
                "R squared function for", self.model_name
            ]
        )
        if times is None:
            exp_times = sorted(
                list(self.experimental_measurements.keys())
            )
        else:
            exp_times = times
            
        if max_time is None:
            max_time = max(exp_times)

        min_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(exp_times, min_time)
        max_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(exp_times, max_time)
        min_data_idx = exp_times.index(min_time)
        max_data_idx = exp_times.index(max_time)
        exp_times = exp_times[min_data_idx:max_data_idx]
        exp_data = [
            self.experimental_measurements[t] for t in exp_times
        ]
        probe = self.probes_for_plots[self.probe_num_qubits]
        datamean = np.mean(exp_data[0:max_data_idx])
        total_sum_of_squares = 0
        for d in exp_data:
            total_sum_of_squares += (d - datamean)**2
        self.true_exp_val_mean = datamean
        self.total_sum_of_squares = total_sum_of_squares

        ham = self.learned_hamiltonian
        sum_of_residuals = 0
        available_expectation_values = sorted(
            list(self.expectation_values.keys()))

        chi_squared = 0
        self.r_squared_of_t = {}
        for t in exp_times:
            if t in available_expectation_values:
                sim = self.expectation_values[t]
            else:
                sim = self.growth_class.expectation_value(
                    ham=ham,
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
            print(
                "[ModelForStorage - r_squared] Total sum of squares is 0",
                total_sum_of_squares)
            print("data mean:", datamean)
            print("d:", d)
            print("exp_data:", exp_data)
        self.final_r_squared = 1 - (sum_of_residuals / total_sum_of_squares)
        self.sum_of_residuals = sum_of_residuals
        self.chi_squared = chi_squared
        self.p_value = (
            1 -
            sp.stats.chi2.cdf(
                self.chi_squared,
                len(exp_times) - 1  # number of degrees of freedom
            )
        )
        return self.final_r_squared

    def r_squared_by_epoch(
        self,
        plot_probes,
        times=None,
        min_time=0,
        max_time=None,
        num_points=10 
    ):
        self.log_print(
            [
                "R squared by epoch function for",
                self.model_name,
                "Times passed:",
                times
            ]
        )

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

        exp_data = [
            self.experimental_measurements[t]
            for t in exp_times
        ]
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

            ham = np.tensordot(
                self.track_param_means[int(e)],
                self.model_terms_matrices,
                axes=1
            )  # the Hamiltonian this model held at epoch e
            sum_of_residuals = 0
            available_expectation_values = sorted(
                list(self.expectation_values.keys())
            )
            for t in exp_times:
                sim = self.growth_class.expectation_value(
                    ham=ham,
                    t=t,
                    state=probe
                )
                true = self.experimental_measurements[t]
                diff_squared = (sim - true)**2
                sum_of_residuals += diff_squared

            Rsq = 1 - sum_of_residuals / datavar

            r_squared_by_epoch[e] = Rsq
        self.r_squared_by_epoch = r_squared_by_epoch
        self.final_r_squared = Rsq
        return r_squared_by_epoch


