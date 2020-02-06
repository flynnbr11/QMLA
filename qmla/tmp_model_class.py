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
        host_name='localhost',
        port_number=6379,
        log_file='QMD_log.log',
        **kwargs
    ):
        self.redis_host_name = host_name
        self.redis_port_number = port_number
        self.qmla_id = qid

        redis_databases = rds.databases_from_qmd_id(
            self.redis_host_name,
            self.redis_port_number,
            self.qmla_id
        )
        qmla_core_info_database = redis_databases['qmla_core_info_database']
        self.model_name = model_name
        self.model_id = model_id
        self.model_terms_matrices = model_terms_matrices
        self.model_id = model_id
        # Get data from redis database which is needed to learn from
        self.probes_system = pickle.loads(qmla_core_info_database['ProbeDict'])
        self.probes_simulator = pickle.loads(qmla_core_info_database['SimProbeDict'])
        qmla_core_info_dict = pickle.loads(qmla_core_info_database.get('qmla_settings'))

        self.measurement_class = qmla_core_info_dict['measurement_type']
        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        self.use_experimental_data = qmla_core_info_dict['use_experimental_data']
        self.probe_number = qmla_core_info_dict['num_probes']
        self.qinfer_resampler_threshold = qmla_core_info_dict['resampler_thresh']
        self.qinfer_resampler_a = qmla_core_info_dict['resampler_a']
        self.qinfer_PGH_heuristic_factor = qmla_core_info_dict['pgh_prefactor']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.true_model_name = qmla_core_info_dict['true_name']
        self.probes_for_plots = pickle.load(
            open(qmla_core_info_dict['probes_plot_file'], 'rb')
        )
        self.times_to_plot = qmla_core_info_dict['plot_times']
        self.use_qle = qmla_core_info_dict['qle']
        self.use_custom_exponentiation = qmla_core_info_dict['use_exp_custom']
        self.store_particle_locations_and_weights = qmla_core_info_dict[
            'store_particles_weights'
        ]
        self.model_bayes_factors = {}
        self.model_num_qubits = qmla.database_framework.get_num_qubits(
            self.model_name)
        self.probe_num_qubits = self.model_num_qubits
        self.log_file = log_file
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
        # fitness_parameters,
        learned_info=None,
        **kwargs
    ):
        """
        Pass a dict, learned_info, with essential info on
        reconstructing the state of the model, updater and GenSimModel

        """
        if self.values_updated == False:
            self.values_updated = True
            redis_databases = rds.databases_from_qmd_id(
                self.redis_host_name,
                self.redis_port_number,
                self.qmla_id
            )
            learned_models_info = redis_databases['learned_models_info']
            self.log_print(
                [
                    "Updating learned info for model {}".format(self.model_id),
                ]
            )

            if learned_info is None:
                model_id_float = float(self.model_id)
                model_id_str = str(model_id_float)
                try:
                    learned_info = pickle.loads(
                        learned_models_info.get(model_id_str),
                        encoding='latin1'
                    )
                except BaseException:
                    self.log_print(
                        [
                            "Unable to load learned info",
                            "model_id_str: ", model_id_str,
                            "model id: ", self.model_id,
                            "learned info keys:, ", learned_models_info.keys(),
                            "learned info:, ", learned_models_info.get(
                                model_id_str)
                        ]
                    )
            self.num_particles = learned_info['num_particles']
            self.num_experiments = learned_info['num_experiments']
            self.times_learned_over = list(learned_info['times'])
            self.final_learned_params = learned_info['final_params']
            self.model_terms_parameters_final = np.array(
                [[self.final_learned_params[0, 0]]])
            self.model_prior = learned_info['final_prior']
            self.model_normalization_record = learned_info['normalization_record']
            self.log_total_likelihod = learned_info['log_total_likelihood']
            self.raw_volume_list = learned_info['volume_list']
            self.volume_by_epoch = {}
            for i in range(len(self.raw_volume_list)):
                self.volume_by_epoch[i] = self.raw_volume_list[i]

            self.track_mean_params = np.array(learned_info['track_mean_params'])
            self.track_covariance_matrices = np.array(
                learned_info['track_cov_matrices'])
            self.track_param_dist_widths = np.array(
                learned_info['track_param_sigmas'])
            self.track_prior_means = np.array(
                learned_info['track_prior_means'])
            self.track_posterior_dist = np.array(
                learned_info['track_posterior'])
            self.track_prior_std_dev = np.array(
                learned_info['track_prior_std_devs'])
            self.epochs_after_resampling = learned_info['epochs_after_resampling']
            self.quadratic_losses_record = learned_info['quadratic_losses']
            self.learned_parameters_qhl = learned_info['learned_parameters']
            self.final_sigmas_qhl = learned_info['final_sigmas']

            self.cov_matrix = learned_info['cov_matrix']
            self.growth_rule_of_this_model = learned_info['growth_generator']
            try:
                self.growth_class = get_growth_rule.get_growth_generator_class(
                    growth_generation_rule=self.growth_rule_of_this_model,
                    use_experimental_data=self.use_experimental_data,
                    log_file=self.log_file
                )
            except BaseException:
                raise
            self.model_heuristic_class = learned_info['heuristic']

            self.model_name_latex = self.growth_class.latex_name(
                name=self.model_name
            )

            self.track_parameter_estimates = {}
            num_params = np.shape(self.track_mean_params)[1]
            max_exp = np.shape(self.track_mean_params)[0] - 1
            for i in range(num_params):
                for term in self.learned_parameters_qhl.keys():
                    if self.learned_parameters_qhl[term] == self.track_mean_params[max_exp][i]:
                        self.track_parameter_estimates[term] = self.track_mean_params[:, i]

            try:
                self.particles = np.array(learned_info['particles'])
                self.weights = np.array(learned_info['weights'])
            except BaseException:
                self.particles = 'Particles not stored.'
                self.weights = 'Weights not stored.'

            sim_params = list(self.final_learned_params[:, 0])
            try:
                self.learned_hamiltonian = np.tensordot(
                    sim_params,
                    self.model_terms_matrices,
                    axes=1
                )
            except BaseException:
                print(
                    "[QML] (failed) trying to build learned hamiltonian for ",
                    self.model_id, " : ",
                    self.model_name,
                    "\nsim_params:", sim_params,
                    "\nsim op list", self.model_terms_matrices
                )
                raise

            self.log_print(
                [
                    "Updated learned info for model {}".format(self.model_id),

                ]
            )

    def compute_expectation_values(
        self,
        times=[],
    ):
        probe = self.probes_for_plots[self.probe_num_qubits]
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

        min_time = qmla.experimental_data_processing.nearestAvailableExpTime(exp_times, min_time)
        max_time = qmla.experimental_data_processing.nearestAvailableExpTime(exp_times, max_time)
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

        min_time = qmla.experimental_data_processing.nearestAvailableExpTime(
            exp_times,
            min_time
        )
        max_time = qmla.experimental_data_processing.nearestAvailableExpTime(
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
                self.track_mean_params[int(e)],
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

