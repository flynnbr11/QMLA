import numpy as np
import scipy as sp
import os
import time
import copy
import qinfer as qi
import random 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import redis
import pickle

import qmla.redis_settings
import qmla.logging
import qmla.get_growth_rule
import qmla.shared_functionality.prior_distributions
import qmla.database_framework
import qmla.analysis
import qmla.utilities

pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    'ModelInstanceForLearning',
]

class ModelInstanceForLearning():
    """
    Model used for parameter learning. 

    Models are specified by their name; they can be separated into 
    separate terms by splitting the name string by '+'.
    Individual terms correspond to base matrices and are assigned parameters.
    Each term is assigned a parameter probability distribution, or a prior distribution:
    this will be iteratively changed according to evidence from experiments, and its mean
    gives the estimate for that parameter. Prior distributions are used by the QInfer updater, 
    and can be specified by the :meth:`~qmla.growth_rules.GrowthRule.get_prior` method. 
    The individual terms are parsed into matrices for calculations. This is achieved by
    :func:`~qmla.process_basic_operator`: different string syntax enable different core oeprators. 

    Parameter estimation is done by :meth:`~qmla.ModelInstanceForLearning.update_model`.     
    The final parameter estimates are set as the mean of the
    posterior distribution after n_experiments wherein n_particles
    are sampled per experiment (these  user definted 
    parameters are retrieved from `qmla_core_info_dict`).
    :meth:`~qmla.ModelInstanceForLearning.learned_info_dict` returns the pertinent learned information. 

    :param int model_id: ID of the model to study
    :param str model_name: name of the model to be learned
    :param qid: ID of the QMLA instance
    :param str growth_generator: name of growth_rule
    :param dict qmla_core_info_database: essential details about the QMLA 
        instance needed to learn/compare models. 
        If None, this is retrieved instead from the redis database. 
    :param str host_name: name of host server on which redis database exists.
    :param int port_number: port number unique to this QMLA instance on redis database
    :param str log_file: path of QMLA instance's log file.
    """

    def __init__(
        self,
        model_id,
        model_name,
        qid,
        growth_generator,
        log_file,
        qmla_core_info_database=None, 
        host_name='localhost',
        port_number=6379,
        **kwargs
    ):
        self.qmla_id = qid
        self.model_id = int(model_id)
        self.model_name = model_name
        self.log_file = log_file
        self.volume_by_epoch = np.array([])
        self.redis_host = host_name
        self.redis_port_number = port_number
        self.growth_rule_of_this_model = growth_generator
        self.log_print([
            "QHL for model (id:{}) {} ".format(
                model_id, model_name, 
            )
        ])

        # Set up the model for learning
        self._initialise_model_for_learning(
            model_name = self.model_name, 
            qmla_core_info_database=qmla_core_info_database,
        )
        self._initialise_tracking_infrastructure()

    ##########
    # Section: Setup
    ##########

    def _initialise_model_for_learning(
        self,
        model_name,
        qmla_core_info_database,
        **kwargs
    ):
        r"""
        Preliminary set up necessary before parameter learning. 

        Start instances of classes used throughout, generally by calling the growth rule's method, 
            * qinfer inferface: :meth:`~qmla.growth_rules.GrowthRule.qinfer_model`.
            * updater is default `QInfer.SMCUpdater <http://docs.qinfer.org/en/latest/guide/smc.html#using-smcupdater>`_.
            * parameter distribution prior: :meth:`~qmla.growth_rules.GrowthRule.get_prior`.

        :param str model_name: name of the model to be learned
        :param str growth_generator: name of growth_rule
        :param dict qmla_core_info_database: essential details about the QMLA 
            instance needed to learn/compare models. 
            If None, this is retrieved instead from the redis database. 
        """

        # Retrieve data held on redis databases. 
        redis_databases = qmla.redis_settings.get_redis_databases_by_qmla_id(
            self.redis_host,
            self.redis_port_number,
            self.qmla_id
        )
        if qmla_core_info_database is None:
            qmla_core_info_database = redis_databases['qmla_core_info_database']
            qmla_core_info_dict = pickle.loads(qmla_core_info_database.get('qmla_settings'))
            self.probes_system = pickle.loads(qmla_core_info_database['ProbeDict'])
            self.probes_simulator = pickle.loads(qmla_core_info_database['SimProbeDict'])
        else: 
            qmla_core_info_dict = qmla_core_info_database.get('qmla_settings')
            self.probes_system = qmla_core_info_database['ProbeDict']
            self.probes_simulator = qmla_core_info_database['SimProbeDict']
        
        # Extract data from core database
        self.num_particles = qmla_core_info_dict['num_particles']
        self.num_experiments = qmla_core_info_dict['num_experiments']
        self.probe_number = qmla_core_info_dict['num_probes']
        self.qinfer_resampler_threshold = qmla_core_info_dict['resampler_thresh']
        self.qinfer_resampler_a = qmla_core_info_dict['resampler_a']
        self.qinfer_PGH_heuristic_factor = qmla_core_info_dict['pgh_prefactor']
        self.qinfer_PGH_heuristic_exponent = qmla_core_info_dict['pgh_exponent']
        self.qinfer_PGH_heuristic_increase_time = qmla_core_info_dict['increase_pgh_time']
        self.store_particle_locations_and_weights = qmla_core_info_dict['store_particles_weights']
        self.results_directory = qmla_core_info_dict['results_directory']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.true_model_name = qmla_core_info_dict['true_name']
        self.true_param_dict = qmla_core_info_dict['true_param_dict']
        self.sigma_threshold = qmla_core_info_dict['sigma_threshold']
        self.times_to_plot = qmla_core_info_dict['plot_times']
        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        self.experimental_measurement_times = qmla_core_info_dict['experimental_measurement_times']
        self.true_params_path = qmla_core_info_dict['true_params_pickle_file']
        # poterntially use different resources depending on model complexity
        
        
        # Instantiate growth rule
        try:
            self.growth_class = qmla.get_growth_rule.get_growth_generator_class(
                growth_generation_rule=self.growth_rule_of_this_model,
                log_file=self.log_file
            )
        except BaseException:
            print("Failed to load growth class {}".format(
                    self.growth_rule_of_this_model
                )
            )
            raise

        # Get initial configuration for this model 
        op = qmla.database_framework.Operator(name=model_name)
        self.model_terms_names = op.constituents_names
        self.model_name_latex = self.growth_class.latex_name(
            name=self.model_name
        )
        self.model_terms_matrices = np.asarray(op.constituents_operators)
        self.model_dimension = qmla.database_framework.get_num_qubits(self.model_name)

        # Prior parameter distribution via growth rule
        self._consider_reallocate_resources()
        self.model_prior = self.growth_class.get_prior(
            model_name = self.model_name,
            log_file = self.log_file,
            log_identifier = str("QHL {}".format(self.model_id)),
        )
        self.model_terms_parameters = self.model_prior.sample()
        self._store_prior()

        # Initialise model, updater and heuristic used with QInfer 
        self.qinfer_model = self.growth_class.qinfer_model(
            model_name=self.model_name,
            modelparams=self.model_terms_parameters,
            oplist=self.model_terms_matrices,
            true_oplist=self.true_model_constituent_operators,
            true_param_dict = self.true_param_dict, 
            truename=self.true_model_name,
            trueparams=self.true_model_params,
            num_probes=self.probe_number,
            probe_dict=self.probes_system,
            sim_probe_dict=self.probes_simulator,
            growth_generation_rule=self.growth_rule_of_this_model,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            log_file=self.log_file,
        )
        self.qinfer_updater = qi.SMCUpdater(
            self.qinfer_model,
            self.num_particles,
            self.model_prior,
            resample_thresh = self.growth_class.qinfer_resampler_threshold,
            resampler = qi.LiuWestResampler( a=self.growth_class.qinfer_resampler_a ),
        )
        self.model_heuristic = self.growth_class.heuristic(
            updater=self.qinfer_updater,
            oplist=self.model_terms_matrices,
            time_list=self.times_to_plot,
            num_experiments=self.num_experiments,
            log_file = self.log_file,
            # TODO these should be in GR or found automatically by heuristic
            inv_field = [item[0] for item in self.qinfer_model.expparams_dtype[1:]],
            increase_time=self.qinfer_PGH_heuristic_increase_time,
            pgh_exponent=self.qinfer_PGH_heuristic_exponent,
            max_time_to_enforce=self.growth_class.max_time_to_consider,
        )
        self.log_print(["Heuristic built"])
        self.model_heuristic_class = self.model_heuristic.__class__.__name__
        self.prior_marginal = [
            self.qinfer_updater.posterior_marginal(idx_param=i)
            for i in range(self.qinfer_model.n_modelparams)
        ]



    def _initialise_tracking_infrastructure(
        self
    ):
        r"""Arrays, dictionaries etc for tracking learning across experiments"""

        self.quadratic_losses = []
        self.track_total_log_likelihood = np.array([])
        self.particles = np.array([])
        self.weights = np.array([])
        self.epochs_after_resampling = []
        self.final_learned_params = np.empty(
            # TODO remove final_leared_params and references to it, 
            # use dictionaries defined here instead.
            [len(self.model_terms_matrices), 2])
        self.track_param_means = [self.qinfer_updater.est_mean()]
        self.track_param_uncertainties = []
        self.track_posterior_dist = []
        self.track_experimental_times = []
        self.qhl_final_param_estimates = {}
        self.qhl_final_param_uncertainties = {}
        self.track_covariance_matrices = []
        if self.growth_class.track_quadratic_loss:
            self.true_model_params_dict = self.growth_class.true_params_dict
            self.all_params_for_q_loss = list(
                set(list(self.true_model_params_dict.keys())).union(self.model_terms_names)
            )
            self.param_indices = {
                op_name : self.model_terms_names.index(op_name)
                for op_name in self.model_terms_names
            }

    ##########
    # Section: Model learning
    ##########

    def update_model(
        self,
    ):
        r"""
        Run updates on model, corresponding to quantum Hamiltonian learning procedure. 

        This function is called on an instance of this model to run the entire QHL algorithm.
        
        Get datum corresponding to true system, where true system is either experimental or simulated,
        by calling `simulate_experiment <http://docs.qinfer.org/en/latest/guide/smc.html#using-smcupdater>`_
        on the QInfer.SMCUpdater. This datum is taken as the true expected value for the system, which is used 
        in the likelihood calucation in the Bayesian inference step. 
        This is done by calling the `update` method on the `qinfer_updater 
        <http://docs.qinfer.org/en/latest/apiref/smc.html?highlight=smcupdater#smcupdater-smc-based-particle-updater>`_.
        Effects of the update are then recorded by :meth:`~qmla.ModelInstanceForLearning._record_experiment_updates`,
        and terminate either upon convergence or after a fixed `num_experiments`. 
        Final details are recorded by :meth:`~qmla.ModelInstanceForLearning._finalise_learning`.

        """

        self.log_print(["Updating model."])
        print_frequency = max(
            int(self.num_experiments / 5),
            5
        )
        for update_step in range(self.num_experiments):
            if (update_step % print_frequency == 0):
                # Print so user can see how far along algorithm is.
                self.log_print(["Epoch", update_step])
                try:
                    self.log_print([
                        "epoch {} - time magnitudes used: {}".format(
                            update_step,
                            self.model_heuristic.count_order_of_magnitudes
                        ) 
                    ])
                    self.model_heuristic.count_order_of_magnitudes = {} #reset
                except:
                    pass

            # Design exeriment
            new_experiment = self.model_heuristic(
                num_params=len(self.model_terms_names),
                epoch_id=update_step,
                current_params=self.qinfer_updater.est_mean()
            )

            if update_step == 0:
                self.log_print(
                    ['Initial time selected = ',str(new_experiment['t'])]
                )
            self.track_experimental_times.append(new_experiment['t'])

            # Run (or simulate) the experiment
            datum_from_experiment = self.qinfer_model.simulate_experiment(
                self.model_terms_parameters, # this doesn't actually matter - likelihood overwrites this for true system
                new_experiment,
                repeat=1
            ) 

            # Call updater to update distribution based on datum
            try:
                self.qinfer_updater.update(
                    datum_from_experiment,
                    new_experiment
                )
            except RuntimeError as e:
                import sys
                self.log_print([
                    "RuntimeError from updater on model {} - {}. Error: {}".format(
                        self.model_id, self.model_name, str(e)
                    )
                ])
                print("\n\n[Model class] EXITING; Inspect log\n\n")
                raise NameError("Qinfer update failure")
                sys.exit()
            except: 
                self.log_print(
                    [
                        "Failed to update model ({}) {} at update step {}".format(
                            self.model_id, 
                            self.model_id, 
                            update_step
                        )
                    ]
                )
                sys.exit()

            # Track learning 
            self._record_experiment_updates(update_step=update_step)

            # Terminate
            if (
                self.growth_class.terminate_learning_at_volume_convergence
                and volume_by_epoch[-1] < self.growth_class.volume_convergence_threshold

            ):  # can be reinstated to stop learning when volume converges
                self._finalise_learning()
                break

            if update_step == self.num_experiments - 1:
                self._finalise_learning()

    def _record_experiment_updates(self, update_step):
        r"""Update tracking infrastructure."""
        
        # Data used in plots
        self.volume_by_epoch = np.append(
            self.volume_by_epoch,
            qi.utils.ellipsoid_volume(
                invA = self.qinfer_updater.est_covariance_mtx()
            )
        )
        self.track_param_means.append(self.qinfer_updater.est_mean())
        self.track_param_uncertainties.append(
            np.sqrt(
                np.diag(self.qinfer_updater.est_covariance_mtx())
            )
        )
        if self.qinfer_updater.just_resampled:
            self.epochs_after_resampling.append(update_step)

        # Some optional tracking
        if self.growth_class.track_cov_mtx:
            self.track_covariance_matrices.append(
                self.qinfer_updater.est_covariance_mtx()
            )

        if self.growth_class.track_quadratic_loss:
            quadratic_loss = 0
            for param in self.all_params_for_q_loss:
                if param in self.model_terms_names:
                    learned_param = self.qinfer_updater.est_mean()[self.param_indices[param]]
                else:
                    learned_param = 0

                if param in list(self.true_model_params_dict.keys()):
                    true_param = self.true_model_params_dict[param]
                else:
                    true_param = 0
                quadratic_loss += (learned_param - true_param)**2
            self.quadratic_losses.append(quadratic_loss)


    def _finalise_learning(self):
        r"""Record and log final result."""

        # Print some results.
        try:
            self.log_print([
                "Total number of times each order of magnitude of uncertainty used during learning:", 
                self.model_heuristic.all_count_order_of_magnitudes,
                "Number of counter productive experiments:", self.model_heuristic.counter_productive_experiments
            ])
        except:
            pass

        self.log_print([
            "QHL finished for ", self.model_name,
            "Final time selected:", self.track_experimental_times[-1],
            "{} Resample epochs: {}".format(len(self.epochs_after_resampling), self.epochs_after_resampling)
        ])

        # Final results
        self.model_log_total_likelihood = self.qinfer_updater.log_total_likelihood
        self.posterior_marginal = [
            self.qinfer_updater.posterior_marginal(idx_param=i)
            for i in range(self.qinfer_model.n_modelparams)
        ]
        self.track_param_means = np.array(self.track_param_means)
        self.track_param_uncertainties = np.array(self.track_param_uncertainties)
        self.track_param_estimate_v_epoch = {}
        self.track_param_uncertainty_v_epoch = {}

        cov_mat = self.qinfer_updater.est_covariance_mtx()
        est_params = self.qinfer_updater.est_mean()
        self.log_print(["model_terms_names:", self.model_terms_names])
        for i in range(self.qinfer_model.n_modelparams):
            # Store learned parameters
                # TODO get rid of uses of final_learned_params, use qhl_final_param_estimates instead
            term = self.model_terms_names[i]
            self.final_learned_params[i] = [
                self.qinfer_updater.est_mean()[i],
                np.sqrt(cov_mat[i][i])
            ]
            
            self.qhl_final_param_estimates[term] = est_params[i]
            self.qhl_final_param_uncertainties[term] = np.sqrt(cov_mat[i][i])
            self.log_print([
                "Final parameters estimates and uncertainties (term {}): {} +/- {}".format(
                    self.model_terms_names[i],
                    self.qhl_final_param_estimates[term], 
                    self.qhl_final_param_uncertainties[term]
                )
            ])

            # Arrays of parameter estimates/uncertainties
            self.track_param_estimate_v_epoch[term] = self.track_param_means[:, i]
            self.track_param_uncertainty_v_epoch = self.track_param_uncertainties[:, i]
            
            # Compute the Hamiltonian corresponding to the parameter posterior distribution
            self.learned_hamiltonian = sum([
                self.qhl_final_param_estimates[term]
                * qmla.database_framework.compute(term)
                for term in self.qhl_final_param_estimates
            ])

        # Plots for this model
        self.plot_posterior()

    def learned_info_dict(self):
        """
        Place essential information after learning has occured into a dict.

        This is used to recreate the model for 
            * comparisons: :class:`~qmla.ModelInstanceForComparison` 
            * storage within the main QMLA environment :class:`~qmla.ModelInstanceForStorage>`.
        
        """


        learned_info = {}
        
        # needed by storage class
        learned_info['num_particles'] = self.num_particles
        learned_info['num_experiments'] = self.num_experiments
        learned_info['times_learned_over'] = self.track_experimental_times
        learned_info['final_learned_params'] = self.final_learned_params
        learned_info['model_normalization_record'] = self.qinfer_updater.normalization_record
        learned_info['log_total_likelihood'] = self.qinfer_updater.log_total_likelihood
        learned_info['raw_volume_list'] = self.volume_by_epoch
        learned_info['track_param_means'] = self.track_param_means
        learned_info['track_covariance_matrices'] = self.track_covariance_matrices
        learned_info['track_param_uncertainties'] = self.track_param_uncertainties
        learned_info['track_param_estimate_v_epoch'] = self.track_param_estimate_v_epoch
        learned_info['track_param_uncertainty_v_epoch'] = self.track_param_uncertainty_v_epoch
        learned_info['epochs_after_resampling'] = self.epochs_after_resampling
        learned_info['quadratic_losses_record'] = self.quadratic_losses
        learned_info['qhl_final_param_estimates'] = self.qhl_final_param_estimates
        learned_info['qhl_final_param_uncertainties'] = self.qhl_final_param_uncertainties
        learned_info['covariance_mtx_final'] = self.qinfer_updater.est_covariance_mtx()
        learned_info['estimated_mean_params'] = self.qinfer_updater.est_mean()
        learned_info['learned_hamiltonian'] = self.learned_hamiltonian
        learned_info['growth_rule_of_this_model'] = self.growth_rule_of_this_model
        learned_info['model_heuristic_class'] = self.model_heuristic_class
        learned_info['evaluation_log_likelihood'] = self.evaluation_log_likelihood
        learned_info['evaluation_normalization_record'] = self.evaluation_normalization_record
        learned_info['evaluation_median_likelihood'] = self.evaluation_median_likelihood

        # additionally wanted by comparison class
        learned_info['name'] = self.model_name
        learned_info['model_id'] = self.model_id
        learned_info['final_prior'] = self.qinfer_updater.prior
        learned_info['posterior_marginal'] = self.posterior_marginal
        # TODO restore initial_prior as required for plots in remote_bayes_factor

        return learned_info

    ##########
    # Section: Evaluation
    ##########

    def compute_likelihood_after_parameter_learning(
        self,
    ):
        r""""
        Evaluate the model after parameter learning on independent evaluation data. 
        """
        self.log_print(["Evaluating learned model."])
        
        # Retrieve times and probe states used for evaluation. 
        true_params_dict = pickle.load(open(
            self.true_params_path, 
            'rb'
        ))
        evaluation_times = true_params_dict['evaluation_times']
        evaluation_probe_dict = true_params_dict['evaluation_probes']

        # Construct a fresh updater and model to evaluate on.
        estimated_params = self.qinfer_updater.est_mean()
        cov_mt_uncertainty = [1e-10] * np.shape(estimated_params)[0]
        cov_mt = np.diag(cov_mt_uncertainty)
        posterior_distribution = qi.MultivariateNormalDistribution(
            estimated_params,
            cov_mt
        )
        evaluation_qinfer_model = self.growth_class.qinfer_model(
            model_name=self.model_name,
            modelparams=self.model_terms_parameters,
            oplist=self.model_terms_matrices,
            true_oplist=self.true_model_constituent_operators,
            truename=self.true_model_name,
            trueparams=self.true_model_params,
            true_param_dict = self.true_param_dict, 
            num_probes=self.probe_number,
            probe_dict=evaluation_probe_dict,
            sim_probe_dict=evaluation_probe_dict,
            growth_generation_rule=self.growth_rule_of_this_model,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            log_file=self.log_file,
        )

        evaluation_updater = qi.SMCUpdater(
            model = evaluation_qinfer_model, 
            n_particles = min(5, self.num_particles),
            prior = posterior_distribution,
            # turn off resampling - want to evaluate the learned model, not improved version
            resample_thresh = 0.0,  
            resampler = qi.LiuWestResampler(
                a = self.growth_class.qinfer_resampler_a
            ),
        )

        evaluation_updater._log_total_likelihood = 0.0
        evaluation_updater._normalization_record = []
        for t in evaluation_times:
            exp = qmla.utilities.format_experiment(
                evaluation_qinfer_model, 
                final_learned_params = self.final_learned_params, 
                time = [t],
            )
            params_array = np.array([[self.true_model_params[:]]])
            datum = evaluation_updater.model.simulate_experiment(
                params_array,
                exp,
                repeat=1
            )
            evaluation_updater.update(datum, exp)

        # Store evaluation
        self.evaluation_normalization_record = evaluation_updater.normalization_record
        if np.isnan(evaluation_updater.log_total_likelihood):
            self.evaluation_log_likelihood = None 
            self.evaluation_median_likelihood = None
            self.log_print(["Evaluation ll is nan"])
        else:
            self.evaluation_log_likelihood = qmla.utilities.round_nearest(
                evaluation_updater.log_total_likelihood, 
                0.05
            )
            self.evaluation_median_likelihood = np.round(
                np.median(evaluation_updater.normalization_record),
                2
            )
        
    def plot_posterior(self):
        if not self.growth_class.plot_posterior_after_learning: 
            # GR doesn't want this plotted
            return
        
        posterior_directory = os.path.join(
            self.results_directory, 
            'posteriors'
        )
        if not os.path.exists(posterior_directory): 
            self.log_print(["Making posterior dir:", posterior_directory])
            try:
                os.makedirs(posterior_directory)
            except:
                pass # another instance made it at same time

        posterior_file_path = os.path.join(
            posterior_directory, 
            "qmla_{}_model_{}.png".format(
                self.qmla_id, 
                self.model_id
            )
        )

        num_terms = self.qinfer_model.n_modelparams
        ncols = int(np.ceil(np.sqrt(num_terms)))
        nrows = int(np.ceil(num_terms / ncols))
        fig, axes = plt.subplots(
            figsize=(10, 7),
            nrows=nrows, 
            ncols=ncols
        )

        gs = GridSpec(
            nrows = nrows,
            ncols = ncols,
        )
        row = 0
        col = 0
        for param_idx in range(num_terms):
            term = self.model_terms_names[param_idx]
            ax = fig.add_subplot(gs[row, col])
            
            # plot prior 
            ax.plot(
                self.prior_marginal[param_idx][0], # locations
                self.prior_marginal[param_idx][1], # weights
                color='blue', 
                ls='-',
                label='Prior'
            )

            # plot posterior
            ax.plot(
                self.posterior_marginal[param_idx][0], # locations
                self.posterior_marginal[param_idx][1], # weights
                color='black', 
                ls='-',
                label='Posterior'
            )

            # True param
            if term in self.true_param_dict:
                ax.axvline(
                    self.true_param_dict[term], 
                    color='red',
                    ls='-.',
                    label='True'
                )
            
            # Learned param
            try:
                ax.axvline(
                    self.qhl_final_param_estimates[term], 
                    color='black', 
                    ls = '--',
                    label='Learned'
                )
            except:
                self.log_print(["{} not in {}".format(term, self.qhl_final_param_estimates)])

            ## There is a bug when using log scale which causes overlap on the axis labels:
            ## https://stackoverflow.com/questions/46498157/overlapping-axis-tick-labels-in-logarithmic-plots
            ax.semilogx()
            ax.semilogy()
            # ax.minorticks_off()
            ax.set_title("{}".format(self.growth_class.latex_name(term)))

            if row == 0  and col == ncols-1:
                ax.legend()

            col += 1
            if col == ncols:
                col = 0
                row += 1

        fig.text(0.5, 0.04, 'Particle locations', ha='center')
        fig.text(0.04, 0.5, 'Weights', va='center', rotation='vertical')
        fig.savefig(posterior_file_path)



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
            log_identifier='ModelForLearning {}'.format(self.model_id)
        )


    def _consider_reallocate_resources(self):
        r"""Model might get less resources if it is deemed less complex than others"""
        
        if self.growth_class.reallocate_resources:
            base_resources = qmla_core_info_dict['base_resources']
            this_model_num_qubits = qmla.database_framework.get_num_qubits(
                self.model_name)
            this_model_num_terms = len(
                qmla.database_framework.get_constituent_names_from_name(
                    self.model_name)
            )
            max_num_params = self.growth_class.max_num_parameter_estimate

            new_resources = qmla.utilities.resource_allocation(
                base_qubits = base_resources['num_qubits'],
                base_terms = base_resources['num_terms'],
                max_num_params = max_num_params,
                this_model_qubits = this_model_num_qubits,
                this_model_terms = this_model_num_terms,
                num_experiments = self.num_experiments,
                num_particles = self.num_particles
            )

            self.num_experiments = new_resources['num_experiments']
            self.num_particles = new_resources['num_particles']
            self.log_print(
                'After resource reallocation on {}: {} experiments and {} particles'.format(
                    self.model_name, self.num_experiments, self.num_particles
                )
            )

    def _store_prior(self):
        r"""Save the prior raw and as plot."""

        store_all_priors = False # optional
        if not store_all_priors:
            return

        prior_dir = str(
            self.results_directory +
            'priors/QMLA_{}/'.format(self.qmla_id)
        )

        if not os.path.exists(prior_dir):
            try:
                os.makedirs(prior_dir)
            except BaseException:
                # if already exists (ie created by another QMLA instance)
                pass
        prior_file = str(
            prior_dir +
            'prior_' +
            str(self.model_id) +
            '.png'
        )

        individual_terms_in_name = qmla.database_framework.get_constituent_names_from_name(
            self.model_name
        )
        latex_terms = []
        for term in individual_terms_in_name:
            lt = self.growth_class.latex_name(
                name=term
            )
            latex_terms.append(lt)
        
        try:
            qmla.shared_functionality.prior_distributions.plot_prior(
                model_name=self.model_name_latex,
                model_name_individual_terms=latex_terms,
                prior=self.model_prior,
                plot_file=prior_file,
            )
        except:
            self.log_print([
                "Failed to plot prior"
            ])
            
    def plot_distribution_progression(self,
                                      renormalise=False,
                                      save_to_file=None
                                      ):
        qmla.analysis.plot_distribution_progression_of_model(
            mod=self,
            num_steps_to_show=2,
            show_means=True,
            renormalise=renormalise,
            save_to_file=save_to_file
        )
