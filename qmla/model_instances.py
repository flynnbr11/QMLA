import numpy as np
import os
import sys
import time
import copy
import scipy
import random 
import logging

import qinfer as qi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
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
        self.redis_host = host_name
        self.redis_port_number = port_number
        self.growth_rule_of_this_model = growth_generator
        self.log_print([
            "QHL for model (id:{}) {} ".format(
                model_id, model_name, 
            )
        ])
        # logging.info("Model {}".format(self.model_id))

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
        self.store_particle_locations_and_weights = qmla_core_info_dict['store_particles_weights']
        self.results_directory = qmla_core_info_dict['results_directory']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.true_model_name = qmla_core_info_dict['true_name']
        if self.model_name == self.true_model_name: 
            self.is_true_model = True
            self.log_print(["This is the true model for learning."])
        else:
            self.is_true_model = False
        self.true_param_dict = qmla_core_info_dict['true_param_dict']
        self.times_to_plot = qmla_core_info_dict['plot_times']
        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        self.experimental_measurement_times = qmla_core_info_dict['experimental_measurement_times']
        self.true_params_path = qmla_core_info_dict['true_params_pickle_file']      
        self.plots_directory = qmla_core_info_dict['plots_directory']
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

        # Poterntially use different resources depending on relative model complexity.
        self._consider_reallocate_resources()

        # Prior parameter distribution via growth rule
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
            debug_log_print=False,
        )
        
        # get resampler treshold
        if (
            self.growth_class.hard_fix_resample_effective_sample_size is not None
            and self.growth_class.hard_fix_resample_effective_sample_size < self.num_particles
        ): 
            resampler_threshold = self.growth_class.hard_fix_resample_effective_sample_size / self.num_particles
        else:
            resampler_threshold = self.growth_class.qinfer_resampler_threshold
        self.log_print(["Using fixed resampler effective number partices = {}, so resampler threshold = {}".format(
            self.growth_class.hard_fix_resample_effective_sample_size, resampler_threshold
        )])


        self.qinfer_updater = qi.SMCUpdater(
            self.qinfer_model,
            self.num_particles,
            self.model_prior,
            resample_thresh = resampler_threshold,
            resampler = qi.LiuWestResampler( a=self.growth_class.qinfer_resampler_a ),
        )
        self.model_heuristic = self.growth_class.heuristic(
            model_id = self.model_id, 
            updater=self.qinfer_updater,
            oplist=self.model_terms_matrices,
            num_experiments=self.num_experiments,
            log_file = self.log_file,
            inv_field = [item[0] for item in self.qinfer_model.expparams_dtype[1:]],
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
        self.timings = { 'update_qinfer': 0 } 
        # Unused
        self.track_total_log_likelihood = np.array([])
        self.particles = np.array([])
        self.weights = np.array([])
        self.epochs_after_resampling = []
        self.track_posterior_dist = []
        
        # Final results
        self.final_learned_params = np.empty(
            # TODO remove final_leared_params and references to it, 
            # use dictionaries defined here instead.
            [len(self.model_terms_matrices), 2])
        self.qhl_final_param_estimates = {}
        self.qhl_final_param_uncertainties = {}

        # self.true_params_dict = self.growth_class.true_params_dict
        self.all_params_for_q_loss = list(
            set(list(self.true_param_dict.keys())).union(self.model_terms_names)
        )
        self.param_indices = {
            op_name : self.model_terms_names.index(op_name)
            for op_name in self.model_terms_names
        }
        # To track at every epoch  
        self.track_experimental_times = [] 

        self.volume_by_epoch = np.array([])
        self.track_param_means = []
        self.track_param_uncertainties = []
        self.track_norm_cov_matrices = []
        self.track_covariance_matrices = []
        self.quadratic_losses_record = []
        # Initialise all 
        self._record_experiment_updates(update_step = 0 )


        # self.track_experimental_times = [] 

        # self.volume_by_epoch = np.array([
        #     qi.utils.ellipsoid_volume( invA = self.qinfer_updater.est_covariance_mtx() )
        # ])
        # self.track_param_means = [self.qinfer_updater.est_mean()]
        # self.track_param_uncertainties = [np.sqrt(
        #     np.diag(self.qinfer_updater.est_covariance_mtx()))
        # ]
        # self.track_norm_cov_matrices = [
        #     np.linalg.norm(self.qinfer_updater.est_covariance_mtx())
        # ]
        # self.track_covariance_matrices = []
        # self.quadratic_losses_record = [None]
        # if self.growth_class.track_quadratic_loss:
        #     self.true_params_dict = self.growth_class.true_params_dict # TODO check these are loaded properly by GR
        #     self.all_params_for_q_loss = list(
        #         set(list(self.true_params_dict.keys())).union(self.model_terms_names)
        #     )
        #     self.param_indices = {
        #         op_name : self.model_terms_names.index(op_name)
        #         for op_name in self.model_terms_names
        #     }

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


            # Design exeriment
            # print( #debug
            #     "Current param distribution mean\n", self.qinfer_updater.est_mean(),
            #     "\n uncertainty:\n", np.sqrt(np.diag(self.qinfer_updater.est_covariance_mtx()))
            # )
            new_experiment = self.model_heuristic(
                num_params=len(self.model_terms_names),
                epoch_id=update_step,
                current_params = self.track_param_means[-1],
                current_volume = self.volume_by_epoch[-1]
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
            # self.log_print(["Datum:", datum_from_experiment])

            # Call updater to update distribution based on datum
            try:
                time_init = time.time()
                self.qinfer_updater.update(
                    datum_from_experiment,
                    new_experiment
                )
                self.timings['update_qinfer'] += time.time() - time_init
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
                import sys
                self.log_print([  
                    "Failed to update model ({}) {} at update step {}".format(
                        self.model_id, 
                        self.model_name, 
                        update_step
                    )
                ])
                raise ValueError("Failed to learn model")
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
                self.log_print(["Starting _finalise_learning"])
                self._finalise_learning()
                self.log_print(["After finalise learning"])

    def _record_experiment_updates(self, update_step):
        r"""Update tracking infrastructure."""
        
        cov_mt = self.qinfer_updater.est_covariance_mtx()
        param_estimates = self.qinfer_updater.est_mean()
        # Data used in plots
        self.volume_by_epoch = np.append(
            self.volume_by_epoch,
            qi.utils.ellipsoid_volume(
                invA = cov_mt
            )
        )
        self.track_param_means.append(param_estimates)
        self.track_param_uncertainties.append(
            np.sqrt(
                np.diag(cov_mt)
            )
        )
        self.track_norm_cov_matrices.append( np.linalg.norm(cov_mt ))
        if self.qinfer_updater.just_resampled:
            self.epochs_after_resampling.append(update_step)

        # Some optional tracking
        if self.growth_class.track_cov_mtx:
            self.track_covariance_matrices.append(
                self.qinfer_updater.est_covariance_mtx()
            )

        # compute quadratic loss
        quadratic_loss = 0
        for param in self.all_params_for_q_loss:
            if param in self.model_terms_names:
                learned_param = param_estimates[self.param_indices[param]]
            else:
                learned_param = 0

            if param in list(self.true_param_dict.keys()):
                true_param = self.true_param_dict[param]
            else:
                true_param = 0

            quadratic_loss += (learned_param - true_param)**2
        self.quadratic_losses_record.append(quadratic_loss)


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
        
        self.model_heuristic.finalise_heuristic()

        self.log_print([
            "Epoch {}".format(self.num_experiments), 
            "\n QHL finished for ", self.model_name,
            "\n Final experiment time:", self.track_experimental_times[-1],
            "\n {} Resample epochs: \n{}".format(len(self.epochs_after_resampling), self.epochs_after_resampling),
            "\nTimings:\n", self.timings
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
            self.track_param_uncertainty_v_epoch[term] = self.track_param_uncertainties[:, i]
            
        # Compute the Hamiltonian corresponding to the parameter posterior distribution
        self.learned_hamiltonian = sum([
            self.qhl_final_param_estimates[term]
            * qmla.database_framework.compute(term)
            for term in self.qhl_final_param_estimates
        ])

        # Plots for this model
        # TODO replace excepts prints with warnings
        self._plot_preliminary_preparation()
        try:
            self.plot_posterior()
        except:
            self.log_print(["Failed to plot posterior"])
        try:
            self.plot_parameters()
        except:
            self.log_print(["Failed to plot_parameters"])
            raise
            # pass
        try:
            self.model_heuristic.plot_heuristic_attributes(
                save_to_file = os.path.join(
                    self.model_learning_plots_directory, 
                    '{}heuristic_attributes_{}.png'.format(self.plot_prefix, self.model_id)
                )
            )
        except:
            self.log_print(["Failed to plot_heuristic_attributes"])
            # raise
            # pass
        try:
            self._plot_posterior_mesh_pairwise()
        except:
            self.log_print(["failed to _plot_poster_mesh_pairwise"])

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
        learned_info['track_norm_cov_matrices'] = self.track_norm_cov_matrices
        learned_info['track_param_uncertainties'] = self.track_param_uncertainties
        learned_info['track_param_estimate_v_epoch'] = self.track_param_estimate_v_epoch
        learned_info['track_param_uncertainty_v_epoch'] = self.track_param_uncertainty_v_epoch
        learned_info['epochs_after_resampling'] = self.epochs_after_resampling
        learned_info['quadratic_losses_record'] = self.quadratic_losses_record
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
        learned_info['qinfer_model_likelihoods'] = self.qinfer_model.store_likelihoods
        learned_info['qinfer_pr0_diff_from_true'] = np.array(self.qinfer_model.store_p0_diffs)

        # additionally wanted by comparison class
        learned_info['name'] = self.model_name
        learned_info['model_id'] = self.model_id
        learned_info['final_prior'] = self.qinfer_updater.prior
        learned_info['posterior_marginal'] = self.posterior_marginal
        # TODO restore initial_prior as required for plots in remote_bayes_factor
        try:
            learned_info['heuristic_data'] = self.model_heuristic.heuristic_data
        except:
            pass

        try:
            learned_info['heuristic_distances'] = self.model_heuristic.distances
        except:
            pass
        try:
            learned_info['heuristic_assorted_times'] = self.model_heuristic.designed_times
            learned_info['volume_derivatives'] = self.model_heuristic.derivatives
        except:
            pass

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

        if self.num_experiments < 20:
            # TODO make optional robustly in GR or pass dev arg to QMLA instance. 
            self.log_print(["<20 experiments; presumed dev mode. Not evaluating all models"])
            evaluation_times = evaluation_times[::10]


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
            debug_log_print=False,
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
                qhl_final_param_estimates = self.qhl_final_param_estimates,
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
        self.log_print([
            "Model evaluation ll:", self.evaluation_log_likelihood
        ])
        
    def _plot_preliminary_preparation(self):
        self.model_learning_plots_directory = os.path.join(
            self.plots_directory, 
            'model_learning'
        )
        self.plot_prefix =''
        if self.is_true_model:
            self.plot_prefix = ''
            # TODO turn back on when not in dev
            # self.plot_prefix = 'true_'

        if not os.path.exists(self.model_learning_plots_directory): 
            try:
                os.makedirs(self.model_learning_plots_directory)
            except:
                pass # another instance made it at same time


    def plot_posterior(self):
        if not self.growth_class.plot_posterior_after_learning: 
            # GR doesn't want this plotted
            # TODO replace by levelled plotting
            return

        bf_posterior = qi.MultivariateNormalDistribution(
            self.qinfer_updater.est_mean(), 
            self.qinfer_updater.est_covariance_mtx()
        )
        bf_posterior_updater = qi.SMCUpdater(
            model = self.qinfer_model, 
            n_particles = self.num_particles, 
            prior = bf_posterior
        )
        bf_posterior_marginal = [
            bf_posterior_updater.posterior_marginal(idx_param=i)
            for i in range(self.qinfer_model.n_modelparams)
        ]
        
        num_terms = self.qinfer_model.n_modelparams
        ncols = int(np.ceil(np.sqrt(num_terms)))
        nrows = int(np.ceil(num_terms / ncols))
        fig, axes = plt.subplots(
            figsize=(18, 10),
            nrows=nrows, 
            ncols=ncols,
            constrained_layout=True, 
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

            # plot posterior_used for BF comparison
            ax.plot(
                bf_posterior_marginal[param_idx][0], # locations
                bf_posterior_marginal[param_idx][1], # weights
                color='green', 
                ls=':',
                label='Prior for BF'
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
            # ax.semilogx()
            # ax.semilogy()
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

        # save the plot
        fig.savefig(
            os.path.join(self.model_learning_plots_directory, "{}distributions_{}.png".format(
                self.plot_prefix, 
                self.model_id
            )
        ))

        # Plot covariance matrix heatmap
        plt.clf()
        fig, ax = plt.subplots(
            figsize=(10, 10),
        )

        sns.heatmap(self.qinfer_updater.est_covariance_mtx(), ax = ax)
        fig.savefig(
            os.path.join(self.model_learning_plots_directory, 
            '{}cov_mtx_final_{}.png'.format(self.plot_prefix, self.model_id))
        )

    def plot_parameters(self):
        terms = self.track_param_estimate_v_epoch.keys()
        num_terms = len(terms)
        
        extra_plots = ['volume', 'time', 'pr0_diff']
        resample_colour = 'grey' 
        ncols = int(np.ceil(np.sqrt(num_terms))) 
        nrows = int(np.ceil(num_terms / ncols))+ len(extra_plots)
        self.log_print(["Plotting parameters. ncol={} nrow={}".format(ncols, nrows)])

        plt.clf()
        fig = plt.figure( 
            figsize=(
                max(3*ncols, 12), 3*nrows)
        )

        gs = GridSpec(
            nrows = nrows,
            ncols = ncols,
        )

        row = 0
        col = 0
        # Parameter estimates
        for term in terms:
            self.log_print(["Getting ax {},{} for term {}".format(row, col, term)])
            ax = fig.add_subplot(gs[row, col])
            estimates = self.track_param_estimate_v_epoch[term]
            uncertainty = self.track_param_uncertainty_v_epoch[term]
            lower_bound = estimates - uncertainty
            upper_bound = estimates + uncertainty

            epochs = range(len(estimates))

            ax.plot(epochs, estimates, label='Estimate')
            ax.fill_between(
                epochs, 
                lower_bound, 
                upper_bound,
                alpha = 0.2,
                label='Uncertainty'
            )

            if len(self.epochs_after_resampling) > 0:
                ax.axvline(
                    self.epochs_after_resampling[0], 
                    ls='--', 
                    c=resample_colour, alpha = 0.5, label='Resample'
                )

                for e in self.epochs_after_resampling[1:]:
                    ax.axvline(
                        e, 
                        ls='--', 
                        c=resample_colour, alpha = 0.5, 
                    )

            if term in self.true_param_dict:
                true_param = self.true_param_dict[term]
                ax.axhline(true_param, color='red', ls='--', label='True')
                
            try:
                term_latex = self.growth_class.latex_name(term)
                ax.set_title(term_latex)
                self.log_print(["Latex for this term:", term_latex])
            except:
                self.log_print(["Failed to get latex name"])
                raise
            ax.set_ylabel('Parameter')
            ax.set_xlabel('Epoch')
                
            if row == 0  and col == ncols-1:
                ax.legend()
                
            col += 1
            if col == ncols:
                col = 0
                row += 1

        # fig.text(0.5, 0.04, 'Epoch', ha='center')
        # fig.text(0.04, 0.5, 'Parameter', va='center', rotation='vertical')

        # Volume
        row = nrows-3
        col = 0
        ax = fig.add_subplot(gs[row, :])

        ax.plot(
            range(len(self.volume_by_epoch)), 
            self.volume_by_epoch,
            label = 'Volume',
            color='k'
        )

        if len(self.epochs_after_resampling) > 0:
            ax.axvline( # label first resample only
                self.epochs_after_resampling[0], 
                ls='--', 
                c=resample_colour, 
                alpha = 0.5, 
                label='Resample'
            )

            for e in self.epochs_after_resampling[1:]:
                ax.axvline(
                    e, 
                    ls='--', 
                    c=resample_colour, 
                    alpha = 0.5, 
                )

        ax.set_title('Volume and Experimental Times')
        ax.set_ylabel('Volume')
        ax.set_xlabel('Epoch')
        # ax.semilogy()
        ax.set_yscale('log')

        # Times learned upon
        time_ax = ax.twinx()
        histogram = False # False -> scatter plot of time v epoch
        times = qmla.utilities.flatten(self.track_experimental_times)
        if histogram:
            hist_time_bins = scipy.stats.reciprocal.rvs(
                min(times), 
                max(times), 
                size=100
            ) # evaluation times generated log-uniformly
            hist_time_bins = sorted(hist_time_bins)

            ax.hist(
                times,
                bins = hist_time_bins,
            )
            ax.semilogx()
            ax.set_title('Times learned on', pad = -15)
            ax.set_ylabel('Frequency learned upon')
            ax.set_xlabel('Time')
        else:
            if self.num_experiments > 100:
                s = 4 # size of time dots
            else:
                s = 7
            time_ax.scatter(
                range(len(self.track_experimental_times)), 
                self.track_experimental_times,
                label=r"$t \sim k \frac{1}{V}$",
                s=s,
            )
            # time_ax.set_xlabel('Epoch')
            time_ax.set_ylabel('Time')
            time_ax.semilogy()
        time_ax.legend(loc='upper right')
        ax.legend(loc='upper center')

        # Covariance mtx norm and quadratic loss
        row = nrows-2
        col = 0

        ax = fig.add_subplot(gs[row, :])
        ax.plot(
            range(len(self.track_norm_cov_matrices)), 
            self.track_norm_cov_matrices,
            label = "Covariance norm",
            color='green', ls=':'
        )
        ax.semilogy()
        ax.set_ylabel('Q.L / Norm')

        ax.plot(
            range(len(self.quadratic_losses_record)),
            self.quadratic_losses_record,
            label='Quadratic loss',
            c = 'orange', ls='--'
        )
        # ax.set_ylabel('Quadratic loss')
        ax.legend(loc='upper right')

        # | system-pr0 - particles-pr0 |
        row = nrows-1
        col = 0
        ax = fig.add_subplot(gs[row, :])
        self.qinfer_pr0_diff_from_true = np.array(self.qinfer_model.store_p0_diffs)
        medians = self.qinfer_pr0_diff_from_true[:, 0]
        std = self.qinfer_pr0_diff_from_true[:, 1]
        ax.scatter(
            range(len(medians)), 
            medians,
            s=3,
            color='Blue'
        )
        ax.fill_between(
            range(len(medians)), 
            medians + std, 
            medians - std, 
            alpha = 0.3,
            color='Blue'
        )
        # ax.set_ylim(0,1)        
        ax.set_ylabel("$ \|Pr(0)_{sys} - Pr(0)_{sim} \|  $")
        ax.set_xlabel("Epoch")
        ax.semilogy()
        try:
            ax.axhline(0.5, label='0.5', ls='--', alpha=0.3, c='grey')
            ax.axhline(0.1, label='0.1', ls=':', alpha=0.3, c='grey')
        except:
            pass
        # ax.set_yscale('log', basey=10)

        # Save figure
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.model_learning_plots_directory, 
            '{}learning_summary_{}.png'.format(self.plot_prefix, self.model_id))
        )

    def _plot_posterior_mesh_pairwise(self):
        r"""
        Plots the posterior mesh as contours for each pair of parameters. 

        Mesh from  qinfer.SMCUpdater.posterior_mesh
        """
        import itertools
        fig, axes = plt.subplots(
            figsize=(18, 10),
            constrained_layout=True
        )
        selected_cmap = plt.cm.Paired

        n_param = self.qinfer_model.n_modelparams
        nrows = ncols = n_param
        gs = GridSpec(
            nrows+1, ncols,
        )
        
        include_param_self_correlation = True
        if include_param_self_correlation: 
            pairs_of_params = list(itertools.combinations_with_replacement(range(n_param), 2))
        else:
            pairs_of_params = list(itertools.combinations(range(n_param), 2)) # exlcude param with itself
        vmin = 1e3
        vmax = 0
        posterior_meshes = {}
        for i, j in pairs_of_params:
            post_mesh = self.qinfer_updater.posterior_mesh(
                idx_param1 = i, 
                idx_param2 = j,
                res1=50, res2=50
            )
            # store the post mesh - don't want to compute twice
            posterior_meshes[i,j] = post_mesh

            # find global min/max contour value for consistency across plots
            if np.min(post_mesh[2]) < vmin: 
                vmin = np.min(post_mesh[2])
            if np.max(post_mesh[2]) > vmax:
                vmax = np.max(post_mesh[2])
                self.log_print(["Setting vmax={} for i,j={},{}".format(vmax, i, j)])
        
        for i in range(n_param):
            for j in range(n_param):
                ax = fig.add_subplot(gs[i,j])

                y_term = self.qinfer_model.modelparam_names[i]
                x_term = self.qinfer_model.modelparam_names[j]
                if ax.is_first_col():
                    ax.set_ylabel(
                        self.growth_class.latex_name(y_term),
                        rotation = 0
                    )
                if ax.is_first_row():
                    ax.set_title(
                        self.growth_class.latex_name(x_term)
                    )
                if (i, j) in pairs_of_params:
                    ax.contourf(*posterior_meshes[i,j], vmin=vmin, vmax=vmax,  cmap=selected_cmap)

                    if x_term in self.true_param_dict:
                        true_param = self.true_param_dict[x_term]
                        if ax.get_xlim()[0] < true_param < ax.get_xlim()[1]:
                            ax.axvline(true_param, c='black', ls='--', alpha=0.3)
                    if y_term in self.true_param_dict:
                        true_param = self.true_param_dict[y_term]
                        if ax.get_ylim()[0] < true_param < ax.get_ylim()[1]:
                            ax.axhline(true_param, c='black', ls='--', alpha=0.3)


                else:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                
        # colour bar
        ax = fig.add_subplot(gs[nrows, :])
        m = plt.cm.ScalarMappable(cmap=selected_cmap)
        # m.set_array(post_mesh[2])
        m.set_array([])
        m.set_clim(vmin, vmax)
        fig.colorbar(m, 
            cax = ax, orientation='horizontal', shrink=0.7
        )    

        # save
        self.log_print(["min/max contour values:{}/{}".format(vmin, vmax)])
        fig.text(0.5, 0.04, 'Posterior mesh', ha='center')
        fig.savefig(
            os.path.join(self.model_learning_plots_directory, 
            '{}posterior_mesh_pairwise_{}.png'.format(self.plot_prefix, self.model_id))
        )




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
