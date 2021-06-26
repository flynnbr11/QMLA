import numpy as np
import os
import sys
import time
import copy
import scipy
import random
import logging
import pandas as pd

import qinfer as qi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import redis
import pickle

try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure

import qmla.redis_settings
import qmla.logging
import qmla.get_exploration_strategy
import qmla.shared_functionality.prior_distributions
import qmla.model_building_utilities
import qmla.analysis
import qmla.utilities

pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    "ModelInstanceForLearning",
]


class ModelInstanceForLearning:
    """
    Model used for parameter learning.

    Models are specified by their name; they can be separated into
    separate terms by splitting the name string by '+'.
    Individual terms correspond to base matrices and are assigned parameters.
    Each term is assigned a parameter probability distribution, or a prior distribution:
    this will be iteratively changed according to evidence from experiments, and its mean
    gives the estimate for that parameter. Prior distributions are used by the QInfer updater,
    and can be specified by the :meth:`~qmla.exploration_strategies.ExplorationStrategy.get_prior` method.
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
    :param str exploration_rule: name of exploration_strategy
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
        exploration_rule,
        log_file,
        qmla_core_info_database=None,
        host_name="localhost",
        port_number=6379,
        **kwargs
    ):
        self.qmla_id = qid
        self.model_id = int(model_id)
        self.model_name = model_name
        self.log_file = log_file
        self.redis_host = host_name
        self.redis_port_number = port_number
        self.exploration_strategy_of_this_model = exploration_rule
        self.log_print(
            [
                "QHL for model (id:{}) {} ".format(
                    model_id,
                    model_name,
                )
            ]
        )

        # Set up the model for learning
        self._initialise_model_for_learning(
            model_name=self.model_name,
            qmla_core_info_database=qmla_core_info_database,
        )
        self._initialise_tracking_infrastructure()

    ##########
    # Section: Setup
    ##########

    def _initialise_model_for_learning(
        self, model_name, qmla_core_info_database, **kwargs
    ):
        r"""
        Preliminary set up necessary before parameter learning.

        Start instances of classes used throughout, generally by calling the exploration strategy's method,
            * qinfer inferface: :meth:`~qmla.exploration_strategies.ExplorationStrategy.qinfer_model`.
            * updater is default `QInfer.SMCUpdater <http://docs.qinfer.org/en/latest/guide/smc.html#using-smcupdater>`_.
            * parameter distribution prior: :meth:`~qmla.exploration_strategies.ExplorationStrategy.get_prior`.

        :param str model_name: name of the model to be learned
        :param str exploration_rule: name of exploration_strategy
        :param dict qmla_core_info_database: essential details about the QMLA
            instance needed to learn/compare models.
            If None, this is retrieved instead from the redis database.
        """

        # Retrieve data held on redis databases.
        redis_databases = qmla.redis_settings.get_redis_databases_by_qmla_id(
            self.redis_host, self.redis_port_number, self.qmla_id
        )
        if qmla_core_info_database is None:
            qmla_core_info_database = redis_databases["qmla_core_info_database"]
            qmla_core_info_dict = pickle.loads(
                qmla_core_info_database.get("qmla_settings")
            )
            self.probes_system = pickle.loads(qmla_core_info_database["probes_system"])
            self.probes_simulator = pickle.loads(
                qmla_core_info_database["probes_simulator"]
            )
        else:
            qmla_core_info_dict = qmla_core_info_database.get("qmla_settings")
            self.probes_system = qmla_core_info_database["probes_system"]
            self.probes_simulator = qmla_core_info_database["probes_simulator"]

        # Extract data from core database
        self.num_particles = qmla_core_info_dict["num_particles"]
        self.num_experiments = qmla_core_info_dict["num_experiments"]
        self.probe_number = qmla_core_info_dict["num_probes"]
        self.results_directory = qmla_core_info_dict["results_directory"]
        self.true_model_constituent_operators = qmla_core_info_dict["true_oplist"]
        self.true_model_params = qmla_core_info_dict["true_model_terms_params"]
        self.true_model_name = qmla_core_info_dict["true_name"]
        if self.model_name == self.true_model_name:
            self.is_true_model = True
            self.log_print(["This is the true model for learning."])
        else:
            self.is_true_model = False
        self.true_param_dict = qmla_core_info_dict["true_param_dict"]
        self.true_model_constructor = qmla_core_info_dict["true_model_constructor"]
        self.times_to_plot = qmla_core_info_dict["plot_times"]
        self.experimental_measurements = qmla_core_info_dict[
            "experimental_measurements"
        ]
        self.experimental_measurement_times = qmla_core_info_dict[
            "experimental_measurement_times"
        ]
        self.true_params_path = qmla_core_info_dict["run_info_file"]
        self.plot_probes = pickle.load(
            open(qmla_core_info_dict["probes_plot_file"], "rb")
        )
        self.plots_directory = qmla_core_info_dict["plots_directory"]
        self.debug_mode = qmla_core_info_dict["debug_mode"]
        self.plot_level = qmla_core_info_dict["plot_level"]
        self.figure_format = qmla_core_info_dict["figure_format"]

        # Instantiate exploration strategy
        self.exploration_class = qmla.get_exploration_strategy.get_exploration_class(
            exploration_rules=self.exploration_strategy_of_this_model,
            log_file=self.log_file,
            qmla_id=self.qmla_id,
        )

        # Get initial configuration for this model
        self.model_constructor = self.exploration_class.model_constructor(
            name=model_name
        )
        self.model_terms_names = self.model_constructor.terms_names
        # self.model_name_latex = self.exploration_class.latex_name(
        #     name=self.model_name
        # )
        self.model_name_latex = self.model_constructor.name_latex
        self.model_terms_matrices = np.asarray(self.model_constructor.terms_matrices)
        self.num_parameters = self.model_constructor.num_parameters
        self.model_dimension = self.model_constructor.num_qubits
        self.log_print(["Getting num qubits"])
        self.model_num_qubits = int(np.log2(np.shape(self.model_terms_matrices[0])[0]))
        self.log_print(["model num qubits:", self.model_num_qubits])
        self.log_print(["model dimension:", self.model_dimension])

        # Poterntially use different resources depending on relative model
        # complexity.
        self._consider_reallocate_resources()

        # Set up
        self._setup_qinfer_infrastructure()

    def _setup_qinfer_infrastructure(self):
        r"""
        Set up prior, model and updater (via QInfer) which are used to run Bayesian inference.
        """

        # Prior parameter distribution vian exploration strategy
        self.model_prior = self.exploration_class.get_prior(
            model_name=self.model_name,
            log_file=self.log_file,
            log_identifier=str("QHL {}".format(self.model_id)),
        )
        self.model_terms_parameters = self.model_prior.sample()
        self._store_prior()

        # Initialise model to infereace with QInfer as specified in ES
        self.qinfer_model = self.exploration_class.get_qinfer_model(
            model_name=self.model_name,
            model_constructor=self.model_constructor,
            true_oplist=self.true_model_constituent_operators,
            true_model_constructor=self.true_model_constructor,
            num_probes=self.probe_number,
            probes_system=self.probes_system,
            probes_simulator=self.probes_simulator,
            exploration_rules=self.exploration_strategy_of_this_model,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            qmla_id=self.qmla_id,
            log_file=self.log_file,
            debug_mode=self.debug_mode,
        )

        # Updater to perform Bayesian inference with

        if (
            self.exploration_class.hard_fix_resample_effective_sample_size is not None
            and self.exploration_class.hard_fix_resample_effective_sample_size
            < self.num_particles
        ):
            # get resampler treshold
            resampler_threshold = (
                self.exploration_class.hard_fix_resample_effective_sample_size
                / self.num_particles
            )
        else:
            resampler_threshold = self.exploration_class.qinfer_resampler_threshold

        self.qinfer_updater = qi.SMCUpdater(
            self.qinfer_model,
            self.num_particles,
            self.model_prior,
            resample_thresh=resampler_threshold,
            resampler=qi.LiuWestResampler(a=self.exploration_class.qinfer_resampler_a),
        )

        # Experiment design heuristic
        self.model_heuristic = self.exploration_class.get_heuristic(
            model_id=self.model_id,
            updater=self.qinfer_updater,
            oplist=self.model_terms_matrices,
            num_experiments=self.num_experiments,
            num_probes=self.probe_number,
            log_file=self.log_file,
            inv_field=[item[0] for item in self.qinfer_model.expparams_dtype[1:]],
            max_time_to_enforce=self.exploration_class.max_time_to_consider,
            figure_format=self.figure_format,
        )
        self.log_print(["Heuristic built"])
        self.model_heuristic_class = self.model_heuristic.__class__.__name__

        self.prior_marginal = [
            self.qinfer_updater.posterior_marginal(idx_param=i)
            for i in range(self.model_constructor.num_terms)
        ]

    def _initialise_tracking_infrastructure(self):
        r"""
        Arrays, dictionaries etc for tracking learning across experiments
        """

        # Unused
        self.timings = {"update_qinfer": 0}
        self.track_total_log_likelihood = np.array([])
        self.particles = np.array([])
        self.weights = np.array([])

        self.track_posterior_dist = []

        # Final results
        self.final_learned_params = np.empty(
            # TODO remove final_leared_params and references to it,
            # use dictionaries defined here instead.
            [self.num_parameters, 2]
        )
        self.qhl_final_param_estimates = {}
        self.qhl_final_param_uncertainties = {}

        # Miscellaneous
        self.progress_tracker = pd.DataFrame()
        self.all_params_for_q_loss = list(
            set(list(self.true_param_dict.keys())).union(self.model_terms_names)
        )
        self.param_indices = {
            op_name: self.model_terms_names.index(op_name)
            for op_name in self.model_terms_names
        }
        self.epochs_after_resampling = []
        # To track at every epoch
        self.track_experimental_times = []
        self.track_experiment_parameters = []
        self.volume_by_epoch = np.array([])
        self.track_param_means = []
        self.track_param_uncertainties = []
        self.track_norm_cov_matrices = []
        self.track_covariance_matrices = []
        self.quadratic_losses_record = []
        # Initialise all
        self._record_experiment_updates(update_step=0)

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
        print_frequency = max(int(self.num_experiments / 5), 5)
        for update_step in range(self.num_experiments):
            if update_step % print_frequency == 0:
                # Print so user can see how far along the learning is.
                self.log_print(["Epoch", update_step])

            # Design exeriment
            new_experiment = self.model_heuristic(
                num_params=self.model_constructor.num_parameters,
                epoch_id=update_step,
                current_params=self.track_param_means[-1],
                current_volume=self.volume_by_epoch[-1],
            )
            self.track_experimental_times.append(new_experiment["t"])
            self.track_experiment_parameters.append(new_experiment)
            self.log_print_debug(["New experiment:", new_experiment])

            # Run (or simulate) the experiment
            datum_from_experiment = self.qinfer_model.simulate_experiment(
                self.model_terms_parameters,
                # this doesn't actually matter - likelihood overwrites this for
                # true system
                new_experiment,
                repeat=1,
            )
            self.log_print(["Datum:", datum_from_experiment])
            self.log_print(["Exp:", new_experiment])
            # Call updater to update distribution based on datum
            try:
                update_start = time.time()
                self.qinfer_updater.update(datum_from_experiment, new_experiment)
                update_time = time.time() - update_start
            except RuntimeError as e:
                import sys

                self.log_print(
                    [
                        "RuntimeError from updater on model {} - {}. Error: {}".format(
                            self.model_id, self.model_name, str(e)
                        )
                    ]
                )
                print("\n\n[Model class] EXITING; Inspect log\n\n")
                raise NameError("Qinfer update failure")
                sys.exit()
            except BaseException:
                import sys

                self.log_print(
                    [
                        "Failed to update model ({}) {} at update step {}".format(
                            self.model_id, self.model_name, update_step
                        )
                    ]
                )
                raise ValueError("Failed to learn model")
                sys.exit()

            # Track learning
            self._record_experiment_updates(
                update_step=update_step,
                new_experiment=new_experiment,
                datum=datum_from_experiment,
                update_time=update_time,
            )

            # Terminate
            if (
                self.exploration_class.terminate_learning_at_volume_convergence
                and volume_by_epoch[-1]
                < self.exploration_class.volume_convergence_threshold
            ):  # can be reinstated to stop learning when volume converges
                self._finalise_learning()
                break

        self._finalise_learning()
        self.compute_likelihood_after_parameter_learning()
        t1 = time.time()
        self._model_plots()
        self.log_print(
            ["Time to do plots: {} sec".format(np.round(time.time() - t1, 3))]
        )
        t2 = time.time()
        self.model_heuristic.finalise_heuristic()
        self.log_print(
            ["Time to finalise heuristic: {} sec".format(np.round(time.time() - t2, 3))]
        )

    def _record_experiment_updates(
        self,
        update_step,
        new_experiment=None,
        datum=None,
        update_time=0,
    ):
        r"""Update tracking infrastructure."""

        cov_mt = self.qinfer_updater.est_covariance_mtx()
        param_estimates = self.qinfer_updater.est_mean()

        # Data used in plots
        volume = np.abs(qi.utils.ellipsoid_volume(invA=cov_mt))
        self.volume_by_epoch = np.append(self.volume_by_epoch, volume)
        self.track_param_means.append(param_estimates)
        self.track_param_uncertainties.append(np.sqrt(np.diag(cov_mt)))
        self.track_norm_cov_matrices.append(np.linalg.norm(cov_mt))
        if self.qinfer_updater.just_resampled:
            self.epochs_after_resampling.append(update_step)

        # Some optional tracking
        if self.exploration_class.track_cov_mtx:
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

            quadratic_loss += (learned_param - true_param) ** 2
        self.quadratic_losses_record.append(quadratic_loss)

        if new_experiment is None:
            exp_time = None
            probe_id = None
            total_likelihood = None
        else:
            exp_time = new_experiment["t"][0]
            probe_id = new_experiment["probe_id"]
            total_likelihood = self.qinfer_updater.normalization_record[-1][0]

        try:
            residual_median = self.qinfer_model.store_p0_diffs[-1][0]
            residual_std = self.qinfer_model.store_p0_diffs[-1][1]
        except:
            residual_median = None
            residual_std = None

        if update_time == 0:
            storage_time = 0
            likelihood_time = 0
        else:
            try:
                storage_time = self.qinfer_model.single_experiment_timings["simulator"][
                    "storage"
                ]
                likelihood_time = self.qinfer_model.single_experiment_timings[
                    "simulator"
                ]["likelihood"]
            except Exception as e:
                raise
                self.log_print(["Can't find storage/likelihood time. Exception : ", e])

        experiment_summary = pd.Series(
            {
                "model_id": self.model_id,
                "model_name": self.model_name_latex,
                "num_qubits": self.model_num_qubits,
                "experiment_id": update_step + 1,  # update_step counts from 0
                "parameters_true": self.true_model_params,
                "parameters_estimates": param_estimates,
                "parameters_uncertainties": self.track_param_uncertainties[-1],
                "volume": volume,
                "quadratic_loss": quadratic_loss,
                "experiment_time": exp_time,
                "probe_id": probe_id,
                "residual_median": residual_median,
                "residual_std_dev": residual_std,
                "just_resampled": self.qinfer_updater.just_resampled,
                "effective_sample_size": self.qinfer_updater.n_ess,
                "datum": datum,
                "total_likelihood": total_likelihood,
                "update_time": update_time,
                "storage_time": storage_time,
                "likelihood_time": likelihood_time,
            }
        )
        self.progress_tracker = self.progress_tracker.append(
            experiment_summary, ignore_index=True
        )

    def _finalise_learning(self):
        r"""Record and log final result."""

        self.log_print(
            [
                "Epoch {}".format(self.num_experiments),
                "\n QHL finished for ",
                self.model_name,
                "\n Final experiment time:",
                self.track_experimental_times[-1],
                "\n {} Resample epochs: \n{}".format(
                    len(self.epochs_after_resampling), self.epochs_after_resampling
                ),
                "\nTimings:\n",
                self.timings,
                "\nEffective sample size: {}".format(self.qinfer_updater.n_ess),
            ]
        )

        # Final results
        self.model_log_total_likelihood = self.qinfer_updater.log_total_likelihood
        self.posterior_marginal = [
            self.qinfer_updater.posterior_marginal(idx_param=i)
            for i in range(self.model_constructor.num_terms)
        ]
        self.track_param_means = np.array(self.track_param_means)
        self.track_param_uncertainties = np.array(self.track_param_uncertainties)
        self.track_param_estimate_v_epoch = {}
        self.track_param_uncertainty_v_epoch = {}

        cov_mat = self.qinfer_updater.est_covariance_mtx()

        est_params = self.qinfer_updater.est_mean()
        self.log_print(["model_terms_names:", self.model_terms_names])
        for i in range(self.model_constructor.num_terms):

            # Store learned parameters
            # TODO get rid of uses of final_learned_params, use
            # qhl_final_param_estimates instead
            term = self.model_terms_names[i]
            self.final_learned_params[i] = [
                self.qinfer_updater.est_mean()[i],
                np.sqrt(cov_mat[i][i]),
            ]

            self.qhl_final_param_estimates[term] = est_params[i]
            self.qhl_final_param_uncertainties[term] = np.sqrt(cov_mat[i][i])
            self.log_print(
                [
                    "Final parameters estimates and uncertainties (term {}): {} +/- {}".format(
                        term,
                        self.qhl_final_param_estimates[term],
                        self.qhl_final_param_uncertainties[term],
                    )
                ]
            )

            # Arrays of parameter estimates/uncertainties
            self.track_param_estimate_v_epoch[term] = self.track_param_means[:, i]
            self.track_param_uncertainty_v_epoch[term] = self.track_param_uncertainties[
                :, i
            ]

        # Compute the Hamiltonian corresponding to the parameter
        # posterior distribution
        self.learned_hamiltonian = self.model_constructor.construct_matrix(
            parameters=est_params
        )

        # Record parameter estimates
        pe = pd.DataFrame(self.track_param_estimate_v_epoch)
        pu = pd.DataFrame(self.track_param_uncertainty_v_epoch)
        pu.index.rename("experiment_id", inplace=True)
        pe.index.rename("experiment_id", inplace=True)
        pu.rename(
            columns={d: "uncertainty_{}".format(d) for d in pu.keys()}, inplace=True
        )
        self.parameter_estimates = pu.join(pe, on="experiment_id")

        # Compute dynamics
        self._compute_expectation_values()

    def _model_plots(
        self,
    ):
        self.log_print(["Plotting instance outcomes"])
        self._plot_preliminary_preparation()

        plot_methods_by_level = {
            3: [
                self._plot_learning_summary,
                self._plot_dynamics,
            ],
            4: [
                self._plot_distributions,
            ],
            5: [
                # nothing at this level
            ],
            6: [
                self._plot_posterior_mesh_pairwise,
            ],
        }

        for pl in range(self.plot_level + 1):
            if pl in plot_methods_by_level:
                self.log_print(["Plotting for plot_level={}".format(pl)])
                for method in plot_methods_by_level[pl]:
                    try:
                        method()
                    except Exception as e:
                        self.log_print(
                            [
                                "plot failed {} with exception: {}".format(
                                    method.__name__, e
                                )
                            ]
                        )

        if self.plot_level >= 4:
            try:
                self.model_heuristic.plot_heuristic_attributes(
                    save_to_file=os.path.join(
                        self.model_learning_plots_directory,
                        "{}heuristic_attributes_{}".format(
                            self.plot_prefix, self.model_id
                        ),
                    )
                )
            except BaseException:
                self.log_print(["Failed to plot_heuristic_attributes"])

    def _model_plots_old(self):
        r"""
        Generate plots specific to this model.
        Which plots are drawn depends on the ``plot_level`` set in the launch script.
        """

        if self.plot_level >= 4:
            # Plots for this model, if plot level wants to include them
            # TODO replace excepts prints with warnings
            self._plot_preliminary_preparation()

            try:
                self._plot_learning_summary()
            except BaseException:
                self.log_print(["Failed to _plot_learning_summary"])

            try:
                self._plot_dynamics()
            except:
                self.log_print(["Failed to plot model dynamics."])
                # raise

        if self.plot_level >= 5:
            try:
                self._plot_distributions()
            except BaseException:
                raise
                self.log_print(["Failed to plot posterior"])

            try:
                self.model_heuristic.plot_heuristic_attributes(
                    save_to_file=os.path.join(
                        self.model_learning_plots_directory,
                        "{}heuristic_attributes_{}.png".format(
                            self.plot_prefix, self.model_id
                        ),
                    )
                )
            except BaseException:
                self.log_print(["Failed to plot_heuristic_attributes"])

        if self.plot_level >= 7:
            # very heavy, not very informative
            try:
                self._plot_posterior_mesh_pairwise()
            except BaseException:
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
        learned_info["num_particles"] = self.num_particles
        learned_info["num_experiments"] = self.num_experiments
        learned_info["times_learned_over"] = self.track_experimental_times
        learned_info["track_experiment_parameters"] = self.track_experiment_parameters
        learned_info["final_learned_params"] = self.final_learned_params
        learned_info[
            "model_normalization_record"
        ] = self.qinfer_updater.normalization_record
        learned_info["log_total_likelihood"] = self.qinfer_updater.log_total_likelihood
        learned_info["raw_volume_list"] = self.volume_by_epoch
        learned_info["track_param_means"] = self.track_param_means
        learned_info["track_covariance_matrices"] = self.track_covariance_matrices
        learned_info["track_norm_cov_matrices"] = self.track_norm_cov_matrices
        learned_info["track_param_uncertainties"] = self.track_param_uncertainties
        learned_info["track_param_estimate_v_epoch"] = self.track_param_estimate_v_epoch
        learned_info[
            "track_param_uncertainty_v_epoch"
        ] = self.track_param_uncertainty_v_epoch
        learned_info["epochs_after_resampling"] = self.epochs_after_resampling
        learned_info["quadratic_losses_record"] = self.quadratic_losses_record
        learned_info["qhl_final_param_estimates"] = self.qhl_final_param_estimates
        learned_info[
            "qhl_final_param_uncertainties"
        ] = self.qhl_final_param_uncertainties
        learned_info["covariance_mtx_final"] = self.qinfer_updater.est_covariance_mtx()
        learned_info["estimated_mean_params"] = self.qinfer_updater.est_mean()
        learned_info["learned_hamiltonian"] = self.learned_hamiltonian
        learned_info[
            "exploration_strategy_of_this_model"
        ] = self.exploration_strategy_of_this_model
        learned_info["model_heuristic_class"] = self.model_heuristic_class
        learned_info["evaluation_log_likelihood"] = self.evaluation_log_likelihood
        learned_info[
            "evaluation_normalization_record"
        ] = self.evaluation_normalization_record
        learned_info["akaike_info_criterion"] = self.akaike_info_criterion
        learned_info["akaike_info_criterion_c"] = self.akaike_info_criterion_c
        learned_info["bayesian_info_criterion"] = self.bayesian_info_criterion
        learned_info["evaluation_median_likelihood"] = self.evaluation_median_likelihood
        learned_info["evaluation_pr0_diffs"] = self.evaluation_pr0_diffs
        learned_info["evaluation_mean_pr0_diff"] = np.mean(self.evaluation_pr0_diffs)
        learned_info["evaluation_median_pr0_diff"] = np.median(
            self.evaluation_pr0_diffs
        )
        learned_info["num_evaluation_points"] = self.num_evaluation_points
        learned_info["qinfer_model_likelihoods"] = self.qinfer_model.store_likelihoods
        learned_info["evaluation_likelihoods"] = self.evaluation_likelihoods
        learned_info["evaluation_residual_squares"] = self.evaluation_residual_squares
        learned_info[
            "evaluation_summarise_likelihoods"
        ] = self.evaluation_summarise_likelihoods
        learned_info["qinfer_pr0_diff_from_true"] = np.array(
            self.qinfer_model.store_p0_diffs
        )
        learned_info["expectation_values"] = self.expectation_values
        learned_info["progress_tracker"] = self.progress_tracker
        learned_info["parameter_estimates"] = self.parameter_estimates

        # additionally wanted by comparison class
        learned_info["name"] = self.model_name
        learned_info["model_id"] = self.model_id
        learned_info["final_prior"] = self.qinfer_updater.prior
        learned_info["posterior_marginal"] = self.posterior_marginal
        # TODO restore initial_prior as required for plots in
        # remote_bayes_factor
        try:
            learned_info["heuristic_data"] = self.model_heuristic.heuristic_data
        except BaseException:
            pass

        try:
            learned_info["heuristic_distances"] = self.model_heuristic.distances
        except BaseException:
            pass
        try:
            learned_info[
                "heuristic_assorted_times"
            ] = self.model_heuristic.designed_times
            learned_info["volume_derivatives"] = self.model_heuristic.derivatives
        except BaseException:
            pass

        return learned_info

    ##########
    # Section: Evaluation
    ##########

    def compute_likelihood_after_parameter_learning(
        self,
    ):
        r""" "
        Evaluate the model after parameter learning on independent evaluation data.
        """
        self.log_print(["Evaluating learned model."])

        # Retrieve probes and experiment list used as evaluation data.

        evaluation_data = pickle.load(
            open(os.path.join(self.results_directory, "evaluation_data.p"), "rb")
        )  # TODO get from command line argument instead of reconstructing path here

        # evaluation_times = evaluation_data['times']
        evaluation_probe_dict = evaluation_data["probes"]
        evaluation_experiments = evaluation_data["experiments"]
        self.num_evaluation_points = len(evaluation_experiments)

        if not self.exploration_class.force_evaluation and self.num_experiments < 20:
            # TODO make optional robustly in ES or pass dev arg to QMLA
            # instance.
            self.log_print(
                ["<20 experiments; presumed dev mode. Not evaluating all models"]
            )
            evaluation_experiments = evaluation_experiments[::10]
        if self.exploration_class.exclude_evaluation:
            evaluation_experiments = evaluation_experiments[::10]

        self.log_print(
            [
                "Evaluation experiments len {}. First 5 elements:\n{}".format(
                    len(evaluation_experiments), evaluation_experiments[:5]
                )
            ]
        )

        # Construct a fresh updater and model to evaluate on.
        estimated_params = self.qinfer_updater.est_mean()
        cov_mt_uncertainty = [1e-10] * np.shape(estimated_params)[0]
        cov_mt = np.diag(cov_mt_uncertainty)
        posterior_distribution = self.exploration_class.get_evaluation_prior(
            model_name=self.model_name,
            estimated_params=estimated_params,
            cov_mt=cov_mt,
        )
        evaluation_model_constructor = self.exploration_class.model_constructor(
            name=self.model_name, fixed_parameters=estimated_params
        )

        # TODO using precise mean of posterior to evaluate model
        # want to sample from it -- add flag to qinfer model
        evaluation_qinfer_model = self.exploration_class.get_qinfer_model(
            model_name=self.model_name,
            model_constructor=evaluation_model_constructor,
            true_model_constructor=self.true_model_constructor,
            num_probes=self.probe_number,
            probes_system=evaluation_probe_dict,
            probes_simulator=evaluation_probe_dict,
            exploration_rules=self.exploration_strategy_of_this_model,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            log_file=self.log_file,
            debug_mode=self.debug_mode,
            qmla_id=self.qmla_id,
            evaluation_model=True,
        )

        evaluation_updater = qi.SMCUpdater(
            model=evaluation_qinfer_model,
            n_particles=min(5, self.num_particles),
            prior=posterior_distribution,
            # turn off resampling - want to evaluate the learned model, not
            # improved version
            resample_thresh=0.0,
            resampler=qi.LiuWestResampler(a=self.exploration_class.qinfer_resampler_a),
        )
        evaluation_heuristic = self.exploration_class.get_heuristic(
            model_id=self.model_id,
            updater=evaluation_updater,
            oplist=self.model_terms_matrices,
            num_experiments=self.num_experiments,
            num_probes=self.probe_number,
            log_file=self.log_file,
            inv_field=[item[0] for item in self.qinfer_model.expparams_dtype[1:]],
            max_time_to_enforce=self.exploration_class.max_time_to_consider,
            figure_format=self.figure_format,
        )

        evaluation_updater._log_total_likelihood = 0.0
        evaluation_updater._normalization_record = []
        eval_epoch = 0
        self.log_print(
            ["Evaluating on {} experiments".format(len(evaluation_experiments))]
        )
        for experiment in evaluation_experiments:
            t = experiment["t"].item()
            probe_id = experiment["probe_id"].item()

            exp = evaluation_heuristic(
                num_params=len(self.model_terms_matrices),
                epoch_id=eval_epoch,
                force_time_choice=t,
            )
            exp["probe_id"] = probe_id

            params_array = np.array([[self.true_model_params[:]]])
            datum = evaluation_updater.model.simulate_experiment(
                params_array,
                exp,
                # repeat=1000
                repeat=5,
            )
            self.log_print_debug(
                [
                    "(eval) Datum:",
                    datum,
                ]
            )
            evaluation_updater.update(datum, exp)
            eval_epoch += 1

        # Store evaluation
        self.evaluation_normalization_record = evaluation_updater.normalization_record
        if np.isnan(evaluation_updater.log_total_likelihood):
            self.evaluation_log_likelihood = None
            self.evaluation_median_likelihood = None
            self.log_print(["Evaluation ll is nan"])
        else:
            self.evaluation_log_likelihood = evaluation_updater.log_total_likelihood
            # self.evaluation_log_likelihood /= len(self.evaluation_normalization_record) # normalise
            self.evaluation_log_likelihood = qmla.utilities.round_nearest(
                self.evaluation_log_likelihood, 0.01
            )

            self.evaluation_median_likelihood = np.round(
                np.median(evaluation_updater.normalization_record), 2
            )
            self.evaluation_pr0_diffs = np.array(
                evaluation_qinfer_model.store_p0_diffs
            )[:, 0]

        n_terms = len(self.model_terms_names)
        n_samples = len(self.evaluation_normalization_record)

        self.akaike_info_criterion = 2 * n_terms - 2 * self.evaluation_log_likelihood
        try:
            self.akaike_info_criterion_c = self.akaike_info_criterion + 2 * (
                n_terms ** 2 + n_terms
            ) / (n_samples - n_terms - 1)
        except:
            # when n_samples - n_terms - 1 == 0
            # TODO this is made up to avoid errors - find a better way
            # AICc should not be trusted in these cases b/c sample size is so small
            self.akaike_info_criterion_c = self.akaike_info_criterion + 2 * (
                n_terms ** 2 + n_terms
            ) / (n_samples - n_terms)

        self.bayesian_info_criterion = (
            self.num_parameters * np.log(self.num_evaluation_points)
            - 2 * self.evaluation_log_likelihood
        )
        self.evaluation_likelihoods = evaluation_qinfer_model.store_likelihoods
        self.evaluation_summarise_likelihoods = (
            evaluation_qinfer_model.summarise_likelihoods
        )

        self.evaluation_residual_squares = {
            "mean": np.mean(
                np.abs(
                    (
                        np.array(self.evaluation_summarise_likelihoods["system"])
                        - np.array(
                            self.evaluation_summarise_likelihoods["particles_mean"]
                        )
                    )
                )
            ),
            "median": np.median(
                np.abs(
                    (
                        np.array(self.evaluation_summarise_likelihoods["system"])
                        - np.array(
                            self.evaluation_summarise_likelihoods["particles_median"]
                        )
                    )
                )
            ),
            "mean_sq": np.mean(
                (
                    np.array(self.evaluation_summarise_likelihoods["system"])
                    - np.array(self.evaluation_summarise_likelihoods["particles_mean"])
                )
                ** 2
            ),
            "median_sq": np.median(
                (
                    np.array(self.evaluation_summarise_likelihoods["system"])
                    - np.array(
                        self.evaluation_summarise_likelihoods["particles_median"]
                    )
                )
                ** 2
            ),
        }

        self.log_print(
            [
                "Model {} evaluation ll:{} AIC:{}".format(
                    self.model_id,
                    self.evaluation_log_likelihood,
                    self.akaike_info_criterion,
                )
            ]
        )

    ##########
    # Section: Evaluation
    ##########

    def _plot_preliminary_preparation(self):
        r"""
        Prepare model for plots; make directory.
        """
        self.model_learning_plots_directory = os.path.join(
            self.plots_directory, "model_training"
        )
        self.plot_prefix = ""
        if self.is_true_model:
            self.plot_prefix = ""
            # TODO turn back on when not in dev
            # self.plot_prefix = 'true_'

        if not os.path.exists(self.model_learning_plots_directory):
            try:
                os.makedirs(self.model_learning_plots_directory)
            except BaseException:
                pass  # another instance made it at same time

    def _plot_distributions(self):
        r"""
        For each parameter, plot:
        * prior distribution
        * posterior distributino
        * prior distribution for comparison,
        i.e. posterior from learning recast as a unimodal normal
        * true parameters (if applicable)
        * learned parameter estimates
        * covariance matrix between parameters (separate plot)

        # TODO add plotting levels: run, instance, model
        """

        bf_posterior = qi.MultivariateNormalDistribution(
            self.qinfer_updater.est_mean(), self.qinfer_updater.est_covariance_mtx()
        )
        bf_posterior_updater = qi.SMCUpdater(
            model=self.qinfer_model, n_particles=self.num_particles, prior=bf_posterior
        )
        bf_posterior_marginal = [
            bf_posterior_updater.posterior_marginal(idx_param=i)
            for i in range(self.model_constructor.num_terms)
        ]

        num_terms = self.model_constructor.num_terms
        lf = LatexFigure(auto_gridspec=num_terms)

        for param_idx in range(num_terms):
            term = self.model_terms_names[param_idx]
            ax = lf.new_axis()

            # plot prior
            ax.plot(
                self.prior_marginal[param_idx][0],  # locations
                self.prior_marginal[param_idx][1],  # weights
                color="blue",
                ls="-",
                label="Prior",
            )

            # plot posterior
            ax.plot(
                self.posterior_marginal[param_idx][0],  # locations
                self.posterior_marginal[param_idx][1],  # weights
                color="black",
                ls="-",
                label="Posterior",
            )

            # plot posterior_used for BF comparison
            ax.plot(
                bf_posterior_marginal[param_idx][0],  # locations
                bf_posterior_marginal[param_idx][1],  # weights
                color="green",
                ls=":",
                label="Prior for BF",
            )

            # True param
            if term in self.true_param_dict:
                ax.axvline(
                    self.true_param_dict[term], color="red", ls="-.", label="True"
                )

            # Learned param
            try:
                ax.axvline(
                    self.qhl_final_param_estimates[term],
                    color="black",
                    ls="--",
                    label="Learned",
                )
            except BaseException:
                self.log_print(
                    ["{} not in {}".format(term, self.qhl_final_param_estimates)]
                )

            # There is a bug when using log scale which causes overlap on the axis labels:
            # https://stackoverflow.com/questions/46498157/overlapping-axis-tick-labels-in-logarithmic-plots
            # ax.semilogx()
            # ax.semilogy()
            # ax.minorticks_off()
            # latex_name = self.exploration_class.latex_name(term)
            latex_name = self.model_constructor.latex_name_method(term)
            self.log_print(["Latex name:", latex_name])
            ax.set_title(r"{}".format(latex_name))

            if ax.row == 0 and ax.col == lf.gridspec_layout[1] - 1:
                ax.legend()

        lf.fig.text(0.5, 0.04, "Particle locations", ha="center")
        lf.fig.text(0.04, 0.5, "Weights", va="center", rotation="vertical")

        # save the plot
        lf.save(
            os.path.join(
                self.model_learning_plots_directory,
                "{}distributions_{}".format(self.plot_prefix, self.model_id),
            ),
            file_format=self.figure_format,
        )

        # Plot covariance matrix heatmap
        plt.clf()
        lf = LatexFigure()
        ax = lf.new_axis()

        sns.heatmap(self.qinfer_updater.est_covariance_mtx(), ax=ax)
        lf.save(
            os.path.join(
                self.model_learning_plots_directory,
                "{}cov_mtx_final_{}".format(self.plot_prefix, self.model_id),
            ),
            file_format=self.figure_format,
        )

    def _plot_learning_summary(self):
        r"""
        Plot summary of this model's learning:
            * parameter estimates and uncertainties
            * volume of parameter distribution
            * experimental times used
            * (resample points superposed on the above)
            * likelihoods of system/particles
            * difference between system/particles' likelihoods
        """

        terms = self.track_param_estimate_v_epoch.keys()
        num_terms = len(terms)
        extra_plots = [
            "volume",
            # 'quad_loss',  'residuals', 'likelihoods'
        ]
        resample_colour = "grey"
        if num_terms <= 3:
            ncols = num_terms
        else:
            ncols = int(np.ceil(np.sqrt(num_terms)))
        nrows_for_params = int(np.ceil(num_terms / ncols))
        nrows = nrows_for_params + len(extra_plots)
        height_ratios = [1] * nrows_for_params
        height_ratios.extend([ncols * 0.7] * len(extra_plots))
        plt.clf()
        lf = LatexFigure(
            use_gridspec=True,
            gridspec_layout=(nrows, ncols),
            gridspec_params={"height_ratios": height_ratios},
        )

        # Parameter estimates
        for term in terms:
            ax = lf.new_axis(
                # label_position=(-.3, 1.1)
            )
            estimates = self.track_param_estimate_v_epoch[term]
            uncertainty = self.track_param_uncertainty_v_epoch[term]
            lower_bound = estimates - uncertainty
            upper_bound = estimates + uncertainty

            epochs = range(len(estimates))

            ax.plot(epochs, estimates, label="Estimate")
            ax.fill_between(
                epochs, lower_bound, upper_bound, alpha=0.2, label="Uncertainty"
            )

            # if len(self.epochs_after_resampling) > 0:
            #     ax.axvline(
            #         self.epochs_after_resampling[0],
            #         ls='--',
            #         c=resample_colour, alpha=0.5, label='Resample'
            #     )

            #     for e in self.epochs_after_resampling[1:]:
            #         ax.axvline(
            #             e,
            #             ls='--',
            #             c=resample_colour, alpha=0.5,
            #         )

            if term in self.true_param_dict:
                true_param = self.true_param_dict[term]
                ax.axhline(true_param, color="red", ls="--", label="True")

            try:
                # term_latex = self.exploration_class.latex_name(term)
                term_latex = self.model_constructor.latex_name_method(term)
                ax.set_title(term_latex)
                # ax.set_ylabel(term_latex)
            except BaseException:
                self.log_print(["Failed to get latex name"])
                raise
            # ax.set_ylabel('Parameter')
            ax.set_xlabel("Epoch")
            if ax.row == 0 and ax.col == lf.gridspec_layout[1] - 1:
                ax.legend(bbox_to_anchor=(1.1, 1.1))
            if ax.col == 0:
                ax.set_ylabel("Parameter")
            if ax.row == nrows_for_params - 1:
                ax.set_xlabel("Epoch")
            else:
                ax.set_xlabel("")

        if "volume" in extra_plots:
            # Volume and experimental times
            ax = lf.new_axis(
                # label_position=(-0.1, 1.05),
                span=(1, "all")
            )

            ax.plot(
                range(len(self.volume_by_epoch)),
                self.volume_by_epoch,
                label=r"$V$",
                color="k",
            )

            # if len(self.epochs_after_resampling) > 0:
            #     ax.axvline(  # label first resample only
            #         self.epochs_after_resampling[0],
            #         ls='--',
            #         c=resample_colour,
            #         alpha=0.5,
            #         # label='Resample'
            #     )

            #     for e in self.epochs_after_resampling[1:]:
            #         ax.axvline(
            #             e,
            #             ls='--',
            #             c=resample_colour,
            #             alpha=0.5,
            #         )

            # ax.set_title('Volume and Experimental Times')
            ax.set_ylabel("Volume")
            ax.set_xlabel("Epoch")
            ax.set_yscale("log")

            time_ax = ax.twinx()
            times = qmla.utilities.flatten(self.track_experimental_times)
            if self.num_experiments > 100:
                s = 4  # size of time dots
            else:
                s = 7
            time_ax.scatter(
                range(len(self.track_experimental_times)),
                self.track_experimental_times,
                label=r"$t$",
                s=s,
            )
            time_ax.set_ylabel("Time")
            time_ax.semilogy()
            # time_ax.legend(
            #     bbox_to_anchor=(0.85, 1.25),
            #     # loc='lower center'
            # )

            handles, labels = ax.get_legend_handles_labels()
            t_handles, t_labels = time_ax.get_legend_handles_labels()
            handles.extend(t_handles)
            labels.extend(t_labels)

            ax.legend(
                handles,
                labels,
                ncol=2,
                loc="upper center"
                # bbox_to_anchor=(0.4, 1.25)
            )

        if "quad_loss" in extra_plots:
            # Covariance mtx norm and quadratic loss
            ax = lf.new_axis(span=(1, "all"))

            ax.plot(
                range(len(self.track_norm_cov_matrices)),
                self.track_norm_cov_matrices,
                label="Covariance norm",
                color="green",
                ls=":",
            )
            ax.semilogy()
            ax.set_ylabel("Q.L / Norm")

            ax.plot(
                range(len(self.quadratic_losses_record)),
                self.quadratic_losses_record,
                label="Quadratic loss",
                c="orange",
                ls="--",
            )
            ax.legend(
                loc="lower left"
                # bbox_to_anchor=(1.1, 1.1)
            )

        if "likelihoods" in extra_plots:
            # Likelihoods of system and particles
            row += 1
            ax = lf.fig.add_subplot(lf.gs[row, :])

            particle_likelihoods = self.qinfer_model.summarise_likelihoods[
                "particles_median"
            ]
            particle_likelihoods_std = self.qinfer_model.summarise_likelihoods[
                "particles_std"
            ]
            system_likelihoods = self.qinfer_model.summarise_likelihoods["system"]

            ax.plot(
                range(len(system_likelihoods)),
                system_likelihoods,
                # s=3,
                color="red",
                ls="--",
                label="System",
            )

            ax.scatter(
                range(len(particle_likelihoods)),
                particle_likelihoods,
                s=3,
                color="Blue",
                label="Particles",
            )
            ax.fill_between(
                range(len(particle_likelihoods)),
                self.qinfer_model.summarise_likelihoods["particles_upper_quartile"],
                self.qinfer_model.summarise_likelihoods["particles_lower_quartile"],
                alpha=0.3,
                color="Blue",
                label="Particles IQR",
            )
            ax.set_ylabel("$ Pr(0) $")
            ax.set_xlabel("Epoch")
            ax.semilogy()
            ax.legend()

        if "residuals" in extra_plots:
            # Difference | system-pr0 - particles-pr0 |
            row += 1
            ax = lf.fig.add_subplot(lf.gs[row, :])

            self.qinfer_pr0_diff_from_true = np.array(self.qinfer_model.store_p0_diffs)
            medians = self.qinfer_pr0_diff_from_true[:, 0]
            std = self.qinfer_pr0_diff_from_true[:, 1]
            ax.scatter(range(len(medians)), medians, s=3, color="Blue")
            ax.fill_between(
                range(len(medians)),
                medians + std,
                medians - std,
                alpha=0.3,
                color="Blue",
            )
            ax.set_ylabel(r"$ \|Pr(0)_{sys} - Pr(0)_{sim} \|  $")
            ax.set_xlabel("Epoch")
            ax.semilogy()
            try:
                ax.axhline(0.5, label="0.5", ls="--", alpha=0.3, c="grey")
                ax.axhline(
                    0.1,
                    label="0.1",
                    ls=":",
                    alpha=0.3,
                    c="grey",
                )
            except BaseException:
                pass
            ax.legend()

        # Save figure
        lf.save(
            os.path.join(
                self.model_learning_plots_directory,
                "{}learning_summary_{}".format(self.plot_prefix, self.model_id),
            ),
            file_format=self.figure_format,
        )

    def _plot_posterior_mesh_pairwise(self):
        r"""
        Plots the posterior mesh as contours for each pair of parameters.

        Mesh from  qinfer.SMCUpdater.posterior_mesh
        """
        import itertools

        fig, axes = plt.subplots(figsize=(18, 10), constrained_layout=True)
        selected_cmap = plt.cm.Paired

        n_param = self.model_constructor.num_terms
        nrows = ncols = n_param
        gs = GridSpec(
            nrows + 1,
            ncols,
        )

        include_param_self_correlation = True
        if include_param_self_correlation:
            pairs_of_params = list(
                itertools.combinations_with_replacement(range(n_param), 2)
            )
        else:
            pairs_of_params = list(
                itertools.combinations(range(n_param), 2)
            )  # exlcude param with itself
        vmin = 1e3
        vmax = 0
        posterior_meshes = {}
        for i, j in pairs_of_params:
            post_mesh = self.qinfer_updater.posterior_mesh(
                idx_param1=j, idx_param2=i, res1=50, res2=50
            )
            # store the post mesh - don't want to compute twice
            posterior_meshes[i, j] = post_mesh

            # find global min/max contour value for consistency across plots
            if np.min(post_mesh[2]) < vmin:
                vmin = np.min(post_mesh[2])
            if np.max(post_mesh[2]) > vmax:
                vmax = np.max(post_mesh[2])

        for i in range(n_param):
            for j in range(n_param):
                ax = fig.add_subplot(gs[i, j])

                y_term = self.qinfer_model.modelparam_names[i]
                x_term = self.qinfer_model.modelparam_names[j]
                if ax.is_first_col():
                    ax.set_ylabel(
                        # self.exploration_class.latex_name(y_term),
                        self.model_constructor.latex_name_method(y_term),
                        rotation=0,
                    )
                if ax.is_first_row():
                    ax.set_title(
                        # self.exploration_class.latex_name(x_term)
                        self.model_constructor.latex_name_method(x_term)
                    )
                if (i, j) in pairs_of_params:
                    ax.contourf(
                        *posterior_meshes[i, j],
                        vmin=vmin,
                        vmax=vmax,
                        cmap=selected_cmap
                    )

                    if x_term in self.true_param_dict:
                        true_param = self.true_param_dict[x_term]
                        if ax.get_xlim()[0] < true_param < ax.get_xlim()[1]:
                            ax.axvline(true_param, c="black", ls="--", alpha=0.3)
                    if y_term in self.true_param_dict:
                        true_param = self.true_param_dict[y_term]
                        if ax.get_ylim()[0] < true_param < ax.get_ylim()[1]:
                            ax.axhline(true_param, c="black", ls="--", alpha=0.3)

                else:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])

        # Colour bar
        ax = fig.add_subplot(gs[nrows, :])
        m = plt.cm.ScalarMappable(cmap=selected_cmap)
        m.set_array([])
        m.set_clim(vmin, vmax)
        fig.colorbar(m, cax=ax, orientation="horizontal", shrink=0.7)

        # Save
        fig.text(0.5, 0.04, "Posterior mesh", ha="center")
        fig.savefig(
            os.path.join(
                self.model_learning_plots_directory,
                "{}posterior_mesh_pairwise_{}.png".format(
                    self.plot_prefix, self.model_id
                ),
            )
        )

    def _compute_expectation_values(self):
        # TODO replace with call to method which precomputes
        # U = e^{-iH} and takes
        # scipy.linalg.fractional_matrix_power(U, t),
        # instead of computing e^{-iHt} for all values of t

        times = self.experimental_measurement_times
        self.log_print(["Getting expectation values for times:", times])
        if self.model_num_qubits > 5:
            # TODO compute U=e^{-iH} once then it doesn't really matter how many times computed here
            times = times[::10]  # reduce times to compute
        plot_probe = self.plot_probes[self.model_num_qubits]

        self.expectation_values = {
            t: self.exploration_class.get_expectation_value(
                ham=self.learned_hamiltonian, t=t, state=plot_probe
            )
            for t in times
        }

    def _plot_dynamics(self):
        """
        Plots the dynamics reproduced by this model against system data.
        """

        # Plot dynamics of model vs system
        lf = LatexFigure(auto_label=False, fraction=0.75)
        ax = lf.new_axis()
        # System
        times = sorted(self.expectation_values.keys())
        ax.scatter(
            times,
            [self.experimental_measurements[t] for t in times],
            color="red",
            label="System",
            s=3,
        )

        # This model
        ax.plot(
            times,
            [self.expectation_values[t] for t in times],
            color="blue",
            label="Model",
        )
        # label_fontsize = 15
        ax.set_xlim(0, max(times))
        ax.set_ylabel(
            "Expectation value",
        )
        ax.set_xlabel(
            "Time",
        )
        ax.set_title(
            "Dynamics for {}".format(self.model_name_latex),
        )
        ax.legend(
            # prop={'size' : label_fontsize}
        )
        lf.save(
            os.path.join(
                self.model_learning_plots_directory,
                "{}dynamics_{}".format(self.plot_prefix, self.model_id),
            ),
            file_format=self.figure_format,
        )

    ##########
    # Section: Utilities
    ##########

    def log_print(self, to_print_list, log_identifier=None):
        r"""Wrapper for :func:`~qmla.print_to_log`"""

        if log_identifier is None:
            log_identifier = "ModelForLearning {}".format(self.model_id)
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier=log_identifier,
        )

    def log_print_debug(self, to_print_list):
        r"""Log print if global debug_log_print set to True."""

        if self.debug_mode:
            self.log_print(
                to_print_list=to_print_list,
                log_identifier="Debug Model {}".format(self.model_id),
            )

    def _consider_reallocate_resources(self):
        r"""Model might get less resources if it is deemed less complex than others"""

        if self.exploration_class.reallocate_resources:
            base_resources = qmla_core_info_dict["base_resources"]
            this_model_num_qubits = self.model_dimension
            this_model_num_terms = self.model_constructor.num_terms
            max_num_params = self.exploration_class.max_num_parameter_estimate

            new_resources = qmla.utilities.resource_allocation(
                base_qubits=base_resources["num_qubits"],
                base_terms=base_resources["num_terms"],
                max_num_params=max_num_params,
                this_model_qubits=this_model_num_qubits,
                this_model_terms=this_model_num_terms,
                num_experiments=self.num_experiments,
                num_particles=self.num_particles,
            )

            self.num_experiments = new_resources["num_experiments"]
            self.num_particles = new_resources["num_particles"]
            self.log_print(
                "After resource reallocation on {}: {} experiments and {} particles".format(
                    self.model_name, self.num_experiments, self.num_particles
                )
            )

    def _store_prior(self):
        r"""Save the prior raw and as plot."""

        store_all_priors = False  # optional
        if not store_all_priors:
            return

        prior_dir = str(self.results_directory + "priors/QMLA_{}/".format(self.qmla_id))

        if not os.path.exists(prior_dir):
            try:
                os.makedirs(prior_dir)
            except BaseException:
                # if already exists (ie created by another QMLA instance)
                pass
        prior_file = str(prior_dir + "prior_" + str(self.model_id) + ".png")

        individual_terms_in_name = self.model_constructor.terms_names
        latex_terms = []
        for term in individual_terms_in_name:
            # lt = self.exploration_class.latex_name(
            #     name=term
            # )
            lt = self.model_constructor.latex_name_method(term)
            latex_terms.append(lt)

        try:
            qmla.shared_functionality.prior_distributions.plot_prior(
                model_name=self.model_name_latex,
                model_name_individual_terms=latex_terms,
                prior=self.model_prior,
                plot_file=prior_file,
            )
        except BaseException:
            self.log_print(["Failed to plot prior"])

    def plot_distribution_progression(self, renormalise=False, save_to_file=None):
        qmla.analysis.plot_distribution_progression_of_model(
            mod=self,
            num_steps_to_show=2,
            show_means=True,
            renormalise=renormalise,
            save_to_file=save_to_file,
        )
