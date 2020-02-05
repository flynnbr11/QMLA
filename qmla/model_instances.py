from __future__ import print_function  # so print doesn't show brackets
import numpy as np
import scipy as sp
import os
import time
import copy
import qinfer as qi
from psutil import virtual_memory
import redis
import pickle
import matplotlib.pyplot as plt
pickle.HIGHEST_PROTOCOL = 2

import qmla.analysis
import qmla.database_framework as database_framework
import qmla.prior_distributions as Distributions
import qmla.experimental_data_processing as expdt
import qmla.expectation_values as expectation_values
import qmla.get_growth_rule as get_growth_rule
import qmla.logging
import qmla.model_naming as model_naming
from qmla.memory_tests import print_loc
import qmla.qinfer_model_interface as qml_qi
import qmla.redis_settings as rds

global print_mem_status
global debug_log_print
debug_log_print = True
print_mem_status = True
global_print_loc = False

"""
In this file are class definitions:
    - ModelInstanceForLearning
    - ModelInstanceForStorage
    - ModelInstanceForComparison

"""

__all__ = [
    'ModelInstanceForLearning',
    'ModelInstanceForStorage',
    'ModelInstanceForComparison'
]


def resource_allocation(
    base_qubits,
    base_terms,
    max_num_params,
    this_model_qubits,
    this_model_terms,
    num_experiments,
    num_particles,
    given_resource_as_cap=True
):
    new_resources = {}
    if given_resource_as_cap == True:
        # i.e. reduce number particles for models with fewer params
        proportion_of_particles_to_receive = (
            this_model_terms / max_num_params
        )
        print(
            "Model gets proportion of particles:",
            proportion_of_particles_to_receive
        )

        if proportion_of_particles_to_receive < 1:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = max(
                int(
                    proportion_of_particles_to_receive
                    * num_particles
                ),
                10
            )
        else:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = num_particles

    else:
        # increase proportional to number params/qubits
        qubit_factor = float(this_model_qubits / base_qubits)
        terms_factor = float(this_model_terms / base_terms)

        overall_factor = int(qubit_factor * terms_factor)

        if overall_factor > 1:
            new_resources['num_experiments'] = overall_factor * num_experiments
            new_resources['num_particles'] = overall_factor * num_particles
        else:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = num_particles

    print("New resources:", new_resources)
    return new_resources


def time_seconds():
    # return time in h:m:s format for logging.
    import datetime
    now = datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour) + ':' + str(minute) + ':' + str(second))
    return time


class ModelInstanceForLearning():
    """
    Class to learn individual model. Model name is given when initialised.
    A host_name and port_number are given to initialise_model_for_learning.
    The qmd_info dict from Redis is pulled and pickled to find
    the true model and other QMD parameters needed.
    A GenSimModel is set which details the SMCUpdater
    used to update the posterior distribution.
    update_model calls the updater in a loop of n_experiments.
    The final parameter estimates are set as the mean of the
    posterior distribution after n_experiments wherein n_particles
    are sampled per experiment (set in qmd_info).

    """

    def __init__(
        self,
        modelID,
        name,
        qid,
        log_file,
        growth_generator,
        model_terms_matrices,
        model_terms_parameters,
        model_terms_names,
        host_name='localhost',
        port_number=6379,
        **kwargs
    ):
        self.qmla_id = qid
        self.model_id = int(modelID)
        self.model_name = name
        self.log_file = log_file
        self.volume_by_epoch = np.array([])
        self.redis_host = host_name
        self.redis_port_number = port_number

        self.initialise_model_for_learning(
            growth_generator = growth_generator,
            model_terms_matrices = model_terms_matrices,
            model_terms_parameters = model_terms_parameters,
            model_terms_names = model_terms_names,
        )

    def log_print(
        self,
        to_print_list
    ):
        qmla.logging.print_to_log(
            to_print_list = to_print_list, 
            log_file = self.log_file, 
            log_identifier = 'ModelForLearning {}'.format(self.model_id)
        )

    def initialise_model_for_learning(
        self,
        growth_generator,
        model_terms_matrices,
        model_terms_parameters,
        model_terms_names,
        checkloss=True,
        debug_directory=None,
        **kwargs
    ):
        rds_dbs = rds.databases_from_qmd_id(
            self.redis_host, 
            self.redis_port_number, 
            self.qmla_id
        )
        qmd_info_db = rds_dbs['qmd_info_db']
        init_model_print_loc = False
        qmd_info = pickle.loads(qmd_info_db.get('qmla_core_data'))
        self.use_experimental_data = qmd_info['use_experimental_data']
        self.probes_system = pickle.loads(qmd_info_db['ProbeDict'])
        self.probes_simulator = pickle.loads(qmd_info_db['SimProbeDict'])
        self.num_particles = qmd_info['num_particles']
        self.num_experiments = qmd_info['num_experiments']
        self.growth_rule_of_this_model = growth_generator

        try:
            self.growth_class = get_growth_rule.get_growth_generator_class(
                growth_generation_rule = self.growth_rule_of_this_model,
                use_experimental_data = self.use_experimental_data,
                log_file = self.log_file
            )
        except BaseException:
            raise
            # self.growth_class = None


        if qmd_info['reallocate_resources'] == True:

            base_resources = qmd_info['base_resources']
            base_num_qubits = base_resources['num_qubits']
            base_num_terms = base_resources['num_terms']
            this_model_num_qubits = database_framework.get_num_qubits(self.model_name)
            this_model_num_terms = len(
                database_framework.get_constituent_names_from_name(self.model_name)
            )
            max_num_params = self.growth_class.max_num_parameter_estimate

            new_resources = resource_allocation(
                base_qubits=base_num_qubits,
                base_terms=base_num_terms,
                max_num_params=max_num_params,
                this_model_qubits=this_model_num_qubits,
                this_model_terms=this_model_num_terms,
                num_experiments=self.num_experiments,
                num_particles=self.num_particles
            )

            self.num_experiments = new_resources['num_experiments']
            self.num_particles = new_resources['num_particles']
            self.log_print(
                [
                    'After resource reallocation, QML on', self.model_name,
                    '\n\tParticles:', self.num_particles,
                    '\n\tExperiments:', self.num_experiments,
                ]
            )

        self.probe_number = qmd_info['num_probes']
        self.qinfer_resampler_threshold = qmd_info['resampler_thresh']
        self.qinfer_resampler_a = qmd_info['resampler_a']
        self.qinfer_PGH_heuristic_factor = qmd_info['pgh_prefactor']
        self.qinfer_PGH_heuristic_exponent = qmd_info['pgh_exponent']
        self.qinfer_PGH_heuristic_increase_time = qmd_info['increase_pgh_time']
        self.store_particle_locations_and_weights = qmd_info['store_particles_weights']
        # self.do_qhl_plots = qmd_info['qhl_plots']
        self.results_directory = qmd_info['results_directory']
        self.true_model_constituent_operators = qmd_info['true_oplist']
        self.true_model_params = qmd_info['true_params']
        self.true_model_name = qmd_info['true_name']
        self.use_time_dependent_true_model = qmd_info['use_time_dep_true_params']
        self.time_dependent_true_params = qmd_info['time_dep_true_params']
        self.num_time_dependent_true_params = qmd_info['num_time_dependent_true_params']
        self.use_qle = qmd_info['qle']
        self.sigma_threshold = qmd_info['sigma_threshold']
        self.times_to_plot = qmd_info['plot_times']
        self.use_custom_exponentiation = qmd_info['use_exp_custom']
        self.exponentiation_tolerance = qmd_info['compare_linalg_exp_tol']
        self.measurement_class = qmd_info['measurement_type']
        self.experimental_measurements = qmd_info['experimental_measurements']
        self.experimental_measurement_times = qmd_info['experimental_measurement_times']
        self.model_terms_names = model_terms_names
        self.model_name_latex = self.growth_class.latex_name(
            name=self.model_name
        )
        self.model_terms_matrices = np.asarray(model_terms_matrices)
        self.model_terms_parameters = np.asarray([model_terms_parameters[0]])

        individual_terms_in_name = database_framework.get_constituent_names_from_name(
            self.model_name
        )

        for i in range(len(individual_terms_in_name)):
            term = individual_terms_in_name[i]
            term_mtx = database_framework.compute(term)
            if np.all(term_mtx == self.model_terms_matrices[i]) is False:
                print("[QML] UNEQUAL SIM OP LIST / TERM MATRICES.")
                print("==> INSPECT PRIORS ORDERING.")
                self.log_print(
                    [
                        "Term", term,
                        "\ncalculated mtx:", term_mtx,
                        "\nSimOpList:", self.model_terms_matrices[i]
                    ]
                )
            elif term != self.model_terms_names[i]:
                print("!!! Check log -- QML", self.model_name)

                self.log_print(
                    [
                        "term {} != SimOpsNames[i] {}".format(
                            term, self.model_terms_names[i]
                        )
                    ]
                )

        self.check_quadratic_loss = True
        print_loc(print_location=init_model_print_loc)

        num_params = len(self.model_terms_matrices)
        log_identifier=str("QML " + str(self.model_id))
        # self.model_priorSpecificTerms = qmd_info['prior_specific_terms']
        self.model_prior = self.growth_class.get_prior(
            model_name=self.model_name,
            log_file=self.log_file,
            log_identifier = log_identifier, 
        )

        prior_dir = str(
            self.results_directory +
            'priors/QMD_{}/'.format(self.qmla_id)
        )

        if not os.path.exists(prior_dir):
            try:
                os.makedirs(prior_dir)
            except BaseException:
                # if already exists (ie created by another QMD since if test
                # ran...)
                pass
        prior_file = str(
            prior_dir +
            'prior_' +
            str(self.model_id) +
            '.png'
        )

        latex_terms = []
        for term in individual_terms_in_name:
            lt = self.growth_class.latex_name(
                name=term
            )
            latex_terms.append(lt)

        plot_all_priors = True
        if plot_all_priors == True:
            Distributions.plot_prior(
                model_name=self.model_name_latex,
                model_name_individual_terms=latex_terms,
                prior=self.model_prior,
                plot_file=prior_file,
            )

        self.qinfer_model = qml_qi.QInferModelQML(
            oplist=self.model_terms_matrices,
            modelparams=self.model_terms_parameters,
            true_oplist=self.true_model_constituent_operators,
            trueparams=self.true_model_params,
            truename=self.true_model_name,
            use_time_dep_true_model=self.use_time_dependent_true_model,
            time_dep_true_params=self.time_dependent_true_params,
            num_time_dep_true_params=self.num_time_dependent_true_params,
            num_probes=self.probe_number,
            measurement_type=self.measurement_class,
            growth_generation_rule=self.growth_rule_of_this_model,
            use_experimental_data=self.use_experimental_data,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            probe_dict=self.probes_system,
            sim_probe_dict=self.probes_simulator,
            probecounter=0,
            solver='scipy',
            trotter=True,
            qle=self.use_qle,
            use_exp_custom=self.use_custom_exponentiation,
            exp_comparison_tol=self.exponentiation_tolerance,
            enable_sparse=True,
            model_name=self.model_name,
            log_file=self.log_file,
            log_identifier=log_identifier
        )

        self.qinfer_updater = qi.SMCUpdater(
            self.qinfer_model,
            self.num_particles,
            self.model_prior,
            resample_thresh=self.qinfer_resampler_threshold,
            resampler=qi.LiuWestResampler(a=self.qinfer_resampler_a),
            debug_resampling=False
        )

        self.initial_prior = []
        for i in range(len(self.model_terms_parameters[0])):
            self.initial_prior.append(
                self.qinfer_updater.posterior_marginal(idx_param=i)
            )

        self.inversion_field = [
            item[0]
            for item
            in self.qinfer_model.expparams_dtype[1:]
        ]

        self.model_heuristic = self.growth_class.model_heuristic_function(
            updater=self.qinfer_updater,
            oplist=self.model_terms_matrices,
            inv_field=self.inversion_field,
            increase_time=self.qinfer_PGH_heuristic_increase_time,
            pgh_exponent=self.qinfer_PGH_heuristic_exponent,
            time_list=self.times_to_plot,
            num_experiments=self.num_experiments,
        )
        self.model_heuristic_class = self.model_heuristic.__class__.__name__

        # if checkloss == True or self.check_quadratic_loss==True:
        #     self.quadratic_losses = np.array([])
        self.quadratic_losses = []
        self.track_total_log_likelihood = np.array([])
        self.track_experimental_times = np.array([])  # only for debugging
        self.particles = np.array([])
        self.weights = np.array([])
        self.epochs_after_resampling = []
        # self.ExperimentsHistory = np.array([])
        # average and standard deviation at the final step of the parameters
        # inferred distributions
        self.final_learned_params = np.empty([len(self.model_terms_matrices), 2])
        # print_loc(print_location=init_model_print_loc)
        # self.log_print(['Initialization Ready'])

    def update_model(
        self,
        # n_experiments=None,
        # sigma_threshold=10**-13,
        checkloss=False #TODO is this needed?
    ):
        # self.num_experiments = n_experiments

        # if self.check_quadratic_loss == True:
        #     self.quadratic_losses = np.empty(self.num_experiments)
        self.covariances = np.empty(self.num_experiments)
        self.track_mean_params = [self.qinfer_updater.est_mean()]
        self.track_covariance_matrices = []
        self.track_param_dist_widths = []
        self.track_posterior_dist = []
        self.track_prior_means = []
        self.track_prior_std_dev = []
        # self.track_posterior_distMarginal = np.empty(self.num_experiments, self.NumParameters)
        self.track_experimental_times = np.empty(self.num_experiments)  # only for debugging

        self.particles = np.empty([self.num_particles,
                                   len(self.model_terms_parameters[0]), self.num_experiments]
                                  )
        self.weights = np.empty([self.num_particles, self.num_experiments])
        self.DistributionMeans = np.empty([self.num_experiments])
        self.DistributionStdDevs = np.empty([self.num_experiments])

        # self.new_experiment = self.model_heuristic()
        # This is the value of the Norm of the COvariance matrix which stops
        # the IQLE
        # self.sigma_threshold = sigma_threshold
        # self.model_log_total_likelihood = []  # log_total_likelihood

        # self.datum_gather_cumulative_time = 0
        # self.update_cumulative_time = 0
        # self.learned_est_means = {}
        self.true_model_params_dict = {}

        true_params_names = database_framework.get_constituent_names_from_name(
            self.true_model_name
        )
        if self.use_experimental_data == False:
            for i in range(len(true_params_names)):
                term = true_params_names[i]
                true_param_val = self.true_model_params[i]
                self.true_model_params_dict[term] = true_param_val

        all_params_for_q_loss = list(
            set(true_params_names).union(self.model_terms_names)
        )
        param_indices = {}
        for op_name in self.model_terms_names:
            param_indices[op_name] = self.model_terms_names.index(op_name)

        print_frequency = max(
            int(self.num_experiments / 10),
            5
        )
        # print("[QML] STARTING QHL UPDATES")
        # true_params = np.array([[self.true_model_params[0]]])
        for istep in range(self.num_experiments):
            # print("Epoch", istep)
            if (istep % print_frequency == 0):
                # print so we can see how far along algorithm is.
                self.log_print(
                    [
                        "Epoch", istep
                    ]
                )

            # print("[QML] Calling heuristic")
            if istep == 0:
                param_estimates = self.qinfer_updater.est_mean()
            else:
                param_estimates = self.track_mean_params[-1]
            self.new_experiment = self.model_heuristic(
                test_param="from QML",
                num_params=len(self.model_terms_names),
                epoch_id=istep,
                current_params=param_estimates
            )
            print_loc(global_print_loc)
            # TODO prefactor, if used, should be inside specific heuristic
            self.new_experiment[0][0] = self.new_experiment[0][0] * self.qinfer_PGH_heuristic_factor
            if self.use_experimental_data:
                t = self.new_experiment[0][0]
                nearest = expdt.nearestAvailableExpTime(
                    times=self.experimental_measurement_times,
                    t=t
                )
                self.new_experiment[0][0] = nearest

            # self.NumExperimentsToDate += 1
            print_loc(global_print_loc)
            if istep == 0:
                print_loc(global_print_loc)
                self.log_print(['Initial time selected > ',
                                str(self.new_experiment[0][0])]
                               )

            self.track_experimental_times[istep] = self.new_experiment[0][0]

            before_datum = time.time()

            # self.log_print(
            #    [
            #    'Getting Datum',
            #    '\nSimParams:', self.model_terms_parameters,
            #    '\nExperiment:', self.new_experiment
            #    ]
            # )

            self.datum_from_experiment = self.qinfer_model.simulate_experiment(
                self.model_terms_parameters,
                self.new_experiment,
                repeat=1
            )  # TODO reconsider repeat number
            # self.datum_from_experiment = 1
            after_datum = time.time()
            # self.datum_gather_cumulative_time += after_datum - before_datum

            # exp_t = self.new_experiment[0][0]
            before_upd = time.time()
            # Call updater to update distribution based on datum
            try:
                # print("[QML] calling updater")
                self.qinfer_updater.update(
                    self.datum_from_experiment,
                    self.new_experiment
                )
            except RuntimeError as e:
                import sys
                self.log_print(
                    [
                        "RuntimeError from updater on model ID ",
                        self.model_id,
                        ":",
                        self.model_name,
                        "\nError:\n",
                        str(e)
                    ]
                )
                print("\n\nEXITING; Inspect log\n\n")
                raise NameError("Qinfer update failure")
                sys.exit()

            after_upd = time.time()
            # self.update_cumulative_time += after_upd - before_upd

            if self.qinfer_updater.just_resampled is True:
                self.epochs_after_resampling.append(istep)

            print_loc(global_print_loc)
            # self.covmat = self.qinfer_updater.est_covariance_mtx()
            self.volume_by_epoch = np.append(
                self.volume_by_epoch,
                np.linalg.det(
                    sp.linalg.sqrtm(
                        # self.covmat
                        self.qinfer_updater.est_covariance_mtx()
                    )  # TODO seems unnecessary to do this every epoch - every 10th would be enough for plot
                )
            )

            self.track_mean_params.append(self.qinfer_updater.est_mean())
            self.track_param_dist_widths.append(
                np.sqrt(
                    np.diag(self.qinfer_updater.est_covariance_mtx())
                )
            )
            # TODO this doesn't seem necessary to store
            self.track_covariance_matrices.append(self.qinfer_updater.est_covariance_mtx())
            prior_sample = self.qinfer_updater.sample(int(5))

            these_means = []
            these_std = []
            for i in range(len(self.model_terms_matrices)):
                these_means.append(np.mean(prior_sample[:, i]))
                these_std.append(np.std(prior_sample[:, i]))

            self.track_posterior_dist.append(prior_sample)
            self.track_prior_means.append(these_means)
            # TODO get this from self.qinfer_updater.est_mean()
            self.track_prior_std_dev.append(these_std)
            # self.track_posterior_distMarginal.append(self.qinfer_updater.posterior_marginal())

            print_loc(global_print_loc)
            self.covariances[istep] = np.linalg.norm(
                self.qinfer_updater.est_covariance_mtx()
            )
            print_loc(global_print_loc)
            self.particles[:, :, istep] = self.qinfer_updater.particle_locations
            #self.weights[:, istep] = self.qinfer_updater.particle_weights

            self.NewEval = self.qinfer_updater.est_mean()
            print_loc(global_print_loc)

            if (
                checkloss == True
                and
                self.use_experimental_data == False
                # and istep%10 == 0
            ):
                quadratic_loss = 0
                for param in all_params_for_q_loss:
                    if param in self.model_terms_names:
                        learned_param = self.NewEval[param_indices[param]]
                    else:
                        learned_param = 0

                    if param in true_params_names:
                        true_param = self.true_model_params_dict[param]
                    else:
                        true_param = 0
                    # print("[QML] param:", param, "learned param:", learned_param, "\t true param:", true_param)
                    quadratic_loss += (learned_param - true_param)**2
                self.quadratic_losses.append(quadratic_loss)

                if False:  # can be reinstated to stop learning when volume converges
                    self.log_print(['Final time selected > ',
                                    str(self.new_experiment[0][0])]
                                   )
                    print('Exiting learning for Reaching Num. Prec. \
                         -  Iteration Number ' + str(istep)
                          )

                    for iterator in range(len(self.final_learned_params)):
                        self.final_learned_params[iterator] = [
                            # final params and sigmas
                            # np.mean(self.particles[:,iterator,istep]),
                            # TODO should this be gotten from updater.est_covariance_mtx()?
                            # np.std(self.particles[:,iterator,istep])
                            self.qinfer_updater.est_mean(),
                            np.sqrt(np.diag(updater.est_covariance_mtx()))
                        ]
                        print('Final Parameters mean and stdev:' +
                              str(self.final_learned_params[iterator])
                              )
                    self.model_log_total_likelihood = (
                        self.qinfer_updater.log_total_likelihood
                    )
                    # self.quadratic_losses=(np.resize(self.quadratic_losses, (1,istep)))[0]
                    self.covariances = (np.resize(self.covariances, (1, istep)))[0]
                    self.particles = self.particles[:, :, 0:istep]
                    self.weights = self.weights[:, 0:istep]
                    self.track_experimental_times = self.track_experimental_times[0:istep]
                    break

            if self.covariances[istep] < self.sigma_threshold and False:
                # can be reinstated to stop learning when volume converges
                self.log_print(['Final time selected > ',
                                str(self.new_experiment[0][0])]
                               )
                self.log_print(['Exiting learning for Reaching Cov. \
                    Norm. Thrshold of ', str(self.covariances[istep])]
                               )
                self.log_print([' at Iteration Number ', str(istep)])
                for iterator in range(len(self.final_learned_params)):
                    self.final_learned_params[iterator] = [
                        #                        np.mean(self.particles[:,iterator,istep]),
                        self.qinfer_updater.est_mean(),
                        np.std(self.particles[:, iterator, istep])
                    ]
                    self.log_print(['Final Parameters mean and stdev:',
                                    str(self.final_learned_params[iterator])]
                                   )
                self.model_log_total_likelihood = self.qinfer_updater.log_total_likelihood
                # if checkloss == True:
                #     self.quadratic_losses=(
                #         (np.resize(self.quadratic_losses, (1,istep)))[0]
                #     )

                self.covariances = (np.resize(self.covariances, (1, istep)))[0]
                self.particles = self.particles[:, :, 0:istep]
                self.weights = self.weights[:, 0:istep]
                self.track_experimental_times = self.track_experimental_times[0:istep]

                break

            if istep == self.num_experiments - 1:
                self.log_print(["Results for QHL on ", self.model_name])
                self.log_print(
                    [
                        'Final time selected >',
                        str(self.new_experiment[0][0])
                    ]
                )
                self.model_log_total_likelihood = self.qinfer_updater.log_total_likelihood

                #self.log_print(['Sizes:\t updater:', asizeof.asizeof(self.qinfer_updater), '\t GenSim:', asizeof.asizeof(self.qinfer_model) ])
                self.learned_parameters_qhl = {}
                self.final_sigmas_qhl = {}
                cov_mat = self.qinfer_updater.est_covariance_mtx()
                for iterator in range(len(self.final_learned_params)):
                    self.final_learned_params[iterator] = [
                        #                        np.mean(self.particles[:,iterator,istep-1]),
                        self.qinfer_updater.est_mean()[iterator],
                        np.sqrt(cov_mat[iterator][iterator])
                        # np.std(self.particles[:,iterator,istep-1])
                        # self.qinfer_updater.est_mean(),
                        # np.sqrt(np.diag(updater.est_covariance_mtx()))
                    ]
                    self.log_print([
                        'Final Parameters mean and stdev (term ',
                        self.model_terms_names[iterator], '):',
                        str(self.final_learned_params[iterator])]
                    )
                    self.learned_parameters_qhl[self.model_terms_names[iterator]] = (
                        self.final_learned_params[iterator][0]
                    )
                    self.final_sigmas_qhl[self.model_terms_names[iterator]] = (
                        self.final_learned_params[iterator][1]
                    )

    def learned_info_dict(self):
        """
        Place essential information after learning has occured into a dict.
        This can be used to recreate the model on another node.
        """

        all_post_margs = []
        for i in range(len(self.final_learned_params)):
            all_post_margs.append(
                self.qinfer_updater.posterior_marginal(idx_param=i)
            )

        learned_info = {}
        learned_info['times'] = self.track_experimental_times
        learned_info['final_params'] = self.final_learned_params
        learned_info['normalization_record'] = self.qinfer_updater.normalization_record
        learned_info['log_total_likelihood'] = self.qinfer_updater.log_total_likelihood
        learned_info['data_record'] = self.qinfer_updater.data_record
        learned_info['name'] = self.model_name
        learned_info['model_id'] = self.model_id
        # TODO regenerate this from mean and std_dev instead of saving it
        learned_info['updater'] = pickle.dumps(self.qinfer_updater, protocol=2)
        # TODO regenerate this from mean and std_dev instead of saving it
        learned_info['final_prior'] = self.qinfer_updater.prior
        learned_info['initial_prior'] = self.initial_prior
        learned_info['sim_op_names'] = self.model_terms_names
        learned_info['final_cov_mat'] = self.qinfer_updater.est_covariance_mtx()
        learned_info['est_mean'] = self.qinfer_updater.est_mean()
        """
        1st is still the initial prior!
        that object does not get updated by the learning!
        modify e.g. using the functions defined in /QML_lib/Distrib.py
        2nd is fine
        """

        learned_info['posterior_marginal'] = all_post_margs
        learned_info['initial_params'] = self.model_terms_parameters
        learned_info['volume_list'] = self.volume_by_epoch
        learned_info['track_eval'] = self.track_mean_params
        learned_info['track_cov_matrices'] = self.track_covariance_matrices
        learned_info['track_param_sigmas'] = self.track_param_dist_widths
        learned_info['track_posterior'] = self.track_posterior_dist
        # repeat of track param sigmas?
        learned_info['track_prior_means'] = self.track_prior_means
        learned_info['track_prior_std_devs'] = self.track_prior_std_dev
        # learned_info['track_posterior_marginal'] = self.track_posterior_distMarginal
        learned_info['resample_epochs'] = self.epochs_after_resampling
        learned_info['quadratic_losses'] = self.quadratic_losses
        learned_info['learned_parameters'] = self.learned_parameters_qhl
        learned_info['final_sigmas'] = self.final_sigmas_qhl
        learned_info['cov_matrix'] = self.qinfer_updater.est_covariance_mtx()
        learned_info['num_particles'] = self.num_particles
        learned_info['num_experiments'] = self.num_experiments
        learned_info['growth_generator'] = self.growth_rule_of_this_model
        learned_info['heuristic'] = self.model_heuristic_class
        if self.store_particle_locations_and_weights:
            self.log_print(
                [
                    "Storing particles and weights for model",
                    self.model_id
                ]
            )
            learned_info['particles'] = self.particles
            learned_info['weights'] = self.weights

        return learned_info

    # def store_particles(self, debug_dir=None):
    #     if debug_dir is not None:
    #         save_dir = debug_dir
    #     elif self.debug_directory is not None:
    #         save_dir = self.debug_directory
    #     else:
    #         self.log_print([
    #             "Need to pass debug_dir to QML.debug_save function"]
    #         )
    #         return False
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     save_file = save_dir + '/particles_mod_' + str(self.model_id) + '.dat'

    #     particle_file = open(save_file, 'w')
    #     particle_file.write("\n".join(str(elem) for elem in self.particles.T))
    #     particle_file.close()

    # def store_covariances(self, debug_dir=None):
    #     if debug_dir is not None:
    #         save_dir = debug_dir
    #     elif self.debug_directory is not None:
    #         save_dir = self.debug_directory
    #     else:
    #         self.log_print(
    #             ["Need to pass debug_dir to QML.debug_save function"])
    #         return False
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     save_file = save_dir + '/covariances_mod_' + str(self.model_id) + '.dat'
    #     particle_file = open(save_file, 'w')
    #     particle_file.write("\n".join(str(elem) for elem in self.covariances))
    #     particle_file.close()

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


### Reduced class with only essential information saved ###
class ModelInstanceForStorage():
    """
    Class holds what is required for updates only.
    i.e.
        - times learned over
        - final parameters
        - oplist
        - true_oplist (?) needed to regenerate GenSimModel identically (necessary?)
        - true_params (?)
        - resample_thresh
        - resample_a [are resampling params needed only for updates?]
        - Prior (specified by mean and std_dev?)

    Then initialises an updater and GenSimModel which are used for updates.
    """

    def __init__(
        self,
        model_name,
        sim_oplist,
        true_oplist,
        true_params,
        modelID,
        # numparticles,
        # resample_thresh=0.5,
        # resample_a=0.9,
        # qle=True,
        # probe_dict= None,
        qid=0,
        host_name='localhost',
        port_number=6379,
        log_file='QMD_log.log'
    ):

        rds_dbs = rds.databases_from_qmd_id(
            host_name,
            port_number,
            qid
        )
        qmd_info_db = rds_dbs['qmd_info_db']
        #print("In reduced model. rds_dbs:", rds_dbs)
      #  print("QMD INFO DB has type", type(qmd_info_db), "\n", qmd_info_db)

        self.model_name = model_name
        self.model_id = modelID
        self.model_terms_matrices = sim_oplist
        self.model_id = modelID
        qmd_info = pickle.loads(qmd_info_db.get('qmla_core_data'))
        self.probes_system = pickle.loads(qmd_info_db['ProbeDict'])
        self.probes_simulator = pickle.loads(qmd_info_db['SimProbeDict'])
        self.measurement_class = qmd_info['measurement_type']
        self.experimental_measurements = qmd_info['experimental_measurements']
        self.use_experimental_data = qmd_info['use_experimental_data']
        # self.num_particles = qmd_info['num_particles']
        # self.num_experiments = qmd_info['num_experiments']
        self.probe_number = qmd_info['num_probes']
        self.qinfer_resampler_threshold = qmd_info['resampler_thresh']
        self.qinfer_resampler_a = qmd_info['resampler_a']
        self.qinfer_PGH_heuristic_factor = qmd_info['pgh_prefactor']
        self.true_model_constituent_operators = qmd_info['true_oplist']
        self.true_model_params = qmd_info['true_params']
        self.true_model_name = qmd_info['true_name']
        self.probes_for_plots = pickle.load(
            open(qmd_info['probes_plot_file'], 'rb')
        )
        self.times_to_plot = qmd_info['plot_times']
        self.use_qle = qmd_info['qle']
        self.use_custom_exponentiation = qmd_info['use_exp_custom']
        self.store_particle_locations_and_weights = qmd_info[
            'store_particles_weights'
        ]
        self.model_bayes_factors = {}
        self.NumQubits = database_framework.get_num_qubits(self.model_name)
        self.ProbeDimension = self.NumQubits
        self.redis_host_name = host_name
        self.redis_port_number = port_number
        self.qmla_id = qid
        self.log_file = log_file
        self.expectation_values = {}
        self.values_updated = False

    def log_print(self, to_print_list):
        identifier = str(str(time_seconds()) +
                         "[QML:Reduced " +
                         str(self.model_id) + "; QMD " + str(self.qmla_id) + "]"
                         )
        if not isinstance(to_print_list, list):
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file)

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
            rds_dbs = rds.databases_from_qmd_id(
                self.redis_host_name,
                self.redis_port_number,
                self.qmla_id
            )
            learned_models_info = rds_dbs['learned_models_info']
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
                    )  # TODO telling pickle which encoding was used, though I'm not sure why/where that encoding was given...
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
            self.Times = list(learned_info['times'])
            # should be final params from learning process
            self.final_learned_params = learned_info['final_params']
            # TODO this won't work for multiple parameters
            self.model_terms_parameters_Final = np.array([[self.final_learned_params[0, 0]]])
            self.SimOpNames = learned_info['sim_op_names']
            # TODO this can be recreated from finalparams, but how for multiple
            # params?
            self.model_prior = learned_info['final_prior']
            self.NormalizationRecord = learned_info['normalization_record']
            self.log_total_likelihod = learned_info['log_total_likelihood']
            self.RawVolumeList = learned_info['volume_list']
            self.volume_by_epoch = {}
            for i in range(len(self.RawVolumeList)):
                self.volume_by_epoch[i] = self.RawVolumeList[i]

            self.track_mean_params = np.array(learned_info['track_eval'])
            self.track_covariance_matrices = np.array(
                learned_info['track_cov_matrices'])
            self.track_param_dist_widths = np.array(
                learned_info['track_param_sigmas'])
            self.track_prior_means = np.array(learned_info['track_prior_means'])
            self.track_posterior_dist = np.array(learned_info['track_posterior'])
            self.track_prior_std_dev = np.array(
                learned_info['track_prior_std_devs'])
            # self.track_posterior_distMarginal = np.array(learned_info['track_posterior_marginal'])

            self.epochs_after_resampling = learned_info['resample_epochs']
            self.QuadraticLosses = learned_info['quadratic_losses']
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
                # raise
                self.growth_class = None
            self.model_heuristic_class = learned_info['heuristic']

            self.model_name_latex = self.growth_class.latex_name(
                name=self.model_name
            )

            self.Trackplot_parameter_estimates = {}
            num_params = np.shape(self.track_mean_params)[1]
            max_exp = np.shape(self.track_mean_params)[0] - 1
            for i in range(num_params):
                for term in self.learned_parameters_qhl.keys():
                    if self.learned_parameters_qhl[term] == self.track_mean_params[max_exp][i]:
                        self.Trackplot_parameter_estimates[term] = self.track_mean_params[:, i]

            try:
                self.particles = np.array(learned_info['particles'])
                self.weights = np.array(learned_info['weights'])
            except BaseException:
                self.particles = 'Particles not stored.'
                self.weights = 'Weights not stored.'

            sim_params = list(self.final_learned_params[:, 0])
            try:
                self.LearnedHamiltonian = np.tensordot(
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

            # if self.model_id not in sorted(fitness_parameters.keys()):
            #     fitness_parameters[self.model_id] = {}
            # fitness_parameters[self.model_id]['r_squared'] =  0.75

    def compute_expectation_values(
        self,
        times=[],
        # plot_probe_path = None,
        # probe = None #  TODO generalise probe
    ):
        # TODO expectation_values dict only for |++> probe as is.
        # if probe is None and plot_probe_path is None:
        #     probe  = expectation_values.n_qubit_plus_state(self.NumQubits)
        # else:

        #     plot_probe_dict = pickle.load(
        #         open(plot_probe_path, 'rb')
        #     )
        #     probe = plot_probe_dict[self.NumQubits]

        probe = self.probes_for_plots[self.ProbeDimension]

        # self.log_print(
        #     [
        #     "Computing expectation values.",
        #     "\nMeasurement Type:", self.measurement_class,
        #     "\nLearnedHamiltonian", self.LearnedHamiltonian,
        #     # "\nPlotProbePath:", plot_probe_path,
        #     "\nProbe:", probe,
        #     "\nTimes:", times
        #     ]
        # )

        present_expec_val_times = sorted(
            list(self.expectation_values.keys())
        )

        required_times = sorted(
            list(set(times) - set(present_expec_val_times))
        )

        for t in required_times:
            self.expectation_values[t] = self.growth_class.expectation_value(
                ham=self.LearnedHamiltonian,
                t=t,
                state=probe,
                log_file=self.log_file,
                log_identifier='[QML - compute expectation values]'
            )
        # self.raw_expectation_values = np.array([
        #     self.expectation_values[t] for t in required_times
        # ])
        # self.times = np.array(
        #     sorted(list(self.expectation_times.keys()))
        # )

    def r_squared(
        self,
        plot_probes,
        times=None,
        min_time=0,
        max_time=None
    ):
        # TODO recheck R squared functions eg which probe used
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

        min_time = expdt.nearestAvailableExpTime(exp_times, min_time)
        max_time = expdt.nearestAvailableExpTime(exp_times, max_time)
        min_data_idx = exp_times.index(min_time)
        max_data_idx = exp_times.index(max_time)
        exp_times = exp_times[min_data_idx:max_data_idx]
        exp_data = [
            self.experimental_measurements[t] for t in exp_times
        ]
        probe = self.probes_for_plots[self.ProbeDimension]

        datamean = np.mean(exp_data[0:max_data_idx])
        # datavar = np.sum( (exp_data[0:max_data_idx] - datamean)**2  )

        total_sum_of_squares = 0
        for d in exp_data:
            total_sum_of_squares += (d - datamean)**2
        self.true_exp_val_mean = datamean
        self.total_sum_of_squares = total_sum_of_squares

        ham = self.LearnedHamiltonian
        sum_of_residuals = 0
        available_expectation_values = sorted(
            list(self.expectation_values.keys()))

        chi_squared = 0
        self.r_squared_of_t = {}
        for t in exp_times:
            # TODO if use_experimental_data is False, call full expectatino
            # value function isntead
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
                "[QML - r_squared] Total sum of squares is 0",
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
        num_points=10  # maximum number of epochs to take R^2 at
    ):
        # TODO recheck R squared functions eg which probe used
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

        min_time = expdt.nearestAvailableExpTime(
            exp_times,
            min_time
        )
        max_time = expdt.nearestAvailableExpTime(
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

        # exp_data = exp_data[0::10]
        # probe = np.array([0.5, 0.5, 0.5, 0.5+0j]) # TODO generalise
        # probe  = plot_probes[self.NumQubits]
        probe = self.probes_for_plots[self.ProbeDimension]

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


#        self.qinfer_model = qml_qi.qinfer_model_interface(oplist=self.model_terms_matrices, modelparams=self.model_terms_parameters_Final, true_oplist = self.true_model_constituent_operators, trueparams = self.true_model_params, truename=self.true_model_name,             use_experimental_data = self.use_experimental_data,
#            experimental_measurements = self.experimental_measurements,
#            experimental_measurement_times=(
#                self.experimental_measurement_times
#            ),
# model_name=self.model_name, probe_dict = self.probes_system)    # probelist=self.true_model_constituent_operators,
#        self.qinfer_updater = qi.SMCUpdater(self.qinfer_model, self.num_particles, self.model_prior, resample_thresh=self.qinfer_resampler_threshold , resampler = qi.LiuWestResampler(a=self.qinfer_resampler_a), debug_resampling=False) ## TODO does the reduced model instance need an updater or GenSimModel?
#        self.qinfer_updater.NormalizationRecord = self.NormalizationRecord


class ModelInstanceForComparison():
    """
    When Bayes factors are calculated remotely (ie on RQ workers),
    they require SMCUpdaters etc to do calculations.
    This class captures the minimum required to enable these calculations.
    These are pickled by the ModelInstanceForLearning to a redis database:
    this class unpickles the useful information and generates new instances
    of GenSimModel etc. to use in those calculations.

    """

    def __init__(
        self,
        modelID,
        host_name='localhost',
        port_number=6379,
        qid=0,
        log_file='QMD_log.log',
        learned_model_info=None,
    ):

        rds_dbs = rds.databases_from_qmd_id(
            host_name,
            port_number,
            qid
        )
        self.log_file = log_file
        self.qmla_id = qid

        qmd_info_db = rds_dbs['qmd_info_db']

        qmd_info = pickle.loads(qmd_info_db.get('qmla_core_data'))
        self.probes_system = pickle.loads(qmd_info_db['ProbeDict'])
        self.probes_simulator = pickle.loads(qmd_info_db['SimProbeDict'])

        self.model_id = modelID
        self.num_particles = qmd_info['num_particles']
        self.probe_number = qmd_info['num_probes']
        self.PlotProbePath = qmd_info['probes_plot_file']
        self.qinfer_resampler_threshold = qmd_info['resampler_thresh']
        self.qinfer_resampler_a = qmd_info['resampler_a']
        self.qinfer_PGH_heuristic_factor = qmd_info['pgh_prefactor']
        self.true_model_constituent_operators = qmd_info['true_oplist']
        self.true_model_params = qmd_info['true_params']
        self.true_model_name = qmd_info['true_name']
        self.use_custom_exponentiation = qmd_info['use_exp_custom']
        self.measurement_class = qmd_info['measurement_type']
        self.use_experimental_data = qmd_info['use_experimental_data']
        self.experimental_measurements = qmd_info['experimental_measurements']
        self.experimental_measurement_times = qmd_info['experimental_measurement_times']
        self.results_directory = qmd_info['results_directory']

        # Get model specific data
        learned_models_info = rds_dbs['learned_models_info']
        model_id_float = float(modelID)
        model_id_str = str(model_id_float)
        try:
            learned_model_info = pickle.loads(
                learned_models_info.get(model_id_str),
                encoding='latin1'
            )
        except BaseException:
            learned_model_info = pickle.loads(
                learned_models_info.get(model_id_str)
            )

        self.model_name = learned_model_info['name']
        self.log_print(
            [
                "Name:", self.model_name
            ]
        )
        op = database_framework.Operator(self.model_name)
        # todo, put this in a lighter function
        self.model_terms_matrices = op.constituents_operators
        self.Times = learned_model_info['times']
        self.final_learned_params = learned_model_info['final_params']
        # TODO this won't work for multiple parameters
        self.model_terms_parameters_Final = np.array(self.final_learned_params)
        # self.model_terms_parameters_Final = np.array([[self.final_learned_params[0,0]]]) # TODO
        # this won't work for multiple parameters

        # print("[QML {}] \nSimParams_Final: {} \nSimOpList: {}".format(
        #     self.model_name,
        #     self.model_terms_parameters_Final,
        #     self.model_terms_matrices
        #     )
        # )
        self.InitialParams = learned_model_info['initial_params']
        self.growth_rule_of_this_model = learned_model_info['growth_generator']
        self.growth_class = get_growth_rule.get_growth_generator_class(
            growth_generation_rule=self.growth_rule_of_this_model,
            use_experimental_data=self.use_experimental_data,
            log_file=self.log_file
        )
        # TODO this can be recreated from finalparams, but how for multiple
        # params?
        self.model_prior = learned_model_info['final_prior']
        self.PosteriorMarginal = learned_model_info['posterior_marginal']
        self.initial_prior = learned_model_info['initial_prior']
        self.NormalizationRecord = learned_model_info['normalization_record']
        self.log_total_likelihood = learned_model_info['log_total_likelihood']
        self.learned_parameters_qhl = learned_model_info['learned_parameters']
        self.final_sigmas_qhl = learned_model_info['final_sigmas']
        self.FinalCovarianceMatrix = learned_model_info['final_cov_mat']
        log_identifier = str("Bayes " + str(self.model_id))

        self.qinfer_model = qml_qi.QInferModelQML(
            oplist=self.model_terms_matrices,
            modelparams=self.model_terms_parameters_Final,
            true_oplist=self.true_model_constituent_operators,
            trueparams=self.true_model_params,
            truename=self.true_model_name,
            measurement_type=self.measurement_class,
            growth_generation_rule=self.growth_rule_of_this_model,
            use_experimental_data=self.use_experimental_data,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=(
                self.experimental_measurement_times
            ),
            model_name=self.model_name,
            num_probes=self.probe_number,
            probe_dict=self.probes_system,
            sim_probe_dict=self.probes_simulator,
            log_file=self.log_file,
            log_identifier=log_identifier
        )

        # recreate prior using final params instead of pickling

        # Plot posterior distribution after learning.
        # model_terms = database_framework.get_constituent_names_from_name(
        #     self.model_name
        # )
        # model_name_individual_terms = [
        #     self.growth_class.latex_name(t)
        #     for t in model_terms
        # ]

        # Distributions.plot_prior(
        #     model_name = self.model_name,
        #     model_name_individual_terms = model_name_individual_terms,
        #     prior = posterior_distribution,
        #     plot_file = str(
        #         self.results_directory
        #         + '/priors/posterior_{}_{}.png'.format(
        #             self.qmla_id,
        #             int(self.model_id)
        #         )
        #     )
        # )
        self.reconstruct_updater = True
        time_s = time.time()
        if self.reconstruct_updater == True:
            posterior_distribution = qi.MultivariateNormalDistribution(
                # final_params,
                learned_model_info['est_mean'],
                self.FinalCovarianceMatrix
                # final_cov_mat
            )

            self.qinfer_updater = qi.SMCUpdater(
                model=self.qinfer_model,
                n_particles=self.num_particles,
                prior=posterior_distribution,
                # prior = self.model_prior,
                resample_thresh=self.qinfer_resampler_threshold,
                resampler=qi.LiuWestResampler(
                    a=self.qinfer_resampler_a
                ),
                debug_resampling=False
            )
            self.qinfer_updater._normalization_record = self.NormalizationRecord
            self.qinfer_updater._log_total_likelihood = self.log_total_likelihood
            time_taken = time.time() - time_s
            self.log_print(
                [
                    "Time to reconstruct updater: {}".format(
                        time_taken
                    )
                ]
            )

        else:
            time_s = time.time()
            self.qinfer_updater = pickle.loads(
                learned_model_info['updater']
            )
            time_taken = time.time() - time_s
            self.log_print(
                [
                    "Time to unpickle updater: {}".format(
                        time_taken
                    )
                ]
            )
        self.log_print(
            [
                "Prior mean:", self.qinfer_updater.est_mean()
            ]
        )
        del qmd_info, learned_model_info

    def log_print(self, to_print_list):
        identifier = str(str(time_seconds()) +
                         "[QML:Bayes " + str(self.model_id) +
                         "; QMD " + str(self.qmla_id) + "]"
                         )
        if not isinstance(to_print_list, list):
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file)
