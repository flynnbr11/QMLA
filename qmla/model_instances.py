import numpy as np
import scipy as sp
import os
import time
import copy
import qinfer as qi
import random 

import redis
import pickle

import qmla.redis_settings as rds
# import qmla.qinfer_model_interface as qml_qi
import qmla.memory_tests
import qmla.logging
import qmla.get_growth_rule as get_growth_rule
import qmla.shared_functionality.prior_distributions
import qmla.database_framework
import qmla.analysis

pickle.HIGHEST_PROTOCOL = 4

from qmla.model_for_comparison import ModelInstanceForComparison
from qmla.model_for_storage import ModelInstanceForStorage

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


class ModelInstanceForLearning():
    """
    Class to learn individual model. Model name is given when initialised.
    A host_name and port_number are given to initialise_model_for_learning.
    The qmla_core_info_dict dict from Redis is pulled and pickled to find
    the true model and other QMD parameters needed.
    A GenSimModel is set which details the SMCUpdater
    used to update the posterior distribution.
    update_model calls the updater in a loop of n_experiments.
    The final parameter estimates are set as the mean of the
    posterior distribution after n_experiments wherein n_particles
    are sampled per experiment (set in qmla_core_info_dict).

    """

    def __init__(
        self,
        model_id,
        model_name,
        qid,
        log_file,
        growth_generator,
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
        self.log_print(
            [
                "QHL for model {} (id:{})".format(
                    model_name,
                    model_id, 
                )
            ]

        )
        self.initialise_model_for_learning(
            growth_generator=growth_generator,
            model_name = self.model_name, 
            qmla_core_info_database=qmla_core_info_database,
        )

    def log_print(
        self,
        to_print_list
    ):
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier='ModelForLearning {}'.format(self.model_id)
        )

    def initialise_model_for_learning(
        self,
        model_name,
        growth_generator,
        qmla_core_info_database,
        checkloss=True,
        debug_directory=None,
        **kwargs
    ):
        redis_databases = rds.get_redis_databases_by_qmla_id(
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
        
        # self.use_experimental_data = qmla_core_info_dict['use_experimental_data']
        self.num_particles = qmla_core_info_dict['num_particles']
        self.num_experiments = qmla_core_info_dict['num_experiments']
        self.growth_rule_of_this_model = growth_generator

        try:
            self.growth_class = get_growth_rule.get_growth_generator_class(
                growth_generation_rule=self.growth_rule_of_this_model,
                # use_experimental_data=self.use_experimental_data,
                log_file=self.log_file
            )
        except BaseException:
            print("Failed to load growth class {}".format(
                    self.growth_rule_of_this_model
                )
            )
            raise


        # Get starting configuration for this model to learn from
        op = qmla.database_framework.Operator(name=model_name)
        self.model_terms_names = op.constituents_names
        self.model_name_latex = self.growth_class.latex_name(
            name=self.model_name
        )
        self.model_terms_matrices = np.asarray(op.constituents_operators)
        self.model_dimension = qmla.database_framework.get_num_qubits(self.model_name)

        model_priors = qmla_core_info_dict['model_priors']
        if (
            model_priors is not None
            and
            qmla.database_framework.alph(model_name) in list(model_priors.keys())
        ):
            prior_specific_terms = model_priors[model_name]
        else:
            prior_specific_terms = qmla_core_info_dict['prior_specific_terms']

        model_terms_parameters = []
        for term in self.model_terms_names:
            try:
                initial_prior_centre = prior_specific_terms[term][0]
                model_terms_parameters.append(initial_prior_centre)
            except BaseException:
                # if prior not defined, start from 0 for all other params
                initial_prior_centre = 0
                model_terms_parameters.append(initial_prior_centre)

        model_terms_parameters = [model_terms_parameters]
        self.model_terms_parameters = np.asarray([model_terms_parameters[0]])

        if qmla_core_info_dict['reallocate_resources'] == True:
            base_resources = qmla_core_info_dict['base_resources']
            base_num_qubits = base_resources['num_qubits']
            base_num_terms = base_resources['num_terms']
            this_model_num_qubits = qmla.database_framework.get_num_qubits(
                self.model_name)
            this_model_num_terms = len(
                qmla.database_framework.get_constituent_names_from_name(
                    self.model_name)
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

        # get info from qmla core info dictionary held on redis database
        self.probe_number = qmla_core_info_dict['num_probes']
        self.qinfer_resampler_threshold = qmla_core_info_dict['resampler_thresh']
        self.qinfer_resampler_a = qmla_core_info_dict['resampler_a']
        self.qinfer_PGH_heuristic_factor = qmla_core_info_dict['pgh_prefactor']
        self.qinfer_PGH_heuristic_exponent = qmla_core_info_dict['pgh_exponent']
        self.qinfer_PGH_heuristic_increase_time = qmla_core_info_dict['increase_pgh_time']
        self.store_particle_locations_and_weights = qmla_core_info_dict['store_particles_weights']
        self.results_directory = qmla_core_info_dict['results_directory']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        # self.true_hamiltonian = qmla_core_info_dict['true_hamiltonian']
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.true_model_name = qmla_core_info_dict['true_name']
        self.sigma_threshold = qmla_core_info_dict['sigma_threshold']
        self.times_to_plot = qmla_core_info_dict['plot_times']
        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        self.experimental_measurement_times = qmla_core_info_dict['experimental_measurement_times']
        self.true_params_path = qmla_core_info_dict['true_params_pickle_file']
        

        individual_terms_in_name = qmla.database_framework.get_constituent_names_from_name(
            self.model_name
        )

        for i in range(len(individual_terms_in_name)):
            term = individual_terms_in_name[i]
            term_mtx = qmla.database_framework.compute(term)
            if np.all(term_mtx == self.model_terms_matrices[i]) is False:
                # TODO make this raise an exception instead
                print("[ModelInstanceForLearning] UNEQUAL LIST / TERM MATRICES.")
                print("==> INSPECT ORDER OF PRIORS.")
                self.log_print(
                    [
                        "Term", term,
                        "\ncalculated mtx:", term_mtx,
                        "\nSimOpList:", self.model_terms_matrices[i]
                    ]
                )
            elif term != self.model_terms_names[i]:
                self.log_print(
                    [
                        "term {} != SimOpsNames[i] {}".format(
                            term, self.model_terms_names[i]
                        )
                    ]
                )

        num_params = len(self.model_terms_matrices)
        log_identifier = str("QML " + str(self.model_id))
        self.model_prior = self.growth_class.get_prior(
            model_name=self.model_name,
            log_file=self.log_file,
            log_identifier=log_identifier,
        )
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

        latex_terms = []
        for term in individual_terms_in_name:
            lt = self.growth_class.latex_name(
                name=term
            )
            latex_terms.append(lt)

        plot_all_priors = True
        if plot_all_priors == True:
            qmla.shared_functionality.prior_distributions.plot_prior(
                model_name=self.model_name_latex,
                model_name_individual_terms=latex_terms,
                prior=self.model_prior,
                plot_file=prior_file,
            )

        self.qinfer_model = self.growth_class.qinfer_model(
            model_name=self.model_name,
            modelparams=self.model_terms_parameters,
            oplist=self.model_terms_matrices,
            true_oplist=self.true_model_constituent_operators,
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
            zero_weight_policy='ignore', #TODO testing ignore - does it cause failures?
            resample_thresh=self.qinfer_resampler_threshold,
            resampler=qi.LiuWestResampler(a=self.qinfer_resampler_a),
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

        self.model_heuristic = self.growth_class.heuristic(
            updater=self.qinfer_updater,
            oplist=self.model_terms_matrices,
            inv_field=self.inversion_field,
            increase_time=self.qinfer_PGH_heuristic_increase_time,
            pgh_exponent=self.qinfer_PGH_heuristic_exponent,
            time_list=self.times_to_plot,
            num_experiments=self.num_experiments,
            max_time_to_enforce=self.growth_class.max_time_to_consider,
            log_file = self.log_file,
        )
        self.model_heuristic_class = self.model_heuristic.__class__.__name__
        self._initialise_tracking_infrastructure()

    def _initialise_tracking_infrastructure(
        self
    ):
        self.quadratic_losses = []
        self.track_total_log_likelihood = np.array([])
        self.track_experimental_times = np.array([])  # only for debugging
        self.particles = np.array([])
        self.weights = np.array([])
        self.epochs_after_resampling = []
        self.final_learned_params = np.empty(
            [len(self.model_terms_matrices), 2])
        self.covariances = np.empty(self.num_experiments)
        self.track_mean_params = [self.qinfer_updater.est_mean()]
        self.track_covariance_matrices = []
        self.track_param_dist_widths = []
        self.track_posterior_dist = []
        self.track_prior_means = []
        self.track_prior_std_dev = []
        self.track_experimental_times = np.empty(
            self.num_experiments)  # only for debugging

        self.particles = np.empty([self.num_particles,
                                   len(self.model_terms_parameters[0]), self.num_experiments]
                                  )
        self.weights = np.empty([self.num_particles, self.num_experiments])
        self.true_model_params_dict = {}
        self.learned_parameters_qhl = {}
        self.final_sigmas_qhl = {}
        self.time_update = 0
        self.time_volume = 0
        self.time_simulate_experiment = 0


    def update_model(
        self,
        checkloss=False  # TODO is this needed?
    ):
        full_update_time_init = time.time()
        true_params_names = qmla.database_framework.get_constituent_names_from_name(
            self.true_model_name
        )

        self.true_model_params_dict = self.growth_class.true_params_dict
        all_params_for_q_loss = list(
            set(list(self.true_model_params_dict.keys())).union(self.model_terms_names)
        )
        param_indices = {}
        for op_name in self.model_terms_names:
            param_indices[op_name] = self.model_terms_names.index(op_name)

        print_frequency = max(
            int(self.num_experiments / 5),
            5
        )
        for istep in range(self.num_experiments):
            if (istep % print_frequency == 0):
                # print so we can see how far along algorithm is.
                self.log_print(
                    [
                        "Epoch", istep
                    ]
                )
            if istep == 0:
                param_estimates = self.qinfer_updater.est_mean()
            else:
                param_estimates = self.track_mean_params[-1]
            _time_init = time.time()
            self.new_experiment = self.model_heuristic(
                num_params=len(self.model_terms_names),
                epoch_id=istep,
                current_params=param_estimates
            )
            self.time_heuristic += time.time() - _time_init
            # TODO prefactor, if used, should be inside specific GR heuristic
            self.new_experiment[0][0] = self.new_experiment[0][0] * \
                self.qinfer_PGH_heuristic_factor

            if istep == 0:
                self.log_print(['Initial time selected = ',
                                str(self.new_experiment[0][0])]
                               )
            self.track_experimental_times[istep] = self.new_experiment[0][0]

            _time_init = time.time()
            self.datum_from_experiment = self.qinfer_model.simulate_experiment(
                self.model_terms_parameters,
                self.new_experiment,
                repeat=1
            ) 
            self.time_simulate_experiment += time.time() - _time_init

            # Call updater to update distribution based on datum
            try:
                sum_wts = sum([a**2 for a in self.qinfer_updater.particle_weights])
                n_ess = 1/sum_wts # corresponds to n_ess in QInfer, for record
                if n_ess < self.num_particles / 2 or n_ess < 10:                        
                    self.log_print([
                        # "Particles:", self.qinfer_updater.particle_locations, 
                        # "\nWeights:", self.qinfer_updater.particle_weights,
                        "Small n_ess - expect to resample soon. Sum weights = {}; n_ess = {}".format(sum_wts, (1/sum_wts))                     
                    ])
                _time_start = time.time()
                self.qinfer_updater.update(
                    self.datum_from_experiment,
                    self.new_experiment
                )
                self.time_update += time.time() - _time_start
                    
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
                print("\n\n[Model class] EXITING; Inspect log\n\n")
                raise NameError("Qinfer update failure")
                sys.exit()
            except: 
                self.log_print(
                    [
                        "Failed to update model {} at update step {}".format(
                            self.model_id, 
                            istep
                        )
                    ]
                )
                sys.exit()

            if self.qinfer_updater.just_resampled:
                self.log_print(["Just resampled"])
                self.epochs_after_resampling.append(istep)

            _time_init = time.time()
            self.volume_by_epoch = np.append(
                self.volume_by_epoch,
                np.linalg.det(
                    sp.linalg.sqrtm(
                        self.qinfer_updater.est_covariance_mtx()
                    )  # TODO seems unnecessary to do this every epoch - every 10th would be enough for plot
                )
            )
            self.time_volume += time.time() - _time_init

            self.track_mean_params.append(self.qinfer_updater.est_mean())
            self.track_param_dist_widths.append(
                np.sqrt(
                    np.diag(self.qinfer_updater.est_covariance_mtx())
                )
            )
            # TODO this doesn't seem necessary to store
            self.track_covariance_matrices.append(
                self.qinfer_updater.est_covariance_mtx())
            prior_sample = self.qinfer_updater.sample(int(5))

            these_means = []
            these_std = []
            for i in range(len(self.model_terms_matrices)):
                these_means.append(np.mean(prior_sample[:, i]))
                these_std.append(np.std(prior_sample[:, i]))

            self.track_posterior_dist.append(prior_sample)
            self.track_prior_means.append(these_means)
            self.track_prior_std_dev.append(these_std)
            self.covariances[istep] = np.linalg.norm(
                self.qinfer_updater.est_covariance_mtx()
            )
            # self.particles[:, :,
                        #    istep] = self.qinfer_updater.particle_locations
            # self.weights[:, istep] = self.qinfer_updater.particle_weights

            if checkloss:
                # TODO not currently recording quadratic loss
                quadratic_loss = 0
                for param in all_params_for_q_loss:
                    if param in self.model_terms_names:
                        learned_param = self.qinfer_updater.est_mean()[param_indices[param]]
                    else:
                        learned_param = 0

                    if param in list(self.true_model_params_dict.keys()):
                        true_param = self.true_model_params_dict[param]
                    else:
                        true_param = 0
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
                            self.qinfer_updater.est_mean(),
                            np.sqrt(np.diag(updater.est_covariance_mtx()))
                        ]
                        print('Final Parameters mean and stdev:' +
                              str(self.final_learned_params[iterator])
                              )
                    self.model_log_total_likelihood = (
                        self.qinfer_updater.log_total_likelihood
                    )
                    self.covariances = (
                        np.resize(
                            self.covariances, (1, istep)))[0]
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
                self.log_print([
                    "{} Resample epochs: {}".format(len(self.epochs_after_resampling), self.epochs_after_resampling)
                ])
                self.model_log_total_likelihood = self.qinfer_updater.log_total_likelihood
                self.log_print([
                    "Subroutine times:", 
                    "Total:", np.round(time.time() - full_update_time_init, 2),
                    "\tUpdates:", np.round(self.time_update,2), 
                    "\t Simulate experiment:", np.round(self.time_simulate_experiment, 2), 
                    "\t Volume:", np.round(self.time_volume, 2),
                    "\t Heuristic:", np.round(self.time_heuristic, 2),
                ])

                cov_mat = self.qinfer_updater.est_covariance_mtx()
                for iterator in range(len(self.final_learned_params)):
                    self.final_learned_params[iterator] = [
                        self.qinfer_updater.est_mean()[iterator],
                        np.sqrt(cov_mat[iterator][iterator])
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
        learned_info['updater'] = pickle.dumps(self.qinfer_updater, protocol=4)
        learned_info['final_prior'] = self.qinfer_updater.prior
        learned_info['initial_prior'] = self.initial_prior
        learned_info['model_terms_names'] = self.model_terms_names
        learned_info['final_cov_mat'] = self.qinfer_updater.est_covariance_mtx()
        learned_info['est_mean'] = self.qinfer_updater.est_mean()
        learned_info['posterior_marginal'] = all_post_margs
        learned_info['initial_params'] = self.model_terms_parameters
        learned_info['volume_list'] = self.volume_by_epoch
        learned_info['track_mean_params'] = self.track_mean_params
        learned_info['track_cov_matrices'] = self.track_covariance_matrices
        learned_info['track_param_sigmas'] = self.track_param_dist_widths
        learned_info['track_posterior'] = self.track_posterior_dist
        # repeat of track param sigmas?
        learned_info['track_prior_means'] = self.track_prior_means
        learned_info['track_prior_std_devs'] = self.track_prior_std_dev
        learned_info['epochs_after_resampling'] = self.epochs_after_resampling
        learned_info['quadratic_losses'] = self.quadratic_losses
        learned_info['learned_parameters'] = self.learned_parameters_qhl
        learned_info['final_sigmas'] = self.final_sigmas_qhl
        learned_info['cov_matrix'] = self.qinfer_updater.est_covariance_mtx()
        learned_info['num_particles'] = self.num_particles
        learned_info['num_experiments'] = self.num_experiments
        learned_info['growth_generator'] = self.growth_rule_of_this_model
        learned_info['heuristic'] = self.model_heuristic_class
        try:
            learned_info['evaluation_log_likelihood'] = self.evaluation_log_likelihood
            learned_info['evaluation_normalization_record'] = self.evaluation_normalization_record
            learned_info['evaluation_median_likelihood'] = self.evaluation_median_likelihood
        except:
            learned_info['evaluation_log_likelihood'] = None
            learned_info['evaluation_normalization_record'] = None
            learned_info['evaluation_median_likelihood'] = None

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

    def compute_likelihood_after_parameter_learning(
        self,
        times = [1, 2]
    ):
        from qmla.remote_bayes_factor import get_exp
        posterior_distribution = qi.MultivariateNormalDistribution(
            self.qinfer_updater.est_mean(),
            self.qinfer_updater.est_covariance_mtx(),
        )

        learned_params = list(self.final_learned_params[:, 0])
        self.learned_hamiltonian = np.tensordot(
            learned_params,
            self.model_terms_matrices,
            axes=1
        )
        self.log_print(
            [
                "Learned parameters:", learned_params, 
            ]
        )

        true_params_dict = pickle.load(
            open(
                self.true_params_path, 
                'rb'
            )
        )
        
        evaluation_times = true_params_dict['evaluation_times']
        evaluation_probe_dict = true_params_dict['evaluation_probes']

        evaluation_qinfer_model = self.growth_class.qinfer_model(
            model_name=self.model_name,
            modelparams=self.model_terms_parameters,
            oplist=self.model_terms_matrices,
            true_oplist=self.true_model_constituent_operators,
            truename=self.true_model_name,
            trueparams=self.true_model_params,
            num_probes=self.probe_number,
            probe_dict=evaluation_probe_dict,
            sim_probe_dict=evaluation_probe_dict,
            growth_generation_rule=self.growth_rule_of_this_model,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            log_file=self.log_file,
        )

        evaluation_updater = qi.SMCUpdater(
            model=self.qinfer_model,
            n_particles=min(25, self.num_particles),
            # n_particles=self.num_particles,
            prior=posterior_distribution,
            # resample more aggressively once learned, since closer to true values
            resample_thresh=max(self.qinfer_resampler_threshold, 0.6), 
            resampler=qi.LiuWestResampler(
                a=self.qinfer_resampler_a
            ),
            # debug_resampling=False
        )

        evaluation_updater._log_total_likelihood = 0.0
        evaluation_updater._normalization_record = []

        for t in evaluation_times:
            exp = format_experiment(
                self.qinfer_model, 
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

        log_likelihood = evaluation_updater.log_total_likelihood
        self.evaluation_normalization_record = evaluation_updater.normalization_record
        if np.isnan(evaluation_updater.log_total_likelihood):
            self.evaluation_log_likelihood = None 
            self.evaluation_median_likelihood = None
            self.log_print([
                "Evaluation ll is nan"
            ])
        else:
            self.evaluation_log_likelihood = round_nearest(
                evaluation_updater.log_total_likelihood, 
                0.05
            )
            self.evaluation_median_likelihood = np.round(
                np.median(evaluation_updater.normalization_record),
                2
            )
        

def round_nearest(x,a):
    return round(round(x/a)*a ,2)



def format_experiment(model, final_learned_params, time):
    # gen = model.qinfer_model
    exp = np.empty(
        len(time),
        dtype=model.expparams_dtype
    )
    exp['t'] = time

    try:
        for i in range(1, len(model.expparams_dtype)):
            col_name = 'w_' + str(i)
            exp[col_name] = final_learned_params[i - 1, 0]
    except BaseException:
        print("failed to get exp. \nFinal params:", final_learned_params)

    return exp
