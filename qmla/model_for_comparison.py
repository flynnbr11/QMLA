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
# import qmla.experimental_data_processing as expdt
import qmla.experimental_data_processing
import qmla.database_framework
import qmla.analysis

pickle.HIGHEST_PROTOCOL = 4


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
        model_id,
        qid,
        qmla_core_info_database=None,
        host_name='localhost',
        port_number=6379,
        log_file='QMD_log.log',
        learned_model_info=None,
    ):
        self.log_file = log_file
        self.qmla_id = qid
        self.model_id = model_id

        if qmla_core_info_database is None:
            redis_databases = rds.get_redis_databases_by_qmla_id(
                host_name,
                port_number,
                qid
            )
            qmla_core_info_database = redis_databases['qmla_core_info_database']
            qmla_core_info_dict = pickle.loads(qmla_core_info_database.get('qmla_settings'))
            self.probes_system = pickle.loads(qmla_core_info_database['ProbeDict'])
            self.probes_simulator = pickle.loads(qmla_core_info_database['SimProbeDict'])
        else: 
            qmla_core_info_dict = qmla_core_info_database.get('qmla_settings')
            self.probes_system = qmla_core_info_database['ProbeDict']
            self.probes_simulator = qmla_core_info_database['SimProbeDict']


        self.num_particles = qmla_core_info_dict['num_particles']
        self.probe_number = qmla_core_info_dict['num_probes']
        self.qinfer_resampler_threshold = qmla_core_info_dict['resampler_thresh']
        self.qinfer_resampler_a = qmla_core_info_dict['resampler_a']
        self.qinfer_PGH_heuristic_factor = qmla_core_info_dict['pgh_prefactor']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.true_model_name = qmla_core_info_dict['true_name']
        self.use_experimental_data = qmla_core_info_dict['use_experimental_data']
        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        self.experimental_measurement_times = qmla_core_info_dict['experimental_measurement_times']
        self.results_directory = qmla_core_info_dict['results_directory']

        # Get model specific data
        if learned_model_info is None:
            # get the learned info dictionary from the redis
            # database corresponding to this model id
            try:
                redis_databases = rds.get_redis_databases_by_qmla_id(
                    host_name,
                    port_number,
                    qid
                )
                learned_models_info_db = redis_databases['learned_models_info_db']
            except: 
                print("Unable to retrieve redis database.")
                raise 

            model_id_str = str(float(model_id))
            try:
                learned_model_info = pickle.loads(
                    learned_models_info_db.get(model_id_str),
                    encoding='latin1'
                )
            except BaseException:
                learned_model_info = pickle.loads(
                    learned_models_info_db.get(model_id_str)
                )
            except: 
                print("Learned model's info not provided to ModelInstanceForComparison")

        self.model_name = learned_model_info['name']
        self.log_print(
            [
                "Name:", self.model_name
            ]
        )
        op = qmla.database_framework.Operator(self.model_name)
        self.model_terms_matrices = op.constituents_operators
        self.times_learned_over = learned_model_info['times']
        self.final_learned_params = learned_model_info['final_params']
        self.model_terms_parameters_final = np.array(self.final_learned_params)
        self.growth_rule_of_this_model = learned_model_info['growth_generator']
        self.growth_class = get_growth_rule.get_growth_generator_class(
            growth_generation_rule=self.growth_rule_of_this_model,
            use_experimental_data=self.use_experimental_data,
            log_file=self.log_file
        )
        self.model_prior = learned_model_info['final_prior']
        self.posterior_marginal = learned_model_info['posterior_marginal']
        self.initial_prior = learned_model_info['initial_prior']
        self.model_normalization_record = learned_model_info['normalization_record']
        self.log_total_likelihood = learned_model_info['log_total_likelihood']
        self.learned_parameters_qhl = learned_model_info['learned_parameters']
        self.final_sigmas_qhl = learned_model_info['final_sigmas']
        self.covariance_mtx_final = learned_model_info['final_cov_mat']
        log_identifier = str("Bayes " + str(self.model_id))

        self.qinfer_model = self.growth_class.qinfer_model(
            model_name=self.model_name,
            modelparams=self.model_terms_parameters_final,
            oplist=self.model_terms_matrices,
            true_oplist=self.true_model_constituent_operators,
            truename=self.true_model_name,
            trueparams=self.true_model_params,
            num_probes=self.probe_number,
            probe_dict=self.probes_system,
            sim_probe_dict=self.probes_simulator,
            growth_generation_rule=self.growth_rule_of_this_model,
            use_experimental_data=self.use_experimental_data,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            log_file=self.log_file,
            # measurement_type=self.measurement_class,
            # log_identifier=log_identifier
        )

        self.reconstruct_updater = True # optionally just load it
        time_s = time.time()
        if self.reconstruct_updater:
            posterior_distribution = qi.MultivariateNormalDistribution(
                learned_model_info['est_mean'],
                self.covariance_mtx_final
            )

            num_particles_for_bf = max(
                5, 
                int(self.growth_class.fraction_particles_for_bf * self.num_particles)
            )
            self.log_print(
                [
                    "For Bayes factor calculation, using {} particles".format(num_particles_for_bf)
                ]
            )

            self.qinfer_updater = qi.SMCUpdater(
                model=self.qinfer_model,
                # n_particles=self.num_particles,
                n_particles=num_particles_for_bf,
                prior=posterior_distribution,
                zero_weight_policy='ignore', #TODO testing ignore - does it cause failures?
                resample_thresh=self.qinfer_resampler_threshold,
                resampler=qi.LiuWestResampler(
                    a=self.qinfer_resampler_a
                ),
                # debug_resampling=False
            )
            self.qinfer_updater._normalization_record = self.model_normalization_record
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
        del qmla_core_info_dict, learned_model_info

    def log_print(
        self,
        to_print_list
    ):
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier='ModelForComparison {}'.format(self.model_id)
        )