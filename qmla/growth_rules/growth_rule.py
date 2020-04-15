from __future__ import absolute_import
import sys
import os
import pickle
import numpy as np
import itertools

import qmla.shared_functionality.prior_distributions
import qmla.shared_functionality.experiment_design_heuristics
import qmla.shared_functionality.probe_set_generation as probe_set_generation
import qmla.shared_functionality.expectation_values
import qmla.database_framework as database_framework
import qmla.growth_rules.rating_system
import qmla.shared_functionality.qinfer_model_interface
from qmla.growth_rules.growth_rule_decorator import GrowthRuleDecorator

__all__ = [
    'GrowthRule'
]

# @GrowthRuleDecorator
class GrowthRule():
    # superclass for growth generation rules
    def __init__(
        self,
        growth_generation_rule,
        # configuration=None, 
        **kwargs
    ):
        # print("GrowthRule __init__. kwargs", kwargs)
        self.growth_generation_rule = growth_generation_rule
        # if 'use_experimental_data' in kwargs:
        #     self.use_experimental_data = kwargs['use_experimental_data']
        # else:
        #     self.use_experimental_data = False

        if 'log_file' in kwargs:
            self.log_file = kwargs['log_file']
        else:
            self.log_file = '.default_qmla_log.log'
        
        if 'true_params_path' in kwargs: 
            self.true_params_path = kwargs['true_params_path']
        else: 
            self.true_params_path = None
        
        if 'plot_probes_path' in kwargs: 
            self.plot_probes_path = kwargs['plot_probes_path']
        else: 
            self.plot_probes_path = None

        self.assign_parameters()

    def assign_parameters(self):
        self.use_experimental_data = False # TODO included for legacy; to be removed everywhere its called
        # by changing the function object these point to,
        # determine how probes are generated and expectation values are computed
        # these can be directly overwritten within class definition
        # by writing self.probe_generator and self.expectation_value methods
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict
        # unless specifically different set of probes required
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = True  # i.e. system and simulator get same probes for learning
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        self.expectation_value_function = qmla.shared_functionality.expectation_values.default_expectation_value
        self.probe_noise_level = 1e-5
        self.fraction_particles_for_bf = 1.0 # testing whether reduced num particles for BF can work 
        self.ratings_class = qmla.growth_rules.rating_system.ELORating(
            initial_rating=1500,
            k_const=30
        ) # for use when ranking/rating models
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MultiParticleGuessHeuristic
        self.qinfer_model_class = qmla.shared_functionality.qinfer_model_interface.QInferModelQMLA
        self.prior_distribution_generator = qmla.shared_functionality.prior_distributions.gaussian_prior
        self.highest_num_qubits = 1
        self.spawn_stage = [None]
        self.prune_step = 0 
        self.model_branches = {} 
        self.champion_determined = False
        self.growth_rule_specific_data_to_store = {}

        # Parameters specific to the growth rule
        self.true_model = 'xTi'
        # qhl_models is the list of models used in a fixed QHL case, where we
        # are not running any QML
        self.qhl_models = ['xTi', 'yTi', 'zTi']
        # initial_models is the first branch of models in QML
        self.initial_models = ['xTi', 'yTi', 'zTi']
        self.max_num_parameter_estimate = 2
        # max_spawn_depth is the maximum number of spawns/branches in a run
        self.max_spawn_depth = 10
        self.max_num_qubits = 5
        self.max_num_probe_qubits = 6
        # self.max_num_probe_qubits = 5  # TODO remove dependency on this -- it is not needed
        self.max_time_to_consider = 15  # arbitrary time units
        self.num_top_models_to_build_on = 1
        # If you want to do just Bayes facotr calculation on a deterministic
        # initial set you set tree_completed_initially to True
        self.tree_completed_initially = False
        self.prune_complete = False
        self.prune_completed_initially = False
        self.check_champion_reducibility = True
        self.learned_param_limit_for_negligibility = 0.05
        self.reduce_champ_bayes_factor_threshold = 1e1

        self.experimental_dataset = 'NVB_rescale_dataset.p'
        # self.measurements_by_time = self.get_measurements_by_time()
        # self.measurement_type = 'full_access'  # deprecated
        # if you have a transverse axis and you want to generate on that axis
        # than set it to True
        # self.fixed_axis_generator = False
        # self.fixed_axis = 'z'  # e.g. transverse axis
        self.num_processes_to_parallelise_over = 6
        

        self.max_num_models_by_shape = {
            1: 0,
            2: 1,
            'other': 0
        }

        self.gaussian_prior_means_and_widths = {
            # term : (mean, sigma)
        }
        self.num_probes = 40
        # self._num_probes = 40
        self.min_param = 0
        self.max_param = 1
        self.prior_random_mean = False
        self.fixed_true_terms = False        
        self.get_true_parameters()
        self.true_model_terms_params = {
            # term : true_param
        }

    def get_true_parameters(
        self,
    ):        
        # get true data from pickled file
        try:
            true_config = pickle.load(
                open(
                    self.true_params_path, 
                    'rb'
                )
            )
            self.true_params_list = true_config['params_list']
            self.true_params_dict = true_config['params_dict']
        except:
            # self.log_print(
            #     [
            #         "Could not unpickle {}".format(self.true_params_path)
            #     ]
            # )
            self.true_params_list = []
            self.true_params_dict = {}

        true_ham = None
        for k in list(self.true_params_dict.keys()):
            param = self.true_params_dict[k]
            mtx = database_framework.compute(k)
            if true_ham is not None:
                true_ham += param * mtx
            else:
                true_ham = param * mtx
        self.true_hamiltonian = true_ham




    def store_growth_rule_configuration(
        self, 
        path_to_pickle_config = None,
        **kwargs
    ):
        dict_for_storage = self.__dict__
        if path_to_pickle_config is not None: 
            pickle.dump(
                dict_for_storage,
                open(
                    path_to_pickle_config, 'wb'
                )                
            )
        return dict_for_storage

    def overwrite_growth_class_methods(
        self,
        **kwargs
    ):
        # print("[GrowthRule] overwrite_growth_class_methods. kwargs", kwargs)
        kw = list(kwargs.keys())

        attributes = [
            'probe_generator'
        ]

        for att in attributes:

            if att in kw and kwargs[att] is not None:
                print("Resetting {} to {}".format(att, kwargs[att]))
                self.__setattr__(att, kwargs[att])

    def true_model_latex(self):
        return self.latex_name(self.true_model)

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        # default is to just return given model list and set spawn stage to
        # complete
        return model_list

    def check_tree_completed(
        self,
        spawn_step,
        **kwargs
    ):
        if spawn_step == self.max_spawn_depth:
            return True
        else:
            return False
        return True

    def name_branch_map(
        self,
        latex_mapping_file,
        **kwargs
    ):
        
        r"""
        branch_is_num_params
        
        """
        import qmla.shared_functionality.branch_mapping        
        return qmla.shared_functionality.branch_mapping.branch_computed_from_qubit_and_param_count(
            latex_mapping_file=latex_mapping_file,
            **kwargs
        )

    def latex_name(
        self,
        name,
        **kwargs
    ):
        # name: string to be formatted for latex
        return str('${}$'.format(name))

    # General wrappers

    def expectation_value(
        self,
        **kwargs
    ):
        return self.expectation_value_function(
            **kwargs
        )

    def heuristic(
        self,
        **kwargs
    ):
        return self.model_heuristic_function(
            **kwargs
        )
    
    def qinfer_model(
        self, 
        **kwargs
    ):
        return self.qinfer_model_class(
            **kwargs
        )

    def get_prior(
        self,
        model_name,
        **kwargs
    ):
        self.prior = self.prior_distribution_generator(
            model_name=model_name,
            prior_specific_terms=self.gaussian_prior_means_and_widths,
            param_minimum=self.min_param,
            param_maximum=self.max_param,
            random_mean=self.prior_random_mean,
            **kwargs
        )
        return self.prior

    def generate_probes(
        self,
        probe_maximum_number_qubits=None, 
        store_probes=True,
        **kwargs
    ):
        if probe_maximum_number_qubits is None: 
            probe_maximum_number_qubits = self.max_num_probe_qubits
        self.log_print(
            [
                "System Generate Probes called",
                "probe max num qubits:", probe_maximum_number_qubits
            ]
        )
        
        new_probes = self.probe_generation_function(
            max_num_qubits=probe_maximum_number_qubits,
            num_probes=self.num_probes,
            **kwargs
        )
        if store_probes:
            self.probes_system = new_probes
            if self.shared_probes == True:
                self.probes_simulator = self.probes_system
            else:
                self.probes_simulator = self.simulator_probe_generation_function(
                    max_num_qubits=probe_maximum_number_qubits,
                    num_probes=self.num_probes,
                    **kwargs
                )
        else:
            return new_probes

    def plot_probe_generator(
        self,
        probe_maximum_number_qubits=None, 
        **kwargs
    ):
        if probe_maximum_number_qubits is None: 
            probe_maximum_number_qubits = self.max_num_probe_qubits

        plot_probe_dict =  self.plot_probe_generation_function(
            max_num_qubits=probe_maximum_number_qubits,
            num_probes=1,
            **kwargs
        )
        for k in list(plot_probe_dict.keys()):
            # replace tuple like key returned, with just dimension.
            plot_probe_dict[k[1]] = plot_probe_dict.pop(k)
        self.plot_probe_dict = plot_probe_dict
        return plot_probe_dict

    @property
    def true_model_terms(self):
        true_terms = database_framework.get_constituent_names_from_name(
            self.true_model
        )

        latex_true_terms = [
            self.latex_name(term) for term in true_terms
        ]

        self.true_op_terms = set(sorted(latex_true_terms))

        return self.true_op_terms

    def get_measurements_by_time(
        self
    ):
        try:
            true_info = pickle.load(
                open(
                    self.true_params_path,
                    'rb'
                )
            )
        except:
            print("Failed to load true params from path", self.true_params_path)
            raise

        self.true_params_dict = true_info['params_dict']
        true_ham = None
        for k in list(self.true_params_dict.keys()):
            param = self.true_params_dict[k]
            mtx = database_framework.compute(k)
            if true_ham is not None:
                true_ham += param * mtx
            else:
                true_ham = param * mtx
        self.true_hamiltonian = true_ham

        true_ham_dim = database_framework.get_num_qubits(self.true_model)
        plot_probes = pickle.load(
            open(
                self.plot_probes_path, 
                'rb'
            )
        )
        probe = plot_probes[true_ham_dim]

        num_datapoints_to_plot = 300
        plot_lower_time = 0
        plot_upper_time = self.max_time_to_consider
        raw_times = list(np.linspace(
            0,
            self.max_time_to_consider,
            num_datapoints_to_plot + 1)
        )
        plot_times = [np.round(a, 2) for a in raw_times]
        plot_times = sorted(plot_times)
        
        self.measurements = {
            t : self.expectation_value(
                ham = self.true_hamiltonian, 
                t = t, 
                state = probe
            )
            for t in plot_times
        }
        self.log_print(
            [
                "Storing measurements:\n", self.measurements
            ]
        )
        return self.measurements

    def tree_pruning(
        self,
        previous_prune_branch,
    ):
        self.prune_step += 1
        prune_step = self.prune_step
        pruning_models = []
        pruning_sets = []
        self.log_print([
            "Pruning within {}".format(self.growth_generation_rule),
            "Branches:", self.tree.branches
        ])
        if prune_step == 1:
            child_parent_pairs = []
            for branch in self.tree.branches.values():
                pruning_models.append(branch.champion_name)
                self.log_print([
                    "Getting child/parents for branch", branch.branch_id
                ])
                try:
                    champ = branch.champion_id
                    parent_champ = branch.parent_branch.champion_id
                    pair = (champ, parent_champ)
                    if champ != parent_champ:
                        pruning_sets.append(pair)
                except:
                    self.log_print([
                        "Branch has no parent:", branch.branch_id
                    ])
                    pass
    
        elif prune_step == 2:
            pruned_branch = self.tree.branches[previous_prune_branch]
            # check bayes factor compairsons on those from previous prune branch, 
            # which corresponds to parent/child collapse
            prune_collapse_threshold = 1e2 # TODO set as GR attribute
            prev_branch_models = []
            for l in list(zip(*pruned_branch.pairs_to_compare)):
                prev_branch_models.extend(list(l))
            prev_branch_models = list(set(prev_branch_models))

            models_to_prune = []
            for pair in pruned_branch.pairs_to_compare:
                id_1 = pair[0]
                id_2 = pair[1]
                mod_1 = pruned_branch.model_instances[id_1]
                try:
                    bf_1_v_2 = mod_1.model_bayes_factors[ float(id_2) ][-1]
                except:
                    self.log_print([
                        "couldnt find bf {}/{}. mod_{} BF:".format( 
                            id_1, 
                            id_2, 
                            id_1,
                            mod_1.model_bayes_factors
                        )
                    ])
                if bf_1_v_2 > prune_collapse_threshold:
                    models_to_prune.append(id_2)
                elif bf_1_v_2 < float(1 / prune_collapse_threshold):
                    models_to_prune(id_1)

                models_to_keep = list(
                    set(prev_branch_models)
                    - set(models_to_prune)
                )
                pruning_models = [
                    pruned_branch.models_by_id[m]
                    for m in models_to_keep
                ]
                pruning_sets = list(itertools.combinations(
                    models_to_keep, 
                    2
                ))
            self.prune_complete = True

        self.log_print([
            "Prune step {}. pruning models: {} \n pruning sets: {}".format(
                prune_step, 
                pruning_models, 
                pruning_sets
            )
        ])
        if len(pruning_models) == 1:
            self.prune_complete = True
        self.log_print(["Returning from pruning fnc"])
        return list(set(pruning_models)), pruning_sets



    def growth_rule_finalise(self):
        # do whatever is needed to wrap up growth rule
        # e.g. store data required for analysis
        self.growth_rule_specific_data_to_store = {}

    def growth_rule_specific_plots(
        self,
        save_directory, 
        **kwargs
    ):
        self.log_print(
            ['No growth rule plots specified.']
        )

    def log_print(
        self,
        to_print_list
    ):
        identifier = "[Growth: {}]".format(self.growth_generation_rule)
        if type(to_print_list) != list:
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(
                identifier,
                str(to_print),
                file=write_log_file,
                flush=True
            )
