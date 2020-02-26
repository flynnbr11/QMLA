from __future__ import absolute_import
import sys
import os
import pickle

import qmla.prior_distributions as Distributions
import qmla.experiment_design_heuristics as experiment_design_heuristics
import qmla.probe_set_generation as probe_set_generation
import qmla.expectation_values as expectation_values
import qmla.database_framework as database_framework
from qmla.growth_rules.growth_rule_decorator import GrowthRuleDecorator

__all__ = [
    'GrowthRuleSuper'
]

# @GrowthRuleDecorator
class GrowthRuleSuper():
    # superclass for growth generation rules
    def __init__(
        self,
        growth_generation_rule,
        # configuration=None, 
        **kwargs
    ):
        # print("GrowthRuleSuper __init__. kwargs", kwargs)
        self.growth_generation_rule = growth_generation_rule
        if 'use_experimental_data' in kwargs:
            self.use_experimental_data = kwargs['use_experimental_data']
        else:
            self.use_experimental_data = False

        if 'log_file' in kwargs:
            self.log_file = kwargs['log_file']
        else:
            self.log_file = '.default_qmla_log.log'

        self.assign_parameters()

    def assign_parameters(self):
        # by changing the function object these point to,
        # determine how probes are generated and expectation values are computed
        # these can be directly overwritten within class definition
        # by writing self.probe_generator and self.expectation_value methods
        self.probe_generation_function = probe_set_generation.separable_probe_dict
        # unless specifically different set of probes required
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = True  # i.e. system and simulator get same probes for learning
        self.plot_probe_generation_function = probe_set_generation.plus_probes_dict
        self.expectation_value_function = expectation_values.default_expectation_value
        self.probe_noise_level = 1e-5
        self.model_heuristic_function = experiment_design_heuristics.MultiParticleGuessHeuristic
        self.prior_distribution_generator = Distributions.gaussian_prior
        self.highest_num_qubits = 1
        self.spawn_stage = [None]
        self.model_branches = {} 

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
        self.check_champion_reducibility = False
        self.learned_param_limit_for_negligibility = 0.05
        self.reduce_champ_bayes_factor_threshold = 1e2

        self.experimental_dataset = 'NVB_rescale_dataset.p'
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
        self.true_model_terms_params = {
            # term : true_param
        }

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
        # print("[GrowthRulesuper] overwrite_growth_class_methods. kwargs", kwargs)
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
        spawn_stage = kwargs['spawn_stage']
        spawn_stage.append('Complete')
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
        import qmla.model_naming
        # branch_is_num_params
        return model_naming.branch_computed_from_qubit_and_param_count(
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
        **kwargs
    ):
        self.log_print(
            [
                "System Generate Probes called"
            ]
        )
        if probe_maximum_number_qubits is None: 
            probe_maximum_number_qubits = self.max_num_probe_qubits
        self.probes_system = self.probe_generation_function(
            max_num_qubits=probe_maximum_number_qubits,
            num_probes=self.num_probes,
            **kwargs
        )
        if self.shared_probes == True:
            self.probes_simulator = self.probes_system
        else:
            self.probes_simulator = self.simulator_probe_generation_function(
                max_num_qubits=probe_maximum_number_qubits,
                num_probes=self.num_probes,
                **kwargs
            )

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
