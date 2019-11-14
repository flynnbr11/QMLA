import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues
import ProbeGeneration
import Heuristics
import Distributions
# import Heuristics

class growth_rule_super_class():
    # superclass for growth generation rules
    
    def __init__(
        self, 
        growth_generation_rule,
        **kwargs
    ): 
        # print("[GrowthRuleSuper] init. kwargs", kwargs)

        
        self.growth_generation_rule = growth_generation_rule
        if 'use_experimental_data' in kwargs:
            self.use_experimental_data = kwargs['use_experimental_data']
        else:
            self.use_experimental_data = False

        if 'log_file' in kwargs:
            self.log_file = kwargs['log_file']
        else:
            self.log_file = '.default_qmd_log.log'

        # by changing the function object these point to, 
        # determine how probes are generated and expectation values are computed
        # these can be directly overwritten within class definition
        # by writing self.probe_generator and self.expectation_value methods         
        self.probe_generation_function = ProbeGeneration.separable_probe_dict
        self.simulator_probe_generation_function = self.probe_generation_function # unless specifically different set of probes required
        self.shared_probes = True # i.e. system and simulator get same probes for learning
        self.plot_probe_generation_function = ProbeGeneration.plus_probes_dict
        self.expectation_value_function = ExpectationValues.expectation_value
        self.heuristic_function = Heuristics.multiPGH
        self.prior_distribution_generator = Distributions.gaussian_prior
        self.highest_num_qubits = 1
        self.spawn_stage = [None]

        # Parameters specific to the growth rule
        self.true_operator = 'xTi'
        #qhl_models is the list of models used in a fixed QHL case, where we are not running any QML
        self.qhl_models = ['xTi', 'yTi', 'zTi'] 
        #initial_models is the first branch of models in QML 
        self.initial_models = ['xTi', 'yTi', 'zTi'] 
        self.max_num_parameter_estimate = 2
        # max_spawn_depth is the maximum number of spawns/branches in a run
        self.max_spawn_depth = 10
        self.max_num_qubits = 5
        self.max_num_probe_qubits = 5 # TODO remove dependency on this -- it is not needed
        self.max_time_to_consider = 15 # arbitrary time units
        # If you want to do just Bayes facotr calculation on a deterministic initial set you set tree_completed_initially to True
        self.tree_completed_initially = False
        self.check_champion_reducibility = False
        self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.measurement_type = 'full_access' #deprecated
        self.fixed_axis_generator = False # if you have a transverse axis and you want to generate on that axis than set it to True
        self.fixed_axis = 'z' # e.g. transverse axis
        self.num_processes_to_parallelise_over = 6
        self.learned_param_limit_for_negligibility = 0.05
        self.reduce_champ_bayes_factor_threshold = 1e2
        self.model_branches = {}

        self.max_num_models_by_shape = {
            1 : 0,
            2 : 1,
            'other' : 0
        }

        self.gaussian_prior_means_and_widths = {
            # term : (mean, sigma)
        }
        self.num_probes = 40
        # self._num_probes = 40
        self.min_param = 0
        self.max_param = 1
        self.prior_random_mean = False
        self.true_params = {
            # term : true_param
        }

    def overwrite_growth_class_methods(
        self, 
        **kwargs
    ):
        # print("[GrowthRuleSuper] overwrite_growth_class_methods. kwargs", kwargs)
        kw = list(kwargs.keys())

        attributes = [
            'probe_generator'
        ]

        for att in attributes:

            if att in kw and kwargs[att] is not None:
                print("Resetting {} to {}".format(att, kwargs[att]))
                self.__setattr__(att, kwargs[att])

    def true_operator_latex(self):
        return self.latex_name(self.true_operator)

    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):
        # default is to just return given model list and set spawn stage to complete        
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
        import ModelNames
        # branch_is_num_params
        return ModelNames.branch_computed_from_qubit_and_param_count(
            latex_mapping_file = latex_mapping_file,
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
        return self.heuristic_function(
            **kwargs
        )

    def get_prior(
        self, 
        model_name,
        **kwargs
    ):
        self.prior = self.prior_distribution_generator(
            model_name = model_name, 
            prior_specific_terms = self.gaussian_prior_means_and_widths,
            param_minimum = self.min_param, 
            param_maximum = self.max_param, 
            random_mean = self.prior_random_mean,
            **kwargs
        )
        return self.prior
        
    def generate_probes(
        # system probes
        self,
        **kwargs
    ):
        self.log_print(
            [
                "System Generate Probes called"
            ]
        )
        self.system_probes = self.probe_generation_function(
            max_num_qubits = self.max_num_probe_qubits,
            num_probes = self.num_probes, 
            **kwargs
        )
        if self.shared_probes == True:
            self.simulator_probes = self.system_probes
        else:
            self.simulator_probes = self.simulator_probe_generation_function(
                max_num_qubits = self.max_num_probe_qubits,
                num_probes = self.num_probes, 
                **kwargs
            )

    def probe_generator(
        # system probes
        self,
        **kwargs
    ):
        self.system_probes = self.probe_generation_function(
            max_num_qubits = self.max_num_probe_qubits,
            num_probes = self.num_probes, 
            **kwargs
        )
        if self.shared_probes == True:
            self.simulator_probes = self.system_probes
        else:
            self.simulator_probes = self.simulator_probe_generation_function(
                max_num_qubits = self.max_num_probe_qubits,
                num_probes = self.num_probes, 
                **kwargs
            )
        return self.system_probes

    def simulator_probe_generator(
        # system probes
        self,
        shared_probes = None,
        **kwargs
    ):
        self.log_print(
            [
                "Simulator Generate Probes called"
            ]
        )
        if shared_probes == None:
            shared_probes = self.shared_probes

        if shared_probes == True:
            self.simulator_probes = self.system_probes
        else:
            self.simulator_probes = self.simulator_probe_generation_function(
                max_num_qubits = self.max_num_probe_qubits,
                num_probes = self.num_probes, 
                **kwargs
            )
        return self.simulator_probes

    def plot_probe_generator(
        self, 
        **kwargs
    ):
        return self.plot_probe_generation_function(
            max_num_qubits = self.max_num_probe_qubits,
            num_probes = 1,
            **kwargs
        )

    @property
    def true_operator_terms(self):
        true_terms = DataBase.get_constituent_names_from_name(
            self.true_operator
        )

        latex_true_terms = [
            self.latex_name(term) for term in true_terms
        ]

        self.true_op_terms = set(sorted(latex_true_terms))

        return self.true_op_terms

    def log_print(
        self, 
        to_print_list
    ):
        identifier = "[Growth: {}]".format(self.growth_generation_rule)
        if type(to_print_list)!=list:
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




    

