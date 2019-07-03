import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues
import ProbeGeneration
# import Heuristics

class GrowthRuleSuper():
    # superclass for growth generation rules
    
    def __init__(
        self, 
        growth_generation_rule,
        **kwargs
    ): 
        # print("[GrowthRuleSuper] init. kwargs", kwargs)

        import Heuristics
        self.growth_generation_rule = growth_generation_rule
        if 'use_experimental_data' in kwargs:
            self.use_experimental_data = kwargs['use_experimental_data']
        else:
            self.use_experimental_data = False

        # by changing the function object these point to, 
        # determine how probes are generated and expectation values are computed
        # these can be directly overwritten within class definition
        # by writing self.probe_generator and self.expectation_value methods         
        self.probe_generation_function = ProbeGeneration.separable_probe_dict
        self.plot_probe_generation_function = ProbeGeneration.plus_probes_dict
        self.expectation_value_function = ExpectationValues.expectation_value
        self.heuristic_function = Heuristics.multiPGH
        self.highest_num_qubits = 1
        self.spawn_stage = [None]

        # Parameters specific to the growth rule
        self.true_operator = 'xTi'
        self.qhl_models = ['xTi', 'yTi', 'zTi'] 
        self.initial_models = ['xTi', 'yTi', 'zTi'] 
        self.max_num_parameter_estimate = 2
        self.max_spawn_depth = 10
        self.max_num_qubits = 5
        self.max_num_probe_qubits = 11
        self.tree_completed_initially = False
        self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.measurement_type = 'full_access'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z' # e.g. transverse axis
        self.num_processes_to_parallelise_over = 5
        
        self.max_num_models_by_shape = {
            1 : 0,
            2 : 1,
            'other' : 0
        }

        self.gaussian_prior_means_and_widths = {
        }
        self.num_probes = 40
        self.min_param = 0
        self.max_param = 1

        # TODO set true params for simulation here
        self.true_params = {
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
        return ModelNames.branch_is_num_params(
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
        
    def probe_generator(
        self,
        **kwargs
    ):
        return self.probe_generation_function(
            max_num_qubits = self.max_num_probe_qubits,
            num_probes = self.num_probes, 
            **kwargs
        )

    def plot_probe_generator(
        self, 
        **kwargs
    ):
        return self.plot_probe_generation_function(
            max_num_qubits = self.max_num_probe_qubits,
            num_probes = 1,
            **kwargs
        )

