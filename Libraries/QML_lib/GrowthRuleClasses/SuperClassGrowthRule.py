import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues

class GrowthRuleSuper():
    # superclass for growth generation rules
    
    def __init__(
        self, 
        growth_generation_rule,
        **kwargs
    ): 
        self.growth_generation_rule = growth_generation_rule
        
        self.true_operator = 'xTi'
        self.qhl_models = ['xTi', 'yTi', 'zTi'] 
        self.initial_models = ['xTi', 'yTi', 'zTi'] 
        self.max_num_parameter_estimate = 2
        self.max_spawn_depth = 10
        self.max_num_qubits = 5
        self.tree_completed_initially = False
        self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.measurement_type = 'full_access'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z' # e.g. transverse axis
        
        self.max_num_models_by_shape = {
                1 : 0,
                2 : 1,
                'other' : 0
        }
       
    def generate_models(
        self, 
        **kwargs
    ):
        # default is to just return given model list and set spawn stage to complete
        
        spawn_stage = kwargs['spawn_stage']
        spawn_stage.append('Complete')
        return model_list
        
        
    def expectation_value(
        self,
        **kwargs
    ):
        exp_val = ExpectationValues.expectation_value(**kwargs)
        return exp_val

    def check_tree_completed(
        self,
        spawn_step, 
        **kwargs
    ):
        print("[default growth class] checking tree completed")
        if spawn_step == self.max_spawn_depth:
            print("[default growth class] MAX SPAWN DEPTH REACHED FOR RULE ", self.growth_generation_rule)
            return True 
        else:
            return False
        return True

    def branch_index(
        self,
        **kwargs
    ):
        return True
    
    def latex_name(
        self,
        **kwargs
    ):  
        # name: string to be formatted for latex
        return str('${}$'.format(name))
        
    def heuristic(
        self,
        **kwargs
    ):
        return True
        
    def probe_generator(
        self,
        **kwargs
    ):
        return True
        


class default_growth(GrowthRuleSuper):
    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        print("[Growth Rules] Default growth rule")
        super().__init__(
    		growth_generation_rule = growth_generation_rule,
        	**kwargs
    	)
