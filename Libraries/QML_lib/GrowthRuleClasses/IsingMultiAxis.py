import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration

# from SuperClassGrowthRule import GrowthRuleSuper
import SuperClassGrowthRule
import IsingChain 



class isingChainMultiAxis(
	# SuperClassGrowthRule.GrowthRuleSuper
    IsingChain.isingChain
):

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        # self.true_operator = 'xTz'
        # self.true_operator = 'pauliSet_xJz_1J2_d2'
        # self.true_operator = 'pauliSet_x_1_d2PPpauliSet_xJx_1J2_d2PPpauliSet_yJz_2J3_d2'
        # self.true_operator = 'pauliSet_x_1_d2PPpauliSet_xJx_1J2_d2PPpauliSet_yJz_2J3_d2'
        # print("[isingChainMultiAxis] true op set" )
        self.true_operator = 'pauliSet_zJz_1J2_d3PPPpauliSet_zJz_2J3_d3PPPtransverse_x_d3'
        # self.true_operator = 'pauliSet_xJx_1J2_d3PPPpauliSet_xJx_2J3_d3'
        self.initial_models = [
            'pauliSet_zJz_1J2_d2'
            # 'pauliSet_x_1_d1',
            # 'pauliSet_y_1_d1', 
            # 'pauliSet_z_1_d1', 
        ] 
        self.qhl_models = [
            'pauliSet_zJz_1J2_d2',
        ]
        # self.available_axes = ['x','y','z']
        self.available_axes = ['z']
        self.max_num_parameter_estimate = 2
        self.max_spawn_depth = 4
        self.max_num_qubits = 3
        self.transverse_axis = 'x'

        self.max_num_models_by_shape = {
            'other' : 9, 
        }
        self.plot_probe_generation_function = ProbeGeneration.zero_state_probes
        self.min_param = 0
        self.max_param = 1
        self.consider_transverse_axis = False

        self.true_params = {
            'pauliSet_xJz_1J2_d2' : 3.5,
            'pauliSet_xJz_1J2_d2' : 6.8
        }


    def generate_models(
        self, 
        model_list,
        **kwargs
    ):
        new_models=[]
        # spawn_stage = kwargs['spawn_stage']
        spawn_stage = self.spawn_stage
        current_champs = kwargs['current_champs']
        if (
            spawn_stage[-1] == None
            or 
            spawn_stage[-1][1] == 'non_transverse__incomplete'
        ):
            for mod in model_list:
                num_sites = DataBase.get_num_qubits(mod)
                new_dimension = num_sites + 1
                p_str = 'P'*num_sites
                p_str_new_mod = 'P'*new_dimension
                potential_terms = self.possible_new_interactions(
                    num_sites = num_sites
                )
                separate_terms = mod.split(p_str)
                initial_model_elements = []
                for term in separate_terms:
                    components = term.split('_')
                    components.remove('pauliSet')
                    for l in components:
                        if l[0] == 'd':
                            dim = int(l.replace('d', ''))
                            new_dim = dim+1
                            new_d_str = str('d'+str(new_dim))
                            components.remove(l)
                            components.append(new_d_str)

                    initial_model_element = '_'.join(components)
                    initial_model_element = 'pauliSet_' + initial_model_element

                    initial_model_elements.append(initial_model_element)
                initial_model = p_str_new_mod.join(initial_model_elements)   # increased dimension              

                for term in potential_terms:
                    new_mod = str(
                        initial_model + 
                        p_str_new_mod + 
                        term
                    )

                    new_models.append(new_mod)

            this_spawn_stage = (new_dimension, 'non_transverse__incomplete')
            spawn_stage.append(this_spawn_stage)

            if self.highest_num_qubits == self.max_num_qubits:
                spawn_stage.append('non_transverse__complete')
                if self.consider_transverse_axis == False:
                    spawn_stage.append('Complete')

        elif spawn_stage[-1] == 'non_transverse__complete':
            for mod in current_champs:
                new = add_transverse_name(
                    mod, 
                    transverse_axis = self.transverse_axis
                )

                new_models.append(new)
            print("isingChainMultiAxis transverse mods:", new_models)
            spawn_stage.append('Complete') 

        return new_models


    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        core_operators = list(sorted(DataBase.core_operator_dict.keys()))
        num_sites = DataBase.get_num_qubits(name)
        p_str = 'P'*num_sites
        separate_terms = name.split(p_str)

        latex_terms = []
        term_type_markers = ['pauliSet', 'transverse']
        for term in separate_terms:
            components = term.split('_')
            if 'pauliSet' in components:
                components.remove('pauliSet')

                for l in components:
                    if l[0] == 'd':
                        dim = int(l.replace('d', ''))
                    elif l[0] in core_operators:
                        operators = l.split('J')
                    else:
                        sites = l.split('J')

                latex_str = '\sigma'

                latex_str += '^{'
                for s in sites:
                    latex_str += str( '{},'.format(s) )

                latex_str = latex_str[0:-1]
                latex_str += '}'

                latex_str += '_{'
                for o in operators:
                    latex_str += str( '{},'.format(o) )
                latex_str = latex_str[0:-1] # remove final comma
                latex_str += '}'

            elif 'transverse' in components:
                components.remove('transverse')
                for l in components:
                    if l[0] == 'd':
                        dim = int(l.replace('d', ''))
                    else:
                        transverse_axis = str(l)

                latex_str = '\sigma'

                latex_str += '^{\otimes'
                latex_str += str(dim)
                latex_str += '}'

                latex_str += '_{'
                latex_str += str(transverse_axis)
                latex_str += '}'

            latex_terms.append(latex_str)

        latex_terms = sorted(latex_terms)
        full_latex_term = ''.join(latex_terms)
        full_latex_term = str( '$' +  full_latex_term +'$' )


        return full_latex_term



    def check_tree_completed(
        self,
        spawn_step, 
        **kwargs
    ):
        if self.spawn_stage[-1] == 'Complete':
            return True 
        else:
            return False
        return True


    ### Supporting functions 

    def possible_new_interactions(
        self, 
        num_sites, 
        available_axes=None
    ):
        if available_axes is None:
            available_axes = self.available_axes
        terms = []
        for a in available_axes:
            for b in available_axes:
                sites = [num_sites, num_sites + 1]
                ops = [a, b]

                sites = [str(i) for i in sites]
                ops = [str(i) for i in ops]
                site_str = 'J'.join(sites)
                op_str = 'J'.join(ops)
                d_str = str('d' + str(num_sites+1))
                
                new_mod = str(
                    'pauliSet_{}_{}_{}'.format(
                        op_str, 
                        site_str,
                        d_str
                    ) 
                )
                
                terms.append(new_mod)
        return terms        

def add_transverse_name(
    term,
    transverse_axis
):
    num_qubits = DataBase.get_num_qubits(term)
    p_str = 'P' * num_qubits
    transverse_name = "transverse_{}_d{}".format(transverse_axis, num_qubits)
    new_term = str(
        term 
        + p_str
        + transverse_name
    )
    return new_term
    
   

