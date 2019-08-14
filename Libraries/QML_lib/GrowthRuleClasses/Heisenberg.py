import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase

import SuperClassGrowthRule

class heisenberg_XYZ(
    SuperClassGrowthRule.growth_rule_super_class
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

        self.true_operator = 'Heis_ix_d3PPPHeis_iy_d3PPPHeis_iz_d3PPPHeis_tz_d3'
        self.initial_models = [
            'Heis_ix_d2PPHeis_iy_d2PPHeis_iz_d2',
            'Heis_ix_d2PPHeis_iy_d2PPHeis_iz_d2PPHeis_tz_d2'
        ] 
        self.qhl_models = [
            'Heis_ix_d3PPPHeis_iy_d3PPPHeis_iz_d3PPPHeis_tz_d3',    
            'Heis_ix_d3PPPHeis_iy_d3PPPHeis_iz_d3',    
        ]
        self.max_num_parameter_estimate = 2
        self.max_spawn_depth = 5
        self.max_num_qubits = 5
        self.fixed_axis_generator = False
        self.fixed_axis = 'x' # e.g. transverse axis
        self.transverse_axis = 'z'

        self.max_num_models_by_shape = {
            'other' : 2, 
        }

    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):
        spawn_stage = kwargs['spawn_stage']
        max_num_qubits = self.max_num_qubits
    #     interaction_axis = fixed_axes_by_generator[growth_generator]
        transverse_axis = self.transverse_axis

        new_models = []
        if spawn_stage[-1] == None:
            for q in range(2, max_num_qubits+1): 
                # include qubit number = 1 so compared against all others fairly 
                new_models.append(
                    get_heisenberg_xyz_name(
                        num_qubits = q, 
                        include_transverse = False
                    )
                )
            spawn_stage.append('non-transverse complete')
        elif spawn_stage[-1] == 'non-transverse complete':
            for q in range(2, max_num_qubits+1): 
                new_models.append(
                    get_heisenberg_xyz_name(
                        num_qubits = q, 
                        include_transverse = True, 
                        transverse_axis = transverse_axis
                    )
                )
            spawn_stage.append('Complete')
        return new_models

    def check_tree_completed(
        self, 
        current_num_qubits,
        **kwargs
    ):
        if (
            current_num_qubits 
            == 
            self.max_num_qubits
        ):
            return True 
        else:
            return False

    def name_branch_map(
        self,
        latex_mapping_file, 
        **kwargs
    ):
        import ModelNames
        name_map = ModelNames.branch_is_num_dims(
            latex_mapping_file = latex_mapping_file,
            **kwargs
        )
        return name_map


    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        individual_terms = DataBase.get_constituent_names_from_name(name)
        chain_axis = transverse_axis = None
        chain_axes = []
        
        for term in individual_terms:
            components = term.split('_')
            try:
                components.remove('Heis')
            except:
                print(
                    "Can not remove 'Heis'. Components:", 
                    components
                )
            for c in components:
                if c[0] == 'd':
                    dim = int(c.replace('d', ''))
                elif c[0] == 'i':
                    chain_axis = str(c.replace('i', ''))
                    chain_axes.append(chain_axis)
                    include_chain_component = True
                elif c[0] == 't' : 
                    include_transverse_component = True
                    transverse_axis = str(c.replace('t', ''))
        chain_axes = list(sorted(set(chain_axes)))
        # print("[latex term, XYZ] Chain axes:", chain_axes)
        latex_term = '$('
        # if chain_axis is not None:
        chain_term = ''
        for chain_axis in chain_axes:
            chain_term += str(
                '\sigma_{'
                + chain_axis
                + ','
                + chain_axis
                +'}'
            )
        latex_term += chain_term
            
        if transverse_axis is not None:
            transverse_term = str(
                '\sigma_{'
                + transverse_axis
                +'}'
            )
            latex_term += transverse_term
        
        
        latex_term += str( 
            ')^{\otimes'
            + str(dim)
            + '}$'
        )
            
        return latex_term


## Supporting functions

def get_heisenberg_xyz_name(
    num_qubits, 
    transverse_axis='z',
    include_transverse=False
):
    model_identifier = 'Heis_'
    model_name = ''
    dimension_term = str( '_d' + str(num_qubits))

    interaction_terms = []
    for axis in ['x', 'y', 'z']:
        axis_interaction_term = str(
            model_identifier +
            'i' +
            axis +
            dimension_term
        )
        interaction_terms.append(axis_interaction_term)
    p_str = 'P'*num_qubits
    print(interaction_terms)
    
    full_model = ''
    for term in interaction_terms:
        if interaction_terms.index(term)>0:
            full_model += str( p_str )
        full_model += str(   term )
    
    if include_transverse == True:
        transverse_term = str(model_identifier +  't' + transverse_axis + dimension_term)
        full_model += str( p_str + transverse_term)
    return full_model
