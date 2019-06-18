import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase

from SuperClassGrowthRule import GrowthRuleSuper

class hopping(
    GrowthRuleSuper
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

        self.true_operator = 'h_1h2_d2'
        self.initial_models = [
            'h_1h2_d2'
        ] 
        self.qhl_models = [
            'h_1h2_d2',    
        ]
        self.max_num_parameter_estimate = 8
        self.max_spawn_depth = 20
        self.max_num_qubits = 4

        self.max_num_models_by_shape = {
            2 : 1,
            3 : 3, 
            4 : 6, 
            5 : 10,
            6 : 15, 
            7 : 21, 
            8 : 29, 
            9 : 38,
            'other' : 0
        }


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


    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):
        # from UserFunctions import initial_models, max_num_qubits_info, fixed_axes_by_generator, transverse_axis_by_generator
            
        # growth_generator = kwargs['generator']
        # model_list = kwargs['model_list']
        spawn_stage = kwargs['spawn_stage']
        max_dimension = self.max_num_qubits
        branch_champs_by_qubit_num = kwargs['branch_champs_by_qubit_num']
        new_models = []
        mod = model_list[0]

        if spawn_stage[-1] == None:
            spawn_stage.append((2,'c'))

        if type(spawn_stage[-1]) == tuple:
            dim = spawn_stage[-1][0]
            num_branches_this_dim = spawn_stage[-1][1] # either a number or 'c'
            
            if dim == max_dimension:
                at_max_dim = True
            else:
                at_max_dim = False
            if spawn_stage[-1][1] == 'c':
                this_dimension_complete = True # champ this dim found
            else:
                this_dimension_complete = False
            if spawn_stage[-1][1] == dim - 1 :
                this_dimension_exhausted = True # all possible models this dim considered
            else:
                this_dimension_exhausted = False

            # now complete logic depending on which spawn_stage passed
            if (
                this_dimension_exhausted == False
                and 
                this_dimension_complete == False
            ):
                # add one parameter to given mod, 
                # return a list
                # spawn_stage -> (N, num_branches+1)
                present_terms = DataBase.get_constituent_names_from_name(mod)
                possible_new_terms_this_dimension = possible_hopping_terms_new_site(dim)
                nonpresent_possible_terms = list(
                    set(possible_new_terms_this_dimension)
                    - set(present_terms)
                )
                new_models = append_model_with_new_terms(mod, nonpresent_possible_terms)
                spawn_stage.append( (dim, num_branches_this_dim+1) )
            elif (
                this_dimension_complete == False
                and
                this_dimension_exhausted == True
            ):
                
                new_models = branch_champs_by_qubit_num[dim]
                spawn_stage.append((dim, 'c'))
                
                # return branch champs corresponding to this num qubits
                # to form a ghost branch
                # spawn_stage -> (N, 'c')
            
            elif (
                this_dimension_complete == True
                and
                at_max_dim == False
            ):
                increased_dim_model = increase_dimension_maintain_distinct_interactions(
                    mod
                )
                # new_terms = possible_hopping_terms_new_site(dim+1)
                # new_models = append_model_with_new_terms(
                #     increased_dim_model, 
                #     new_terms
                # )
                new_models = [increased_dim_model]
                spawn_stage.append((dim+1, 0))
                # spawn_stage.append((dim+1, 1))
                
                # this num qubits complete, 
                # move up to higher dimension
                # return mod, since that was the winner of previous branch
                # and therefore winner for this num_qubits
                # spawn_stage -> (N+1, 1)
            elif (
                at_max_dim == True
                and
                this_dimension_complete == True
            ):
                spawn_stage.append('Complete')
    #             new_models = branch champions which won their ghost as well
                
                all_branch_champions = []
                for q in range(2, max_dimension+1):
                    all_branch_champions.extend(
                        branch_champs_by_qubit_num[q]
                    )
                unique_branch_champs = list(set(all_branch_champions))
                won_multiple_branches = []
                for m in unique_branch_champs:
                    if all_branch_champions.count(m) >= 2:
                        won_multiple_branches.append(m)

                min_dim = min(
                    branch_champs_by_qubit_num.keys()
                )
                won_multiple_branches.extend(
                    branch_champs_by_qubit_num[min_dim]
                )
                new_models = won_multiple_branches
        
        return new_models


    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        individual_terms = DataBase.get_constituent_names_from_name(name)
        latex_term = ''

        hopping_terms = []
        interaction_energy = False
        for constituent in individual_terms:
            components = constituent.split('_')
            for term in components: 
                if term != 'h': #ie entire term not just 'h'
                    if 'h' in term: # ie a hopping term eg 1_1h2_d3, hopping sites 1-2, total num sites 3
                        split_term = term.split('h')
                        hopping_terms.append(split_term)

                    elif 'e' in term:
                        interaction_energy = True
                    elif 'd' in term:
                        dim = int(term.replace('d', ''))

        hopping_latex = 'H_{'
        for site_pair in hopping_terms:
            hopping_latex += str(
                '({},{})'.format(
                    str(site_pair[0]), 
                    str(site_pair[1])
                )
            )

        hopping_latex += '}'

        if hopping_latex != 'H_{}':
            latex_term += hopping_latex
            latex_term += str(
                "^{ \otimes"
                 + str(dim) + "}"
            )

        if interaction_energy is True:
            latex_term += str(
                '\sigma_{z}^{\otimes'
                +str(dim)
                +'}'
            )

        latex_term = str('$' + latex_term + '$')
        return latex_term


## Supporting functions

def possible_hopping_terms_new_site(site_id):
    new_terms = []
    dim = site_id
    for i in range(1, site_id):
        new_term = str(
            'h_' + 
            str(i) + 'h' + str(site_id)
            + '_d'+str(dim)
        )
        new_terms.append(new_term)
        
    return new_terms

def append_model_with_new_terms(mod, new_terms):
    dimension = DataBase.get_num_qubits(mod)
    p_str = 'P'*dimension
    
    new_mods = []
    
    for term in new_terms:
        new = str(
            mod + p_str
            + term
        )
        new_mods.append(new)
    return new_mods

def increase_dimension_maintain_distinct_interactions(
    mod, 
    dim_inc=1
):
    # here distinct means EVERY pair of sites corresponds to a separate parameter.
    dec = deconstruct_hopping_term(mod)
    new_dim = dec['dim'] + dim_inc
    sites = dec['sites']
    p_str = 'P'*new_dim
    
    overall_model = ''
    for site in sites:
        new_model_dec = {
            'dim' : new_dim,
            'sites' : [site]
        }
        new_term = generate_hopping_term(new_model_dec)
        if sites.index(site) != 0:
            overall_model += p_str
        overall_model += str(new_term)
        
    return overall_model


def deconstruct_hopping_term(hopping_string):
    dim = DataBase.get_num_qubits(hopping_string)
    individual_terms = DataBase.get_constituent_names_from_name(hopping_string)
    deconstructed = {
        'sites' : [], 
        'dim' : dim,
        'interaction_energy' : False
    }
    
    for term in individual_terms:
        split_term = term.split('_')
        sites = []
        for i in split_term:
            if i[0] == 'd':
                # dim = int(i[1])
                dim = dim = int(i.replace('d', ''))
            elif 'e' in i:
                deconstructed['interaction_energy'] = True
            elif i != 'h':
                sites = i.split('h')
                sites = [int(a) for a in sites]
                deconstructed['sites'].append(sites)
    return deconstructed

def generate_hopping_term(
    deconstructed, 
    include_interaction_energy=False
):
    sites_list = deconstructed['sites']
    dim = deconstructed['dim']
    try:
        interaction_energy = deconstructed['interaction_energy']
    except:
        interaction_energy = include_interaction_energy
    
    if type(sites_list[0]) != list:
        sites_list = [sites_list]
    p_str = ''
    for i in range(dim):
        p_str += 'P'
    overall_term = ''
    first_term = True
    
    hopping_string = 'h'

    for sites in sites_list:
        hopping_string += str(
            '_' + 
            str(sites[0]) + 
            'h' + 
            str(sites[1])
        )
    
    hopping_string += str( '_d' + str(dim)) 
    overall_term += hopping_string

    if interaction_energy == True:
        interaction_term = str(
            'h_e_d' + str(dim) 
        )
        overall_term += str(
            p_str +
            interaction_term
        )
#         overall_term += str(
#             p_str 
#             + 
#             interaction_energy_pauli_term(dim)
#         )
    return overall_term
