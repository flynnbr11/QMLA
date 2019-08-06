import numpy as np
import itertools
import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelNames
import Heuristics

import SuperClassGrowthRule
import NV_centre_large_spin_bath
import NV_grow_by_fitness
import Spin_probabilistic


class PauliPairwiseNearestNeighbourProbabilistic(
    Spin_probabilistic.SpinProbabilistic
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
        self.heuristic_function = Heuristics.one_over_sigma_then_linspace
        # self.true_operator = 'pauliSet_xJx_1J2_d2PPpauliSet_yJx_1J2_d2'
        self.true_operator = 'pauliSet_xJx_1J2_d3PPPpauliSet_yJy_2J3_d3'
        self.true_operator = DataBase.alph(self.true_operator)

        self.qhl_models = ['pauliSet_xJx_1J2_d2']
        self.base_terms = [
            'x', 
            'y', 
            'z'
        ]
        self.num_top_models_to_build_on = 'all' # at each generation Badassness parameter
        self.model_generation_strictness = 0 #1 #-1 
        self.fitness_win_ratio_exponent = 0.25
        self.initial_models = pairwise_pauli_like_like_terms(
        # self.initial_models = pairwise_pauli_terms(
            base_terms = self.base_terms, 
            new_site = 2
        )
        self.generation_DAG = 1
        self.num_sites = 2
        self.max_num_generations = 5

        for i in range(
            self.generation_DAG, 
            self.max_num_generations+1
        ):
            possible_terms = pairwise_pauli_like_like_terms(
            # possible_terms = pairwise_pauli_terms(

                base_terms = self.base_terms, 
                new_site = i + 1
            )
            self.available_mods_by_generation[i] = possible_terms
            # self.max_num_sub_generations_per_generation[i] = len(possible_terms)
            self.max_num_sub_generations_per_generation[i] = 2
            self.num_sub_generations_per_generation[i] = 0
            self.generation_champs[i] = {}
        self.max_num_sub_generations_per_generation[1] = 1

    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):
        print("In generation fnc probabilistic spin")
        fitness = kwargs['fitness_parameters']
        model_points = kwargs['branch_model_points']
        branch_models = list(model_points.keys())
        # keep track of generation_DAG
        ranked_model_list = sorted(
            model_points, 
            key=model_points.get, 
            reverse=True
        )
        if self.num_top_models_to_build_on == 'all':
            models_to_build_on = ranked_model_list
        else:
            models_to_build_on = ranked_model_list[:self.num_top_models_to_build_on]
        self.models_to_build_on[self.generation_DAG] = models_to_build_on
        new_models = []
        # self.sub_generation_idx += 1 
        self.generation_champs[self.generation_DAG][self.sub_generation_idx] = models_to_build_on

        # if self.spawn_stage[-1] == None:


        
        # if self.spawn_stage[-1] == 'end_generation':
        #     print("[spin prob] End of gen {}; making new gen".format(self.generation_DAG))
        #     print("gen champs:", self.generation_champs[self.generation_DAG])

        #     # get new mod ids using current generation DAG before increasing it. 
        #     new_mod_ids = list( # now a list of lists 
        #         self.generation_champs[self.generation_DAG].values()
        #     )
        #     new_mod_ids = sorted([i for sublist in new_mod_ids for i in sublist])            

        #     self.spawn_stage.append('make_new_gen')
        #     self.generation_DAG += 1
        #     self.models_accepted[self.generation_DAG] = []
        #     self.models_rejected[self.generation_DAG] = []
        #     self.sub_generation_idx = 0


        #     if self.generation_DAG == self.max_num_generations:
        #         self.spawn_stage.append('Complete')
        #         current_champs = kwargs['current_champs']
        #         print("QMD complete; comparing all branch champs so far:", current_champs)
        #         new_models = current_champs
        #     else:
        #         new_models = [ 
        #             kwargs['model_names_ids'][mod_id] for mod_id in new_mod_ids
        #         ]

        if  self.spawn_stage[-1] == 'make_new_gen':
            self.num_sites += 1
        
        if self.spawn_stage[-1] in [None, 'make_new_gen']:

            for mod_id in self.models_to_build_on[self.generation_DAG]:
                mod_name = kwargs['model_names_ids'][mod_id]
                if self.spawn_stage[-1] == 'make_new_gen':
                    # in this growth rule making a new generation corresponds 
                    # to increasing the dimension of the system. 
                    print("Making new generation. increasing dimension; starting model:", mod_name)
                    print("generation:", self.generation_DAG, "has available mods:", 
                        self.available_mods_by_generation[self.generation_DAG]
                    )
                    print("Increasing num sites; before:", self.num_sites)
                    # self.num_sites += 1
                    print("Increasing num sites; after:", self.num_sites)
                    mod_name = Spin_probabilistic.increase_dimension_pauli_set(
                        mod_name,
                        new_dimension = self.num_sites
                    )
                    print("Increased dimension. model now:", mod_name)
                present_terms = DataBase.get_constituent_names_from_name(mod_name)
                possible_new_terms = list(
                    set(self.available_mods_by_generation[self.generation_DAG])
                    - set(present_terms)
                )
                print("possible_new_terms:", possible_new_terms)

                # if len(possible_new_terms) == 0 :
                #     new_models = [mod_name] # bc returning empty list causes crash in QMD #TODO fix
                #     print("possible new terms empty; setting spawn stage complete.")
                #     print("new mods:", new_models)
                #     self.spawn_stage.append('Complete')
                #     return new_models

                self.model_fitness_calculation(
                    model_id = mod_id,
                    fitness_parameters = fitness[mod_id],
                    model_points = model_points
                )
                
                num_sites_this_mod = DataBase.get_num_qubits(mod_name)
                target_num_sites = num_sites_this_mod
                p_str = 'P'*target_num_sites
                # new_num_qubits = num_qubits + 1
                # mod_name_increased_dim = increase_dimension_pauli_set(mod_name) 
                for new_term in possible_new_terms: 
                    new_mod = str(
                        mod_name + 
                        p_str +
                        new_term
                    )
                    if self.determine_whether_to_include_model(mod_id) == True:
                        new_models.append(new_mod)
                        self.models_accepted[self.generation_DAG].append(new_mod)
                    else:
                        self.models_rejected[self.generation_DAG].append(new_mod)

            self.spawn_stage.append(None)

            self.num_sub_generations_per_generation[self.generation_DAG] += 1
            if (
                self.num_sub_generations_per_generation[self.generation_DAG]
                ==
                self.max_num_sub_generations_per_generation[self.generation_DAG]
            ):
                # have exhausted this sub generation
                print("Increasing Generation of DAG from {}.".format(self.generation_DAG))
                self.spawn_stage.append('end_generation')

                # if self.generation_DAG == 1:
                current_champs = kwargs['current_champs']
                print("QMD complete; comparing all branch champs so far:", current_champs)
                self.spawn_stage.append('make_new_gen')
                self.generation_DAG += 1
                self.models_accepted[self.generation_DAG] = []
                self.models_rejected[self.generation_DAG] = []
                self.sub_generation_idx = 0
                if self.generation_DAG == 2: 
                    new_models = current_champs
                elif self.generation_DAG == self.max_num_generations:
                    self.spawn_stage.append('Complete')
                    current_champs = kwargs['current_champs']
                    print("QMD complete; comparing all branch champs so far:", current_champs)
                    new_models = current_champs
                else:
                    new_mod_ids = list( # now a list of lists 
                        self.generation_champs[self.generation_DAG - 1].values()
                    )
                    new_mod_ids = sorted([i for sublist in new_mod_ids for i in sublist])            
                    new_models = [ 
                        kwargs['model_names_ids'][mod_id] for mod_id in new_mod_ids
                    ] 




        print("new models:", new_models)
        return new_models

    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        # return ModelNames.pauliSet_latex_name(
        #     name, 
        #     **kwargs
        # )

        core_operators = list(sorted(DataBase.core_operator_dict.keys()))
        num_sites = DataBase.get_num_qubits(name)
        p_str = 'P'*num_sites
        separate_terms = name.split(p_str)

        latex_terms = []
        term_type_markers = ['pauliSet', 'transverse']
        all_operators = []
        all_sites = []
        rotation_terms = []
        interaction_terms = []
        interaction_sites = []
        transverse_terms = []
        transverse_sites = []
        single_dim = False
        for term in separate_terms:
            dimension = DataBase.get_num_qubits(term)
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
                if len(operators) == 1:
                    # i.e only one operator, it's a rotation term
                    rotation_terms.append(operators[0])
                # elif len(set(operators)) == 1:
                #     # multiple sites but single axis -- interaction term
                #     interaction_terms.append(operators[0])
                #     interaction_sites.append(sites)
                else:
                    transverse_terms.append(operators)
                    transverse_sites.append(sites)

        latex_str = ''
        if len(rotation_terms) > 0:
            rotation_term = 'S_{'
            for t in rotation_terms:
                rotation_term += "{},".format(t)
            rotation_term = rotation_term[:-1]    
            rotation_term += '}'

            latex_str += rotation_term

        if len(interaction_terms) > 0:
            interaction_term = 'I_{'
            for t in interaction_terms:
                interaction_term += "{},".format(t)
            interaction_term = interaction_term[:-1]    
            interaction_term += '}'
            latex_str += interaction_term

        if len(transverse_terms) > 0:
            transverse_term = 'T_{'
            for term in transverse_terms:
                this_term = ','.join(term)
                transverse_term += "({}),".format(this_term)
            transverse_term = transverse_term[:-1]    
            transverse_term += '}'

            transverse_term += '^{'
            for site in transverse_sites:
    #             site = [int(s) for s in site]
                this_site = ','.join(site)
                transverse_term += "({}),".format(this_site)
            
            transverse_term = transverse_term[:-1]    
            transverse_term += '}'
            
            
            latex_str += transverse_term
        latex_str = "${}$".format(latex_str)
        return latex_str




def pairwise_pauli_terms(base_terms, new_site, num_sites=None):
    if num_sites is None:
        num_sites = new_site
    pairs = list(itertools.combinations_with_replacement(base_terms, 2))
    pairs.extend(list(itertools.permutations(base_terms, 2)))
    pairs = list(set(pairs))

    new_nearest_neighbours = [new_site-1, new_site]

    possible_terms = []
    for term in pairs:
        pauli_terms = 'J'.join(list(term))
        acted_on_sites = [str(i) for i in new_nearest_neighbours ]
    #     acted_on_sites = new_nearest_neighbours
        acted_on = 'J'.join(acted_on_sites)
        mod = "pauliSet_{}_{}_d{}".format(pauli_terms, acted_on, num_sites)

        possible_terms.append(mod)
    return possible_terms

def pairwise_pauli_like_like_terms(base_terms, new_site, num_sites=None):
    if num_sites is None:
        num_sites = new_site
    pairs = [(i,)*2 for i in base_terms]

    new_nearest_neighbours = [new_site-1, new_site]

    possible_terms = []
    for term in pairs:
        pauli_terms = 'J'.join(list(term))
        acted_on_sites = [str(i) for i in new_nearest_neighbours ]
        acted_on = 'J'.join(acted_on_sites)
        mod = "pauliSet_{}_{}_d{}".format(pauli_terms, acted_on, num_sites)

        possible_terms.append(mod)
    return possible_terms
