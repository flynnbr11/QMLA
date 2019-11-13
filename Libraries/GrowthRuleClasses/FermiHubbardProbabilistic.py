import numpy as np
import itertools
import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelNames
import ModelGeneration
import SystemTopology
import Heuristics

import SuperClassGrowthRule
import FermiHubbard

flatten = lambda l: [item for sublist in l for item in sublist]  # flatten list of lists



class fermi_hubbard_probabilistic(
    FermiHubbard.fermi_hubbard_predetermined
):
    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        self.true_operator = 'FHhop_1h2_down_d2+FHhop_1h2_up_d2' # for testing  
        self.max_num_sites = 4
        self.max_num_probe_qubits = self.max_num_sites
        self.max_num_qubits = self.max_num_sites
        self.num_probes = 2**self.max_num_sites
        self.lattice_dimension = 1
        self.tree_completed_initially = False
        self.num_top_models_to_build_on = 2
        self.model_generation_strictness = 0
        self.fitness_win_ratio_exponent = 1
        self.qhl_models = [
            'FHhop_1h2_down_d2', 
            'FHhop_1h2_up_d2', 
            'FHhop_1h2_down_d2+FHhop_1h2_up_d2'
            # 'FHonsite_1_d2',
            # 'FHonsite_2_d2',
        ]

        self.true_params = {
            # term : true_param
            # 'FHhop_1h2_down_d2' : 0.96,
        }

        self.setup_growth_class()

    # def generate_models(
    #     self, 
    #     model_list, 
    #     **kwargs
    # ):

    #     model_points = kwargs['branch_model_points']
    #     self.model_group_fitness_calculation(
    #         model_points = model_points
    #     )
    #     branch_models = list(model_points.keys())
    #     ranked_model_list = sorted(
    #         model_points, 
    #         key=model_points.get, 
    #         reverse=True
    #     )
    #     if self.num_top_models_to_build_on == 'all':
    #         models_to_build_on = ranked_model_list
    #     else:
    #         models_to_build_on = ranked_model_list[:self.num_top_models_to_build_on]

    #     self.sub_generation_idx += 1 

    #     # self.generation_champs[self.generation_DAG][self.sub_generation_idx] = models_to_build_on
    #     self.generation_champs[self.generation_DAG][self.sub_generation_idx] = [
    #         kwargs['model_names_ids'][models_to_build_on[0]]
    #     ]

    #     self.counter+=1
    #     new_models = []

    #     if self.spawn_stage[-1] == 'make_new_generation':
    #         # increase generation idx; add site; get newly available terms; add greedily as above
    #         self.new_generation()

    #     if self.spawn_stage[-1] == None:
    #         # new models given by models_to_build_on plus terms in available_terms (greedy)
    #         new_models = self.add_terms_greedy(
    #             models_to_build_on = models_to_build_on, 
    #             available_terms = self.available_mods_by_generation[self.generation_DAG],
    #             model_names_ids = kwargs['model_names_ids'],
    #             model_points = model_points
    #         )

    #     return new_models

    # def new_generation(
    #     self
    # ):
    #     # Housekeeping: generational elements
    #     self.generation_DAG += 1
    #     self.sub_generation_idx = 0 

    #     self.models_to_build_on[self.generation_DAG] =  {}
    #     self.generation_champs[self.generation_DAG] = {}
    #     self.models_rejected[self.generation_DAG] = []
    #     self.models_accepted[self.generation_DAG] = []

    #     # Increases topology and retrieve effects e.g. new sites
    #     self.topology.add_site()

    #     new_connections = self.topology.new_connections[-1]
    #     new_sites = self.topology.new_site_indices[-1]
    #     num_sites = self.topology.num_sites()
    #     newly_available_terms = self.generate_terms_from_new_site(
    #         connected_sites = new_connections, 
    #         num_sites = num_sites, 
    #         new_sites = new_sites
    #     )
    #     self.log_print(
    #         [
    #         "Making new generation",
    #         "new terms:", newly_available_terms
    #         ]
    #     )

    #     self.available_mods_by_generation[self.generation_DAG] = newly_available_terms
    #     self.spawn_stage.append(None)

    # def add_terms_greedy(
    #     self, 
    #     models_to_build_on, 
    #     available_terms, 
    #     model_names_ids,
    #     model_points,
    #     **kwargs
    # ):
    #     # models_to_build_on = [
    #     #     kwargs['model_names_ids'][mod_id] for mod_id in models_to_build_on
    #     # ]
    #     new_models = []
    #     for mod_id in models_to_build_on:
    #         mod_name = model_names_ids[mod_id]
    #         mod_name = self.match_dimension(mod_name, self.topology.num_sites())
    #         present_terms = DataBase.get_constituent_names_from_name(mod_name)
    #         print("[fermi hubbard] model {} has present terms {}".format(mod_name, present_terms))
    #         terms_to_add = list(
    #             set(available_terms)
    #             - set(present_terms)
    #         )

    #         if len(terms_to_add) == 0:
    #             # this dimension exhausted
    #             # return branch champs for this generation so far
    #             # such that final branch computes this generation champion
    #             self.spawn_stage.append('make_new_generation')
    #             new_models = [
    #                 self.generation_champs[self.generation_DAG][k] for k in 
    #                 list(self.generation_champs[self.generation_DAG].keys())
    #             ]
    #             new_models = flatten(new_models)
    #             # new_models = [new_models[0]] # hack to force non-crash for single generation
    #             self.log_print(
    #                 [
    #                     "No remaining available terms. Completing generation",
    #                     self.generation_DAG, 
    #                     "\nModels:", new_models
    #                 ]
    #             )
    #             if self.generation_DAG == self.max_num_generations:
    #                 # this was the final generation to learn.
    #                 # instead of building new generation, skip straight to Complete stage                            
    #                 self.log_print(
    #                     [
    #                         "Completing growth rule"
    #                     ]
    #                 )
    #                 self.spawn_stage.append('Complete')
    #         else:
    #             # self.model_fitness_calculation(
    #             #     model_id = mod_id,
    #             #     model_points = model_points
    #             # )
    #             for term in terms_to_add:
    #                 new_mod = "{}+{}".format(
    #                     mod_name, 
    #                     term    
    #                 )
    #                 # new_mod = DataBase.alph(new_mod) # TODO fix DataBase.alph to take + like terms
    #                 if self.determine_whether_to_include_model(mod_id) == True:
    #                     new_models.append(new_mod)
    #                     self.models_accepted[self.generation_DAG].append(new_mod)
    #                 else:
    #                     self.models_rejected[self.generation_DAG].append(new_mod)
    #     return new_models

    def match_dimension(
        self,
        mod_name,
        num_sites, 
        **kwargs
    ):
        dimension_matched_name = match_dimension_hubbard(
            mod_name, 
            num_sites,
        )    
        return dimension_matched_name

    def generate_terms_from_new_site(
        self, 
        **kwargs 
    ):
        
        return generate_new_terms_hubbard(**kwargs)

    def combine_terms(
        self, 
        terms, 
    ):
        addition_string = '+'
        terms = sorted(terms)
        return addition_string.join(terms)


def generate_new_terms_hubbard( 
    connected_sites,
    num_sites,
    new_sites, 
    **kwargs     
):
    new_terms = []
    for pair in connected_sites:
        i = pair[0]
        j = pair[1]
        for spin in ['up', 'down']:
            hopping_term = "FHhop_{}h{}_{}_d{}".format(
                i, j, spin, num_sites 
            )
            new_terms.append(hopping_term)

    # for site in new_sites:
    #     onsite_term = "FHonsite_{}_d{}".format(
    #         site, num_sites
    #     )
        
    #     new_terms.append(onsite_term)
        
    return new_terms

def match_dimension_hubbard(
    model_name, 
    num_sites,
    **kwargs
):
    redimensionalised_terms = []
    terms = model_name.split('+')
    for term in terms:
        parts = term.split('_')
        for part in parts:
            if part[0] == 'd' and part not in ['down', 'double']:
                parts.remove(part)

        parts.append("d{}".format(num_sites))
        new_term = "_".join(parts)
        redimensionalised_terms.append(new_term)
    new_model_name = "+".join(redimensionalised_terms)
    return new_model_name
    
    