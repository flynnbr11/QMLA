import numpy as np
import itertools
import sys
import os

import ConnectedLattice

sys.path.append(os.path.abspath('..'))
import Heuristics
import SystemTopology
import ModelGeneration
import ModelNames
import ProbeGeneration
import DataBase

class connected_lattice_revivals(
    ConnectedLattice.connected_lattice
):
    def __init__(
        self,
        growth_generation_rule, 
        **kwargs
    ):
        
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.weighted_fitnesses = {}
        self.fitness_minimum = 0.1 # TODO this is overwritten by default set in parent class -- add to setup?
        self.fitness_maximum = 1

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        if self.spawn_stage[-1] == 'Start':
            new_models = self.available_mods_by_generation[self.generation_DAG]
            # self.log_print(["Spawning initial models:", new_models])
            self.spawn_stage.append(None)

        else:
            ranked_model_list = self.model_group_fitness_calculation(
                model_points = kwargs['branch_model_points'],
                generation = self.generation_DAG, 
                subgeneration = self.sub_generation_idx
            )
            if self.spawn_stage[-1] == 'finish_generation':
                # get all models of previous generation and compute compound fitness
                ranked_model_list = self.generation_compound_fitness(
                    generation = self.generation_DAG, 
                    model_name_ids = kwargs['model_names_ids']
                )
                self.spawn_stage.append('make_new_generation')

            if self.num_top_models_to_build_on == 'all':
                models_to_build_on = ranked_model_list
            else:
                models_to_build_on = ranked_model_list[
                    :self.num_top_models_to_build_on
                ]

            self.sub_generation_idx += 1
            self.generation_champs[self.generation_DAG][self.sub_generation_idx] = [
                kwargs['model_names_ids'][models_to_build_on[0]]
            ]
            self.counter += 1
            new_models = []
            
            if self.spawn_stage[-1] == 'make_new_generation':
                # increase generation idx; add site; get newly available terms;
                # add greedily as above
                self.new_generation()
                # starting_new_generation = True

            if self.spawn_stage[-1] is None:
                # new models given by models_to_build_on plus terms in
                # available_terms (greedy)
                new_models = self.add_terms_greedy(
                    models_to_build_on=models_to_build_on,
                    available_terms=self.available_mods_by_generation[self.generation_DAG],
                    model_names_ids=kwargs['model_names_ids'],
                    # model_points=model_points
                )

            # if starting_new_generation == True and self.spawn_stage[-1]!='Complete':
            #     self.spawn_stage.append('start_of_new_generation')

        new_models = [
            DataBase.alph(mod)
            for mod in new_models
            # Final check whether this model is allowed
            if self.check_model_validity(mod)
        ]
        # store branch idx for new models

        registered_models = list(self.model_branches.keys())
        for model in new_models:
            if model not in registered_models:
                latex_model_name = self.latex_name(model)
                branch_id = (
                    self.generation_DAG
                    + len(DataBase.get_constituent_names_from_name(model))
                )
                self.model_branches[latex_model_name] = branch_id

        return new_models


    def generation_compound_fitness(
        self, 
        generation, 
        model_name_ids
    ):
        self.log_print([
            "generation compound fitness function for generation {}".format(generation)
            ]
        )

        subgeneration_ids = list(self.generation_fitnesses[generation].keys())
        subgen_champ_comparison_id = subgeneration_ids[-1] 
        subgen_champ_fitnesses = self.generation_fitnesses[generation][subgen_champ_comparison_id]
        subgen_champs = list(subgen_champ_fitnesses.keys())
        subgeneration_ids.remove(subgen_champ_comparison_id)


        weighted_fitnesses = {}
        for s in subgeneration_ids: 
            this_subgen_fitnesses = self.generation_fitnesses[generation][s]
                
            this_subgen_models = list(this_subgen_fitnesses.keys())
            corresponding_subgen_champ = list(set(subgen_champs).intersection(set(this_subgen_models)))[0]
            corresponding_weight = subgen_champ_fitnesses[corresponding_subgen_champ]
                
            for mod in this_subgen_models: 
                weighted_f = corresponding_weight * this_subgen_fitnesses[mod]
                weighted_f = self.rescale_fitness(
                    weighted_f,
                    original_min = self.fitness_minimum, 
                    rescaled_min = self.finess_minimum
                )
                if mod not in list(weighted_fitnesses.keys()):
                    weighted_fitnesses[mod] = weighted_f
                else: 
                    if weighted_f > weighted_fitnesses[mod]:
                        weighted_fitnesses[mod] = weighted_f
                        
        for mod in list(weighted_fitnesses.keys()):
            # set model fitness to this value in class definition, called by add_greedy
            self.weighted_fitnesses[mod] = weighted_fitnesses[mod]
            # print("{} : {}".format(mod, weighted_fitnesses[mod]))        

        ranked_model_list = sorted(
            weighted_fitnesses,
            key=weighted_fitnesses.get,
            reverse=True
        )
        self.log_print(
            [
                "For starting next generation, ranked model list: ", 
                ranked_model_list,
                "\nwith weights:", weighted_fitnesses
            ]
        )


        return ranked_model_list



