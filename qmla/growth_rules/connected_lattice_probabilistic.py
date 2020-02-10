import numpy as np
# import itertools
import sys
import os

# import ConnectedLattice

# sys.path.append(os.path.abspath('..'))
# import Heuristics
# import SystemTopology
# import ModelGeneration
# import ModelNames
# import ProbeGeneration
# import DataBase

from qmla.growth_rules import connected_lattice
import qmla.database_framework

class ConnectedLatticeProbabilistic(
    connected_lattice.ConnectedLattice
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
        self.fitness_minimum = 0 # TODO this is overwritten by default set in parent class -- add to setup?
        self.fitness_maximum = 1

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        if self.spawn_stage[-1] == 'Start':
            new_models = self.available_mods_by_generation[self.generation_DAG]
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
            qmla.database_framework.alph(mod)
            for mod in new_models
            # Final check whether this model is allowed
            if self.check_model_validity(mod)
        ]

        registered_models = list(self.model_branches.keys())
        for model in new_models:
            if model not in registered_models:
                latex_model_name = self.latex_name(model)
                branch_id = (
                    self.generation_DAG
                    + len(qmla.database_framework.get_constituent_names_from_name(model))
                )
                self.model_branches[latex_model_name] = branch_id

        return new_models

    def model_group_fitness_calculation(
        self,
        model_points,
        generation=None, 
        subgeneration=None, 
        **kwargs
    ):
        ranked_model_list = sorted(
            model_points,
            key=model_points.get,
            reverse=True
        )
        new_fitnesses = {}
        self.log_print(
            ["Prob conn lattice group fitness calculation"]
        )
        for model_id in ranked_model_list:
            try:
                max_wins_model_points = max(model_points.values())
                win_ratio = model_points[model_id] / max_wins_model_points
            except BaseException:
                win_ratio = 1

            # fitness can be calculated by some other callable function here
            fitness = win_ratio 
            fitness = self.rescale_fitness(
                fitness,
                rescaled_min = self.fitness_minimum,
                rescaled_max = self.fitness_maximum
            )

            if model_id not in sorted(self.model_fitness.keys()):
                self.model_fitness[model_id] = []
            self.model_fitness[model_id].append(fitness)
            new_fitnesses[model_id] = fitness

        self.log_print(
            [
                "New fitnesses:\n", new_fitnesses
            ]
        )
        if generation and subgeneration is not None:
            self.generation_fitnesses[generation][subgeneration] = new_fitnesses
        return ranked_model_list

    def rescale_fitness(
        self, 
        fitness, 
        original_max = 1,
        original_min = 0, 
        rescaled_max = 1, 
        rescaled_min = 0.1, 
    ):
        # self.log_print(["rescale fitness min:", rescaled_min])
        if rescaled_max == rescaled_min:
            new_fitness = rescaled_max
        else:
            old_range = original_max - original_min
            new_range = rescaled_max - rescaled_min
            new_fitness = (( (fitness - original_min) * new_range )/old_range) + rescaled_min
        return new_fitness


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
                    rescaled_min = self.fitness_minimum
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



