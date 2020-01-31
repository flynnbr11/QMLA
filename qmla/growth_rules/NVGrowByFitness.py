import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import NVCentreLargeSpinBath
from qmla import probe_set_generation
from qmla import database_framework


class nv_fitness_growth(
    NVCentreLargeSpinBath.nv_centre_large_spin_bath
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        self.base_models = [
            'nv_spin_x_d1',
            'nv_spin_y_d1',
            'nv_spin_z_d1',
        ]

        self.initial_models = all_unique_addition_combinations(
            self.base_models)
        self.available_axes = ['x', 'y', 'z']
        self.true_operator = 'nv_spin_x_d2PPnv_spin_y_d2PPnv_spin_z_d2PPnv_interaction_x_d2PPnv_interaction_y_d2PPnv_interaction_z_d2'
        self.plot_probe_generation_function = probe_set_generation.plus_probes_dict
        self.max_num_qubits = 4
        self.num_top_models_to_build_on = 2  # at each generation
        self.generation_DAG = 0
        self.model_fitness = {}
        # self._fitness_parameters = {}
        self.generational_fitness_parameters = {}
        self.available_mods_by_generation = {}
        self.models_to_build_on = {}
        self.model_generation_strictness = 0
        self.num_processes_to_parallelise_over = 10

        self.max_num_models_by_shape = {
            1: 7,
            2: 20,
            'other': 0
        }

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        # fitness = kwargs['fitness_parameters']
        model_points = kwargs['branch_model_points']
        branch_models = list(model_points.keys())

        # keep track of generation_DAG
        ranked_model_list = sorted(
            model_points,
            key=model_points.get,
            reverse=True
        )
        models_to_build_on = ranked_model_list[:
                                               self.num_top_models_to_build_on]
        self.models_to_build_on[self.generation_DAG] = models_to_build_on
        new_models = []

        if self.spawn_stage[-1] is None:
            for mod_id in self.models_to_build_on[self.generation_DAG]:
                self.model_fitness_calculation(
                    model_id=mod_id,
                    # fitness_parameters = fitness[mod_id],
                    model_points=model_points
                )
                mod_name = kwargs['model_names_ids'][mod_id]
                num_qubits = database_framework.get_num_qubits(mod_name)
                new_num_qubits = num_qubits + 1
                mod_name_increased_dim = increase_dimension_keep_terms_nv_model(
                    model_name=mod_name,
                    new_dimension=new_num_qubits
                )

                new_terms = [
                    "nv_interaction_{}_d{}".format(axis, new_num_qubits)
                    for axis in self.available_axes
                ]

                self.available_mods_by_generation[self.generation_DAG] = all_unique_addition_combinations(
                    new_terms)
                for new_term in self.available_mods_by_generation[self.generation_DAG]:
                    if self.determine_whether_to_include_model(mod_id) == True:
                        # new_term = "nv_interaction_{}_d{}".format(axis, new_num_qubits)
                        p_str = 'P' * new_num_qubits
                        new_mod = str(
                            mod_name_increased_dim
                            + p_str
                            + new_term
                        )
                        new_mod = database_framework.alph(new_mod)
                        new_models.append(new_mod)
            self.spawn_stage.append('Complete')

        self.generation_DAG += 1
        return new_models
        # return super().generate_models(
        #     model_list,
        #     **kwargs
        # )

    def model_fitness_calculation(
        self,
        model_id,
        # fitness_parameters, # of this model_id
        model_points,
        **kwargs
    ):
        # TODO make fitness parameters within QMD
        # pass
        # print("model fitness function. fitness params:", fitness_parameters)
        max_wins_model_points = max(model_points.values())

        win_ratio = model_points[model_id] / max_wins_model_points

        if self.model_generation_strictness == 0:
            # keep all models and work out relative fitness
            fitness = (
                win_ratio
                # win_ratio * fitness_parameters['r_squared']
            )**2
            # fitness = 1
        else:
            # only consider the best model
            # turn off all others
            if model_id == ranked_model_list[0]:
                fitness = 1
            else:
                fitness = 0

        if model_id not in sorted(self.model_fitness.keys()):
            self.model_fitness[model_id] = {}
        print("Setting fitness for {} to {}".format(model_id, fitness))
        self.model_fitness[model_id][self.generation_DAG] = fitness

    def determine_whether_to_include_model(
        self,
        model_id
    ):
        # biased coin flip
        fitness = self.model_fitness[model_id][self.generation_DAG]
        rand = np.random.rand()
        to_generate = (rand < fitness)
        return to_generate

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


def increase_dimension_keep_terms_nv_model(
    model_name,
    new_dimension,
):
    current_dimension = database_framework.get_num_qubits(model_name)
    p_str = 'P' * new_dimension
    individual_terms = database_framework.get_constituent_names_from_name(model_name)

    spin_terms = []
    interaction_terms = []
    for term in individual_terms:
        components = term.split('_')
        for l in components:
            if l[0] == 'd':
                dim = int(l.replace('d', ''))
            elif l == 'spin':
                term_type = 'spin'
            elif l == 'interaction':
                term_type = 'interaction'
            elif l in ['x', 'y', 'z']:
                pauli = l

        if term_type == 'spin':
            spin_terms.append(pauli)
        elif term_type == 'interaction':
            interaction_terms.append(pauli)

    all_terms = []
    for term in spin_terms:
        new_term = "nv_spin_{}_d{}".format(term, new_dimension)
        all_terms.append(new_term)

    for term in interaction_terms:
        new_term = "nv_interaction_{}_d{}".format(term, new_dimension)
        all_terms.append(new_term)

    model_string = p_str.join(all_terms)
    model_string = database_framework.alph(model_string)
    return model_string


def all_unique_addition_combinations(model_list):
    all_models = []
    all_combinations = []
    num_mods = len(model_list) + 1

    for i in range(1, num_mods):
        new_combinations = list(itertools.combinations(model_list, i))
        all_combinations.extend(new_combinations)
    num_qubits = database_framework.get_num_qubits(model_list[0])
    p_str = 'P' * num_qubits

    for combination in all_combinations:
        model = p_str.join(combination)
        all_models.append(model)

    return all_models
