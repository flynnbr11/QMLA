import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import NVGrowByFitness
from qmla import probe_set_generation
from qmla import database_framework


class nv_probabilistic(
    NVGrowByFitness.nv_fitness_growth
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
        self.base_terms = [
            'x', 'y', 'z'
        ]
        self.initial_models = possible_pauli_combinations(
            base_terms=self.base_terms,
            num_sites=1
        )

        self.available_mods_by_generation = {}
        for i in range(self.max_num_qubits):
            self.available_mods_by_generation[i] = possible_pauli_combinations(
                base_terms=self.base_terms,
                num_sites=i
            )

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        fitness = kwargs['fitness_parameters']
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
                    fitness_parameters=fitness[mod_id],
                    model_points=model_points
                )
                mod_name = kwargs['model_names_ids'][mod_id]
                num_sites_this_mod = database_framework.get_num_qubits(mod_name)

                target_num_sites = num_sites_this_mod
                p_str = 'P' * target_num_sites
                # new_num_qubits = num_qubits + 1
                # mod_name_increased_dim = increase_dimension_pauli_set(mod_name)
                for new_term in self.available_mods_by_generation[self.generation_DAG]:
                    if self.determine_whether_to_include_model(mod_id) == True:
                        new_mod = str(
                            mod_name,
                            p_str,
                            new_term
                        )
                        new_models.append(new_mod)
            self.spawn_stage.append('Complete')

        self.generation_DAG += 1
        return new_models

    def latex_name(
        self,
        name,
        **kwargs
    ):
        return model_naming.pauliSet_latex_name(
            name,
            **kwargs
        )


def possible_pauli_combinations(base_terms, num_sites):
    possible_terms_tuples = list(
        itertools.combinations_with_replacement(
            base_terms, num_sites))
    possible_terms = []

    for term in possible_terms_tuples:
        pauli_terms = 'J'.join(list(term))
        acted_on_sites = [str(i) for i in range(1, num_sites + 1)]
        acted_on = 'J'.join(acted_on_sites)
        mod = "pauliSet_{}_{}_d{}".format(pauli_terms, acted_on, num_sites)

        possible_terms.append(mod)
    return possible_terms


def increase_dimension_pauli_set(initial_model, new_dimension=None):
    components = initial_model.split('_')

    for c in components:
        if c[0] == 'd':
            current_dim = int(c.replace('d', ''))
            components.remove(c)

    if new_dimension is None:
        new_dimension = current_dim + 1
    new_component = "d{}".format(new_dimension)
    components.append(new_component)
    new_mod = '_'.join(components)

    return new_mod
