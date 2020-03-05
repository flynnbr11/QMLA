import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import growth_rule_super
from qmla import experiment_design_heuristics
from qmla import topology
from qmla import model_naming
from qmla import probe_set_generation
from qmla import database_framework

__all__ = [
    'ConnectedLattice'
]

# flatten list of lists
def flatten(l): return [item for sublist in l for item in sublist]


class ConnectedLattice(
    growth_rule_super.GrowthRuleSuper
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("Connected lattice __init__. kwargs: ", kwargs)
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        # self.model_heuristic_function = experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        self.lattice_dimension = 2
        self.initial_num_sites = 2
        self.lattice_connectivity_max_distance = 1
        self.lattice_connectivity_linear_only = True
        self.lattice_full_connectivity = False
        self.max_time_to_consider = 50
        self.num_probes = 25
        # self.min_param = 0.25 # for the sake of plots
        # self.max_param = 0.75
        self.min_param = 0.0 # normal
        self.max_param = 1.0

        self.true_model = 'pauliSet_xJx_1J2_d2PPpauliSet_yJy_1J2_d2'
        self.true_model = database_framework.alph(self.true_model)
        self.qhl_models = [self.true_model]
        self.base_terms = [
            'x',
            'y',
            'z'
        ]

        # fitness calculation parameters. fitness calculation inherited.
        # 'all' # at each generation Badassness parameter
        self.num_top_models_to_build_on = 2
        self.model_generation_strictness = 0  # 1 #-1
        self.fitness_win_ratio_exponent = 3
        self.generation_DAG = 1
        self.max_num_sites = 4
        self.tree_completed_initially = False
        self.num_processes_to_parallelise_over = 10
        self.max_num_parameter_estimate = 9
        self.max_num_qubits = 4
        self.max_num_models_by_shape = {
            'other': 10
        }
        self.setup_growth_class()

    def setup_growth_class(self):
        # for classes which use connected_lattice as super class
        # which may change attributes defined above, calculate 
        # further attributes based on those, e.g. max num generations 
        # based on max num sites, defined by the class inheriting. 
        self.log_print(
            [
                "In Growth class setup fnc for {}.".format(
                    self.growth_generation_rule
                )
            ]
        )
        self.max_num_generations = (
            self.max_num_sites -
            self.initial_num_sites +
            self.generation_DAG
        )
        self.max_num_probe_qubits = self.max_num_sites
        self.topology = topology.GridTopology(
            dimension=self.lattice_dimension,
            num_sites=self.initial_num_sites,
            # nearest neighbours only,
            maximum_connection_distance=self.lattice_connectivity_max_distance,
            linear_connections_only=self.lattice_connectivity_linear_only,
            all_sites_connected=self.lattice_full_connectivity,
        )
        self.initially_connected_sites = self.topology.get_connected_site_list()

        self.true_model = database_framework.alph(self.true_model)
        self.model_fitness = {}
        self.models_rejected = {
            self.generation_DAG: []
        }
        self.models_accepted = {
            self.generation_DAG: []
        }

        self.available_mods_by_generation = {}
        self.available_mods_by_generation[self.generation_DAG] = self.generate_terms_from_new_site(
            connected_sites=self.initially_connected_sites,
            base_terms=self.base_terms,
            num_sites=self.topology.num_sites(),
            new_sites=range(1, self.topology.num_sites() + 1),
        )
        self.spawn_stage = ['Start']
        if not self.tree_completed_initially:
            self.initial_models = self.generate_models(
                model_list=['']
            )

        self.site_connections_considered = self.initially_connected_sites
        self.max_num_sub_generations_per_generation = {
            self.generation_DAG: len(self.available_mods_by_generation[self.generation_DAG])
        }
        self.models_to_build_on = {
            self.generation_DAG: {}
        }
        self.generation_champs = {
            self.generation_DAG: {}
        }
        self.generation_fitnesses = {
            self.generation_DAG: {}
        }
        self.sub_generation_idx = 0
        self.counter = 0

    @property
    def num_sites(self):
        return self.topology.num_sites()

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
            # model_points = kwargs['branch_model_points']
            ranked_model_list = self.model_group_fitness_calculation(
                model_points=kwargs['branch_model_points'],
                generation = self.generation_DAG, 
                subgeneration = self.sub_generation_idx
            )

            if self.num_top_models_to_build_on == 'all':
                models_to_build_on = ranked_model_list
            else:
                models_to_build_on = ranked_model_list[:
                    self.num_top_models_to_build_on
                ]


            self.sub_generation_idx += 1
            self.generation_champs[self.generation_DAG][self.sub_generation_idx] = [
                kwargs['model_names_ids'][models_to_build_on[0]]
            ]
            self.counter += 1
            new_models = []

            if self.spawn_stage[-1] == 'finish_generation':
                # increase generation idx; add site; get newly available terms;
                # add greedily as above
                self.new_generation()

            if self.spawn_stage[-1] is None:
                # new models given by models_to_build_on plus terms in
                # available_terms (greedy)
                new_models = self.add_terms_greedy(
                    models_to_build_on=models_to_build_on,
                    available_terms=self.available_mods_by_generation[self.generation_DAG],
                    model_names_ids=kwargs['model_names_ids'],
                    # model_points=model_points
                )

        new_models = [
            database_framework.alph(mod)
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
                    + len(database_framework.get_constituent_names_from_name(model))
                )
                self.model_branches[latex_model_name] = branch_id

        return new_models

    def new_generation(
        self
    ):
        # Housekeeping: generational elements
        self.generation_DAG += 1
        self.sub_generation_idx = 0

        self.models_to_build_on[self.generation_DAG] = {}
        self.generation_champs[self.generation_DAG] = {}
        self.models_rejected[self.generation_DAG] = []
        self.models_accepted[self.generation_DAG] = []
        self.generation_fitnesses[self.generation_DAG] = {}
        # Increase topology and retrieve effects e.g. new sites
        self.topology.add_site()

        new_connections = self.topology.new_connections[-1]
        new_sites = self.topology.new_site_indices[-1]
        num_sites = self.topology.num_sites()
        newly_available_terms = self.generate_terms_from_new_site(
            base_terms=self.base_terms,
            connected_sites=new_connections,
            num_sites=num_sites,
            new_sites=new_sites,
            print_terms=True,
        )
        self.log_print(
            [
                "Making new generation ({})".format(self.generation_DAG),
                "\tNew terms:", newly_available_terms
            ]
        )

        self.available_mods_by_generation[self.generation_DAG] = newly_available_terms
        self.spawn_stage.append(None)

    def add_terms_greedy(
        self,
        models_to_build_on,
        available_terms,
        model_names_ids,
        # model_points,
        **kwargs
    ):
        # models_to_build_on = [
        #     kwargs['model_names_ids'][mod_id] for mod_id in models_to_build_on
        # ]
        self.log_print(
            [
                "[Greedy add terms] models to build on {}".format(
                    models_to_build_on
                )
            ]
        )
        new_models = []
        models_by_parent = {}
        for mod_id in models_to_build_on:
            mod_name = model_names_ids[mod_id]
            models_by_parent[mod_id] = 0
            mod_name = self.match_dimension(
                mod_name, self.topology.num_sites())
            present_terms = database_framework.get_constituent_names_from_name(mod_name)
            terms_to_add = list(
                set(available_terms)
                - set(present_terms)
            )

            if len(terms_to_add) == 0:
                # this dimension exhausted
                # return branch champs for this generation so far
                # such that final branch computes this generation champion
                self.spawn_stage.append('finish_generation')
                new_models = [
                    self.generation_champs[self.generation_DAG][k] for k in
                    list(self.generation_champs[self.generation_DAG].keys())
                ]
                new_models = flatten(new_models)
                # new_models = [new_models[0]] # hack to force non-crash for
                # single generation
                self.log_print(
                    [
                        "No remaining available terms. Completing generation",
                        self.generation_DAG,
                        "\nSub generation champion models:", new_models
                    ]
                )
                if self.generation_DAG == self.max_num_generations:
                    # this was the final generation to learn.
                    # instead of building new generation, skip straight to
                    # Complete stage
                    self.log_print(
                        [
                            "Completing growth rule"
                        ]
                    )
                    self.spawn_stage.append('Complete')
            else:
                for term in terms_to_add:
                    new_mod = self.combine_terms(
                        terms=[mod_name, term]
                    )
                    if self.determine_whether_to_include_model(mod_id) == True:
                        new_models.append(new_mod)
                        self.models_accepted[self.generation_DAG].append(
                            new_mod)
                        models_by_parent[mod_id] += 1
                    else:
                        self.models_rejected[self.generation_DAG].append(
                            new_mod)
        self.log_print(
            ["# models added by parent: {}".format(models_by_parent)]
        )
        return new_models

    def check_model_validity(
        self,
        model,
        **kwargs
    ):
        # possibility that some models not valid; 
        # not needed by default but checked in general
        # so that GR requiring check has it built in. 
        return True

    def combine_terms(
        self,
        terms
    ):
        addition_string = 'P' * self.topology.num_sites()
        terms = sorted(terms)
        new_mod = addition_string.join(terms)
        return new_mod

    def match_dimension(
        self,
        mod_name,
        num_sites,
        **kwargs
    ):
        return increase_dimension_pauli_set(mod_name, num_sites)

    def latex_name(
        self,
        name,
        **kwargs
    ):
        # print("[latex name fnc] name:", name)
        core_operators = list(sorted(database_framework.core_operator_dict.keys()))
        num_sites = database_framework.get_num_qubits(name)
        try:
            p_str = 'P' * num_sites
            separate_terms = name.split(p_str)
        except:
            p_str = '+'
            separate_terms = name.split(p_str)

        site_connections = {}
        for c in list(itertools.combinations(list(range(num_sites + 1)), 2)):
            site_connections[c] = []

        # term_type_markers = ['pauliSet', 'transverse']
        transverse_axis = None
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
                sites = tuple([int(a) for a in sites])
                # assumes like-like pauli terms like xx, yy, zz
                op = operators[0]
                site_connections[sites].append(op)
            elif 'transverse' in components:
                components.remove('transverse')
                for l in components:
                    if l[0] == 'd':
                        transverse_dim = int(l.replace('d', ''))
                    elif l in core_operators:
                        transverse_axis = l

        ordered_connections = list(sorted(site_connections.keys()))
        latex_term = ""

        for c in ordered_connections:
            if len(site_connections[c]) > 0:
                this_term = r"\sigma_{"
                this_term += str(c)
                this_term += "}"
                this_term += "^{"
                for t in site_connections[c]:
                    this_term += "{}".format(t)
                this_term += "}"
                latex_term += this_term
        if transverse_axis is not None:
            latex_term += 'T^{}_{}'.format(transverse_axis, transverse_dim)
        latex_term = "${}$".format(latex_term)
        return latex_term

    def generate_terms_from_new_site(
        self,
        base_terms,
        connected_sites,
        num_sites,
        print_terms=False,
        **kwargs
    ):
        new_terms = pauli_like_like_terms_connected_sites(
            connected_sites=connected_sites,
            base_terms=base_terms,
            num_sites=num_sites
        )
        if print_terms == True:
            self.log_print(
                ["Generating new terms:", new_terms]
            )
        return new_terms

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
        self.log_print(
            [
                "Model group fitness calculation for input models:", 
                model_points
            ]
        )
        new_fitnesses = {}
        for model_id in ranked_model_list:
            try:
                max_wins_model_points = max(model_points.values())
                win_ratio = model_points[model_id] / max_wins_model_points
            except BaseException:
                win_ratio = 1

            if self.model_generation_strictness == 0:
                # keep all models and work out relative fitness
                fitness = (
                    win_ratio
                    # win_ratio * fitness_parameters['r_squared']
                )**self.fitness_win_ratio_exponent
                # fitness = 1
            elif self.model_generation_strictness == -1:
                fitness = 1
            else:
                # only consider the best model
                # turn off all others
                if model_id == ranked_model_list[0]:
                    fitness = 1
                else:
                    fitness = 0
            if model_id not in sorted(self.model_fitness.keys()):
                self.model_fitness[model_id] = []
            new_fitnesses[model_id] = fitness
            self.model_fitness[model_id].append(fitness)
        self.log_print(
            [
                "New fitnesses:\n", new_fitnesses
            ]
        )
        if generation and subgeneration is not None:
            self.generation_fitnesses[generation][subgeneration] = new_fitnesses
        return ranked_model_list


    def determine_whether_to_include_model(
        self,
        model_id
    ):
        # return bool: whether this model should parent a new model, randomly
        # decided according to biased coin flip
        # most recent fitness value calculated for this model
        fitness = self.model_fitness[model_id][-1]
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

    def name_branch_map(
        self,
        latex_mapping_file,
        **kwargs
    ):

        import qmla.model_naming
        # TODO get generation idx + sub generation idx

        return model_naming.branch_is_num_params_and_qubits(
            latex_mapping_file=latex_mapping_file,
            **kwargs
        )



def pauli_like_like_terms_connected_sites(
    connected_sites,
    base_terms,
    num_sites,
    **kwargs
):

    new_terms = []
    for pair in connected_sites:
        site_1 = pair[0]
        site_2 = pair[1]

        acted_on = "{}J{}".format(site_1, site_2)
        for t in base_terms:
            pauli_terms = "{}J{}".format(t, t)
            mod = "pauliSet_{}_{}_d{}".format(acted_on, pauli_terms, num_sites)
            new_terms.append(mod)
    return new_terms


def possible_pauli_combinations(base_terms, num_sites):
    # possible_terms_tuples = list(itertools.combinations_with_replacement(base_terms, num_sites))
    # possible_terms_tuples = list(itertools.combinations(base_terms, num_sites))
    possible_terms_tuples = [
        (a,) * num_sites for a in base_terms
    ]  # only hyerfine type terms; no transverse

    possible_terms = []

    for term in possible_terms_tuples:
        pauli_terms = 'J'.join(list(term))
        acted_on_sites = [str(i) for i in range(1, num_sites + 1)]
        acted_on = 'J'.join(acted_on_sites)
        mod = "pauliSet_{}_{}_d{}".format(pauli_terms, acted_on, num_sites)

        possible_terms.append(mod)
    return possible_terms


def increase_dimension_pauli_set(initial_model, new_dimension=None):
    individual_terms = database_framework.get_constituent_names_from_name(initial_model)
    separate_terms = []

    for model in individual_terms:
        components = model.split('_')

        for c in components:
            if c[0] == 'd':
                current_dim = int(c.replace('d', ''))
                components.remove(c)

        if new_dimension is None:
            new_dimension = current_dim + 1
        new_component = "d{}".format(new_dimension)
        components.append(new_component)
        new_mod = '_'.join(components)
        separate_terms.append(new_mod)

    p_str = 'P' * (new_dimension)
    full_model = p_str.join(separate_terms)
    # full_model = '+'.join(separate_terms)

    return full_model
