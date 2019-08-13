import numpy as np
import itertools
import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelNames
import ModelGeneration
import Heuristics

import SuperClassGrowthRule
import NV_centre_large_spin_bath
import NV_grow_by_fitness
import Spin_probabilistic

flatten = lambda l: [item for sublist in l for item in sublist]  # flatten list of lists


class nearestNeighbourPauli2D(
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
        self.lattice_dimension = 1
        self.initial_num_sites = 2

        self.topology = ModelGeneration.topology_grid(
            dimension = self.lattice_dimension,
            num_sites = self.initial_num_sites
        )
        self.initially_connected_sites = self.topology.get_nearest_neighbour_list()

        self.true_operator = 'pauliSet_xJx_1J2_d2PPpauliSet_yJy_1J2_d2'
        self.true_operator = DataBase.alph(self.true_operator)
        self.qhl_models = [self.true_operator]
        self.base_terms = [
            # 'x', 
            # 'y', 
            'z'
        ]

        self.initial_models = pauli_like_like_terms_connected_sites(
            connected_sites = self.initially_connected_sites, 
            base_terms = self.base_terms, 
            num_sites = self.topology.num_sites() 
        )

        # fitness calculation parameters. fitness calculation inherited.
        self.num_top_models_to_build_on = 1 # 'all' # at each generation Badassness parameter
        self.model_generation_strictness = 0 #1 #-1 
        self.fitness_win_ratio_exponent = 3

        self.generation_DAG = 1
        self.max_num_sites = 4
        self.max_num_generations = self.max_num_sites - self.initial_num_sites + self.generation_DAG


        self.model_fitness = {}
        self.models_rejected = {
            self.generation_DAG : []
        }
        self.models_accepted = {
            self.generation_DAG : []
        }

        self.tree_completed_initially = False
        self.spawn_stage = [None]
        # if len(self.initial_models) == 1:
        #     self.spawn_stage.append('make_new_generation')
        self.available_mods_by_generation = {}
        self.available_mods_by_generation[self.generation_DAG] = pauli_like_like_terms_connected_sites(
            connected_sites = self.initially_connected_sites, 
            base_terms = self.base_terms, 
            num_sites = self.topology.num_sites() 
        )
        self.site_connections_considered = self.initially_connected_sites
        self.max_num_sub_generations_per_generation = {
            self.generation_DAG : len(self.available_mods_by_generation[self.generation_DAG])
        }
        # self.num_sub_generations_per_generation = {}
        self.models_to_build_on = {
            self.generation_DAG : {}
        }
        self.generation_champs = {
            self.generation_DAG : {}
        }
        self.sub_generation_idx = 0 
        self.counter =0

        self.max_num_models_by_shape = {
            'other' : 10
        }
        self.num_processes_to_parallelise_over = 10






    @property
    def num_sites(self):
        return self.topology.num_sites
    

    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):
        """
        new models are generated for different cases:
            * within dimension: add term from available term list until exhausted (greedy)
            * finalise dimension: return champions of branches within this dimension to determine
                which model(s) to build on for next generation
            * new dimension: add site to topology; get available term list; return model list 
                of previous champ(s) plus each of newest terms
            * finalise QMD: return generation champs 
                (here generation directly corresponds to number of sites)
        cases are indicated by self.spawn_stage
        """

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
        self.sub_generation_idx += 1 
        self.models_to_build_on[self.generation_DAG][self.sub_generation_idx] = models_to_build_on
        # self.generation_champs[self.generation_DAG][self.sub_generation_idx] = models_to_build_on
        self.generation_champs[self.generation_DAG][self.sub_generation_idx] = [
            kwargs['model_names_ids'][models_to_build_on[0]]
        ]

        self.counter+=1
        new_models = []

        # print("[generate models] counter", self.counter)
        # print("[generate models] input model list", model_list)
        # print("[generate models] ranked model list", models_to_build_on)
        # print("[generate models] mods to build on:", [self.latex_name(kwargs['model_names_ids'][m]) for m in models_to_build_on])

        if self.spawn_stage[-1] == None:
            # within dimension; just add each term in available terms to 
            # old models (probabilistically). 

            if self.sub_generation_idx == self.max_num_sub_generations_per_generation[self.generation_DAG]:
                # give back champs from this generation and indicate to make new generation
                print("exhausted this generation.")
                print("generation champs:", self.generation_champs[self.generation_DAG])
                self.spawn_stage.append('make_new_generation')
                new_models = [
                    self.generation_champs[self.generation_DAG][k] for k in 
                    list(self.generation_champs[self.generation_DAG].keys())
                ]
                new_models = flatten(new_models)
                print("new mods:", new_models)

                if self.generation_DAG == self.max_num_generations:
                    # this was the final generation to learn.
                    # instead of building new generation, skip straight to Complete stage
                    self.spawn_stage.append('Complete')

            else:
                for mod_id in self.models_to_build_on[self.generation_DAG][self.sub_generation_idx]:
                    mod_name = kwargs['model_names_ids'][mod_id]

                    present_terms = DataBase.get_constituent_names_from_name(mod_name)
                    possible_new_terms = list(
                        set(self.available_mods_by_generation[self.generation_DAG])
                        - set(present_terms)
                    )

                    # print("mod_name:", mod_name)
                    # print("available terms:", self.available_mods_by_generation[self.generation_DAG])
                    # print("present terms:", present_terms)
                    # print("possible_new_terms:", possible_new_terms)
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
                        new_mod = DataBase.alph(new_mod)
                        if self.determine_whether_to_include_model(mod_id) == True:
                            new_models.append(new_mod)
                            self.models_accepted[self.generation_DAG].append(new_mod)
                        else:
                            self.models_rejected[self.generation_DAG].append(new_mod)
        elif self.spawn_stage[-1] == 'make_new_generation':
            self.generation_DAG += 1
            self.sub_generation_idx = 0 

            self.models_to_build_on = {
                self.generation_DAG : {}
            }
            self.generation_champs = {
                self.generation_DAG : {}
            }
            self.models_rejected = {
                self.generation_DAG : []
            }
            self.models_accepted = {
                self.generation_DAG : []
            }
            self.topology.add_site()
            nearest_neighbours = self.topology.get_nearest_neighbour_list()
            new_connections = list(
                set(nearest_neighbours) - set(self.site_connections_considered)
            )
            self.site_connections_considered.extend(new_connections)
            possible_new_terms = pauli_like_like_terms_connected_sites(
                connected_sites = new_connections, 
                base_terms = self.base_terms,
                num_sites = self.topology.num_sites()
            )
            print("Making generation ", self.generation_DAG)
            print("new connections:", new_connections)
            print("possible_new_terms:", possible_new_terms)
            self.available_mods_by_generation[self.generation_DAG] = possible_new_terms
            self.max_num_sub_generations_per_generation[self.generation_DAG] = len(possible_new_terms)

            for mod_id in models_to_build_on:
                new_num_sites = self.topology.num_sites()
                mod_name = kwargs['model_names_ids'][mod_id]
                mod_name = Spin_probabilistic.increase_dimension_pauli_set(
                    mod_name,
                    new_dimension = new_num_sites
                )

                self.model_fitness_calculation(
                    model_id = mod_id,
                    fitness_parameters = fitness[mod_id],
                    model_points = model_points
                )

                p_str = 'P'*new_num_sites
                print("[new generation] p_str:", p_str)
                print("[new generation] possible new terms:", possible_new_terms)
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
            # if self.max_num_sub_generations_per_generation[self.generation_DAG] == 1:
            #     self.spawn_stage.append('make_new_generation')



        elif self.spawn_stage[-1] == 'Complete':
            # return list of generation champs to determine final winner
            champs_all_generations = []
            for gen_idx in list(self.generation_champs.keys()): 
                sub_indices = list(self.generation_champs[gen_idx].keys())
                max_sub_idx = max(sub_indices)
                champ_this_generation =  self.generation_champs[gen_idx][max_sub_idx]
                champs_all_generations.append(champ_this_generation)
                new_models = champ_this_generation
            print(
                "Model generation complete.", 
                "returning list of champions to determine global champion:",
                new_models
            )


        elif self.spawn_stage[-1] == 'Complete':
            return model_list
        new_models = list(set(new_models))
        print("New models:", new_models)
        return new_models

    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        # print("[latex name fnc] name:", name)
        core_operators = list(sorted(DataBase.core_operator_dict.keys()))
        num_sites = DataBase.get_num_qubits(name)
        p_str = 'P'*num_sites
        separate_terms = name.split(p_str)

        site_connections = {}
        for c in list(itertools.combinations(list(range(num_sites+1)), 2)):
            site_connections[c] = []

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
                sites = tuple([int(a) for a in sites])
                op = operators[0] # assumes like-like pauli terms like xx, yy, zz
                site_connections[sites].append(op)

        ordered_connections = list(sorted(site_connections.keys()))
        latex_term = ""

        for c in ordered_connections:
            if len(site_connections[c]) > 0:
                this_term = "\sigma_{"
                this_term += str(c)
                this_term += "}"
                this_term += "^{"
                for t in site_connections[c]:
                    this_term += "{}".format(t)
                this_term += "}"
                latex_term += this_term
        latex_term = "${}$".format(latex_term)
        return latex_term



def pauli_like_like_terms_connected_sites(
    connected_sites, 
    base_terms, 
    num_sites
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