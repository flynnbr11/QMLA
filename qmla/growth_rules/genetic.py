import numpy as np
import itertools
import sys
import os
import random
import copy
import scipy
import time

from qmla.growth_rules import growth_rule_super
# from qmla.growth_rules import GeneticAlgorithm
from qmla import experiment_design_heuristics
from qmla import topology
# from qmla import model_generation
from qmla import model_naming
from qmla import probe_set_generation
from qmla import database_framework

# flatten list of lists
def flatten(l): return [item for sublist in l for item in sublist]


class Genetic(
    growth_rule_super.GrowthRuleSuper
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
        # self.true_model = 'pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_2J3_yJy_d4+pauliSet_1J4_yJy_d4'
        # self.true_model = 'pauliSet_1J2_xJx_d3+pauliSet_1J2_yJy_d3+pauliSet_2J3_yJy_d3+pauliSet_2J3_zJz_d3'
        # self.ising_full_connectivity = 'pauliSet_1J2_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4'
        self.ising_full_connectivity = 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5'
        self.heisenberg_xxz_small = 'pauliSet_1J2_xJx_d3+pauliSet_1J3_yJy_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3'
        self.true_model = self.heisenberg_xxz_small
        self.true_model = database_framework.alph(self.true_model)
        self.num_sites = database_framework.get_num_qubits(self.true_model)
        self.num_probes = 5
        self.base_terms = [
            # 'x', 'z',
            'x', 'y',  'z'
        ]
        self.mutation_probability = 0.1

        self.genetic_algorithm = GeneticAlgorithmQMLA(
            num_sites=self.num_sites,
            base_terms=self.base_terms,
            mutation_probability=self.mutation_probability,
            log_file=self.log_file
        )

        self.true_chromosome = self.genetic_algorithm.map_model_to_chromosome(
            self.true_model
        )
        self.true_chromosome_string = self.genetic_algorithm.chromosome_string(
            self.true_chromosome
        )
        self.num_possible_models = 2**len(self.true_chromosome)

        # self.true_model = 'pauliSet_xJx_1J2_d3+pauliSet_yJy_1J2_d3'
        self.max_num_probe_qubits = self.num_sites
        self.max_spawn_depth = 3
        self.initial_num_models = 10
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.hamming_distance_by_generation_step = {
            0: [
                hamming_distance(
                    self.true_chromosome_string,
                    self.genetic_algorithm.chromosome_string(
                        self.genetic_algorithm.map_model_to_chromosome(
                            mod
                        )
                    )
                )
                for mod in self.initial_models
            ]
        }
        self.fitness_at_step = {}

        
        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            4: self.initial_num_models * self.max_spawn_depth,
            'other': 0
        }

        self.max_time_to_consider = 5
        self.min_param = 0.48
        self.max_param = 0.52
        self.num_processes_to_parallelise_over = 10

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        # print("[Genetic] Calling generate_models")
        self.log_print(
            [
                "Spawn step:", kwargs['spawn_step']
            ]
        )
        model_points = kwargs['branch_model_points']
        # print("Model points:", model_points)
        # print("kwargs: ", kwargs)
        self.fitness_at_step[kwargs['spawn_step']] = model_points
        model_fitnesses = {}
        for m in list(model_points.keys()):
            mod = kwargs['model_names_ids'][m]
            model_fitnesses[mod] = model_points[m]

        # print("Model fitnesses:", model_fitnesses)
        new_models = self.genetic_algorithm.genetic_algorithm_step(
            model_fitnesses=model_fitnesses,
            num_pairs_to_sample=self.initial_num_models / 2
        )

        hamming_distances = [
            hamming_distance(
                self.true_chromosome_string,
                self.genetic_algorithm.chromosome_string(
                    self.genetic_algorithm.map_model_to_chromosome(
                        mod
                    )
                )
            )
            for mod in new_models
        ]
        self.hamming_distance_by_generation_step[
            kwargs['spawn_step']] = hamming_distances

        return new_models

    def latex_name(
        self,
        name,
        **kwargs
    ):
        # print("[latex name fnc] name:", name)
        core_operators = list(sorted(database_framework.core_operator_dict.keys()))
        num_sites = database_framework.get_num_qubits(name)
        p_str = 'P' * num_sites
        p_str = '+'
        separate_terms = name.split(p_str)

        site_connections = {}
        for c in list(itertools.combinations(list(range(num_sites + 1)), 2)):
            site_connections[c] = []

        term_type_markers = ['pauliSet', 'transverse']
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


def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


class GeneticAlgorithmQMLA():
    def __init__(
        self,
        num_sites,
        base_terms=['x', 'y', 'z'],
        mutation_probability=0.1,
        log_file=None, 
    ):
        self.num_sites = num_sites
        self.base_terms = base_terms
        self.get_base_chromosome()
#         self.addition_str = 'P'*self.num_sites
        self.addition_str = '+'
        self.mutation_probability = mutation_probability
        self.previously_considered_chromosomes = []
        self.log_file = log_file
        self.chromosomes_at_generation = {}
        self.genetic_generation = 0


    def get_base_chromosome(self):
        """
        get empty chromosome with binary
        position for each possible term
        within this model type

        Basic: all pairs can be connected operator o on sites i,j:
        e.g. i=4,j=7,N=9: IIIoIIoII
        """

        basic_chromosome = []
        chromosome_description = []
        for i in range(1, 1 + self.num_sites):
            for j in range(i + 1, 1 + self.num_sites):
                for t in self.base_terms:
                    pair = (int(i), int(j), t)
                    pair = tuple(pair)
                    basic_chromosome.append(0)
                    chromosome_description.append(pair)

        self.chromosome_description = chromosome_description
        self.chromosome_description_array = np.array(
            self.chromosome_description)
        self.basic_chromosome = np.array(basic_chromosome)
        self.num_terms = len(self.basic_chromosome)
        # print("Chromosome definition:", self.chromosome_description_array)
#         binary_combinations = list(itertools.product([0,1], repeat=self.num_terms))
#         binary_combinations = [list(b) for b in binary_combinations]
#         self.possible_chromosomes = np.array(binary_combinations)

    def map_chromosome_to_model(
        self,
        chromosome,
    ):
        if isinstance(chromosome, str):
            chromosome = list(chromosome)
            chromosome = np.array([int(i) for i in chromosome])

        nonzero_postions = chromosome.nonzero()
        present_terms = list(
            self.chromosome_description_array[nonzero_postions]
        )
        term_list = []
        for t in present_terms:
            i = t[0]
            j = t[1]
            o = t[2]

            term = 'pauliSet_{i}J{j}_{o}J{o}_d{N}'.format(
                i=i,
                j=j,
                o=o,
                N=self.num_sites
            )
            term_list.append(term)

        model_string = self.addition_str.join(term_list)
        # print(
        #     "[GeneticAlgorithm mapping chromosome to model] \
        #     \n chromosome: {} \
        #     \n model string: {}\
        #     \n nonzero_postions: {}".format(
        #     chromosome,
        #     model_string,
        #     nonzero_postions
        #     )
        # )

        return model_string

    def map_model_to_chromosome(
        self,
        model
    ):
        terms = database_framework.get_constituent_names_from_name(model)
        chromosome_locations = []
        for term in terms:
            components = term.split('_')
            try:
                components.remove('pauliSet')
            except BaseException:
                print(
                    "[GA - map model to chromosome] \
                    \nCannot remove pauliSet from components:",
                    components,
                    "\nModel:", model
                )
                raise
            core_operators = list(sorted(database_framework.core_operator_dict.keys()))
            for l in components:
                if l[0] == 'd':
                    dim = int(l.replace('d', ''))
                elif l[0] in core_operators:
                    operators = l.split('J')
                else:
                    sites = l.split('J')
            # get strings when splitting the list elements
            sites = [int(s) for s in sites]
            sites = sorted(sites)

            term_desc = [sites[0], sites[1], operators[0]]
            term_desc = tuple(term_desc)
            term_chromosome_location = self.chromosome_description.index(
                term_desc)
            chromosome_locations.append(term_chromosome_location)
        new_chromosome = copy.copy(self.basic_chromosome)
        new_chromosome[chromosome_locations] = 1
        return new_chromosome

    def chromosome_string(
        self,
        c
    ):
        b = [str(i) for i in c]
        return ''.join(b)

    def random_initial_models(
        self,
        num_models=5
    ):
        new_models = []
        self.chromosomes_at_generation[0] = []

        while len(new_models) < num_models:
            r = random.randint(1, 2**self.num_terms)
            r = format(r, '0{}b'.format(self.num_terms))

            if self.chromosome_string(
                    r) not in self.previously_considered_chromosomes:
                r = list(r)
                r = np.array([int(i) for i in r])
                mod = self.map_chromosome_to_model(r)

                self.previously_considered_chromosomes.append(
                    self.chromosome_string(r)
                )
                self.chromosomes_at_generation[0].append(
                    self.chromosome_string(r)
                )

                new_models.append(mod)

        # new_models = list(set(new_models))
        # print("Random initial models:", self.previously_considered_chromosomes)
        # print("Random initial models:", new_models)
        return new_models

    def selection(
        self,
        model_fitnesses,
        num_chromosomes_to_select=2,
        num_pairs_to_sample=5,
        **kwargs
    ):
        models = list(model_fitnesses.keys())
        num_nonzero_fitness_models = np.count_nonzero(
            list(model_fitnesses.values()))
        num_models = len(models)

        max_possible_num_combinations = scipy.misc.comb(
            num_nonzero_fitness_models, 2)

        self.log_print(
            [
                "[Selection] Getting max possible combinations: {} choose {} = {}".format(
                num_nonzero_fitness_models, 2, max_possible_num_combinations
                )
            ]        
        )

        num_pairs_to_sample = min(
            num_pairs_to_sample,
            max_possible_num_combinations
        )

        chromosome_fitness = {}
        chromosomes = {}
        weights = []
        self.log_print(
            [
            "Getting weights of input models."
            ]
        )
        for model in models:
            self.log_print(
                [
                    "Mapping {} to chromosome".format(model) 
                ]
            )
            chrom = self.map_model_to_chromosome(model)
            chromosomes[model] = chrom
            weights.append(model_fitnesses[model])
        weights /= np.sum(weights)  # normalise so weights are probabilities
        self.log_print(
            [
            "Models: {} \nWeights: {}".format(models, weights)
            ]
        )
        new_chromosome_pairs = []
        combinations = []

        while len(new_chromosome_pairs) < num_pairs_to_sample:
            # TODO: better way to sample multiple pairs
            selected_models = np.random.choice(
                models,
                size=num_chromosomes_to_select,
                p=weights,
                replace=False
            )
            selected_chromosomes = [
                chromosomes[mod] for mod in selected_models
            ]
            combination = ''.join(
                [
                    str(i) for i in list(selected_chromosomes[0] + selected_chromosomes[1])
                ]
            )
            # print("Trying combination {}".format(combination))
            if combination not in combinations:
                combinations.append(combination)
                new_chromosome_pairs.append(selected_chromosomes)
                self.log_print(
                    [
                    "Including selected models:", selected_models
                    ]
                )
                self.log_print(
                    [
                    "Now {} combinations of {}".format(
                        len(new_chromosome_pairs),
                        num_pairs_to_sample
                    )
                    ]
                )
        self.log_print(
            [
            "[Selection] Returning {}".format(new_chromosome_pairs)
            ]
        )
        return new_chromosome_pairs

    def crossover(
        self,
        chromosomes,
    ):
        """
        This fnc assumes only 2 chromosomes to crossover
        and does so in the most basic method of splitting
        down the middle and swapping
        """

        c1 = copy.copy(chromosomes[0])
        c2 = copy.copy(chromosomes[1])

        x = int(len(c1) / 2)
        tmp = c2[:x].copy()
        c2[:x], c1[:x] = c1[:x], tmp

        return c1, c2

    def mutation(
        self,
        chromosomes,
    ):
        copy_chromosomes = copy.copy(chromosomes)
        for c in copy_chromosomes:
            if np.all(c == 0):
                print(
                    "Input chomosome {} has no interactions -- forcing mutation".format(
                        c)
                )
                mutation_probability = 1.0
            else:
                mutation_probability = self.mutation_probability

            if np.random.rand() < mutation_probability:
                idx = np.random.choice(range(len(c)))
                # print("Flipping idx {}".format(idx))
                if c[idx] == 0:
                    c[idx] = 1
                elif c[idx] == 1:
                    c[idx] = 0
        return chromosomes

    def genetic_algorithm_step(
        self,
        model_fitnesses,
        num_pairs_to_sample=5
    ):
        new_models = []
        chromosomes_selected = self.selection(
            model_fitnesses=model_fitnesses,
            num_pairs_to_sample=num_pairs_to_sample
        )
        new_chromosomes_this_generation = []
        for chromosomes in chromosomes_selected:
            new_chromosomes = self.crossover(chromosomes)
            new_chromosomes = self.mutation(new_chromosomes)
            new_chromosomes_this_generation.extend(new_chromosomes)

            new_models.extend(
                [
                    self.map_chromosome_to_model(c)
                    for c in new_chromosomes
                ]
            )

        self.previously_considered_chromosomes.extend([
            self.chromosome_string(r) for r in new_chromosomes_this_generation
            ]
        )
        self.genetic_generation += 1
        self.chromosomes_at_generation[self.genetic_generation] = [
            self.chromosome_string(r) for r in new_chromosomes_this_generation
        ]
        return new_models

    def log_print(
        self,
        to_print_list
    ):
        identifier = "[Genetic algorithm]"
        if type(to_print_list) != list:
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(
                identifier,
                str(to_print),
                file=write_log_file,
                flush=True
            )

