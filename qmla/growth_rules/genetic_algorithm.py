import numpy as np
import itertools
import sys
import os
import random
import copy
import scipy
import time
import pandas as pd
import sklearn

import qmla.database_framework

class GeneticAlgorithmQMLA():
    def __init__(
        self,
        num_sites,
        true_model,
        base_terms=['x', 'y', 'z'],
        mutation_probability=0.1,
        log_file=None, 
    ):
        self.num_sites = num_sites
        self.base_terms = base_terms
        self.get_base_chromosome()
#         self.addition_str = 'P'*self.num_sites
        self.true_model = true_model
        self.true_chromosome = self.map_model_to_chromosome(self.true_model)
        self.true_chromosome_string = self.chromosome_string(
            self.true_chromosome
        )
        self.all_zero_chromosome_string = '0'*num_sites
        self.addition_str = '+'
        self.mutation_probability = mutation_probability
        self.previously_considered_chromosomes = []
        self.log_file = log_file
        self.chromosomes_at_generation = {}
        self.delta_f_by_generation = {}
        self.genetic_generation = 0
        self.f_score_change_by_generation = {}
        self.most_elite_models_by_generation = {}
        self.best_model_unchanged = False
        self.unchanged_elite_num_generations_cutoff = 5
        


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
        return model_string

    def map_model_to_chromosome(
        self,
        model
    ):
        terms = qmla.database_framework.get_constituent_names_from_name(model)
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
            core_operators = list(sorted(qmla.database_framework.core_operator_dict.keys()))
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

    def chromosome_f_score(
        self, 
        chromosome, 
    ):
        if not isinstance(chromosome, np.ndarray):            
            chromosome = np.array([int(a) for a in list(chromosome)])
        
        return sklearn.metrics.f1_score(
            chromosome, 
            self.true_chromosome
        )

    def log_print(self, to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'GA gen {}'.format(self.genetic_generation)
        )


    def random_initial_models(
        self,
        num_models=5
    ):
        new_models = []
        self.initial_number_models = num_models
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

    ######################
    # Selection functions
    ######################

    def selection(
        self,
        chromosome_selection_probabilities,
        **kwargs
    ):
        r"""
        Wrapper for user's selected selection method. 

        Whatever method is called must return
            * prescribed_chromosomes
            * chromosomes_for_crossover - pairs
        """
        return self.basic_pair_selection(
            chromosome_selection_probabilities,
            **kwargs
        )

    def basic_pair_selection(
        self,
        chromosome_selection_probabilities,
        **kwargs
    ):
        chromosomes = list(chromosome_selection_probabilities.keys())
        probabilities = [chromosome_selection_probabilities[c] for c in chromosomes]
        # new_pair = False
        # self.log_print(
        #     [
        #         "Basic pair selection; finding new combination.",
        #         "Combinations present already:", 
        #     ]
        # )
        # while new_pair is False: 
        #     selected_chromosomes = np.random.choice(
        #         chromosomes,
        #         size=2,
        #         p=probabilities,
        #         replace=False
        #     )
        #     unique_combination = ''.join(
        #             [
        #                 str(i) for i in list(selected_chromosomes[0] + selected_chromosomes[1])
        #             ]
        #         ) 
        #     if unique_combination not in self.unique_pair_combinations_considered:
        #         new_pair = True
            
        # self.unique_pair_combinations_considered.append(unique_combination)
        selected_chromosomes = np.random.choice(
            chromosomes,
            size=2,
            p=probabilities,
            replace=False
        )

        return selected_chromosomes

    ######################
    # Crossover functions
    ######################

    def crossover(
        self,
        pair_to_crossover,
        this_generation_chromosomes=None, 
    ):
        """
        This fnc assumes only 2 chromosomes to crossover
        and does so in the most basic method of splitting
        down the middle and swapping
        """
        suggested_chromosomes =  self.one_point_crossover(pair_to_crossover)
        if this_generation_chromosomes is not None:
            self.log_print(
                [
                    "This generation chromosomes:\n", this_generation_chromosomes, 
                    "\nSuggested chromosomes:\n", suggested_chromosomes
                ]
            )

        return suggested_chromosomes


    def one_point_crossover(
        self, 
        chromosomes
    ):
        c1 = np.array(list(chromosomes[0]))
        c2 = np.array(list(chromosomes[1]))
        # c1 = copy.copy(chromosomes[0])
        # c2 = copy.copy(chromosomes[1])
        self.log_print(
            [
                "[Crossover Input]\n {} / {}".format(repr(c1), repr(c2))
            ]
        )
        # x = int(len(c1) / 2) # select the halfway point for the crossover
        x = random.randint(1, len(c1) - 2 ) # randomly select the position to perform the crossover at, excluding end points
        tmp = c2[:x].copy()
        c2[:x], c1[:x] = c1[:x], tmp
        self.log_print(
            [
                "[Crossover Result] (x={})\n {} / {}".format(x,repr(c1), repr(c2))
            ]
        )

        return c1, c2

    ######################
    # Mutation functions
    ######################

    def mutation(
        self,
        chromosomes,
    ):
        copy_chromosomes = copy.copy(chromosomes)
        mutated_chromosomes = []
        for c in copy_chromosomes:
            if np.all(c == 0):
                self.log_print(
                    [
                        "Input chomosome {} has no interactions -- forcing mutation".format(
                       c)
                    ]
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
            mutated_chromosomes.append(c)
        return mutated_chromosomes

    ######################
    # Elitism functions
    ######################

    def get_elite_models(
        self, 
        **kwargs
    ):
        r"""
        Wrapper for user-defined elite model selection.

        
        """

        return self.elite_ranking_top_two(
            **kwargs
        )

    def elite_ranking_top_two(
        self, 
        model_fitnesses,
        num_elites = 2,
        **kwargs
    ):
        ranked_models = sorted(
            model_fitnesses,
            key=model_fitnesses.get,
            reverse=True
        )
        elite_models = ranked_models[:num_elites]
        self.most_elite_models_by_generation[self.genetic_generation] = elite_models[0]
        num_elites_for_termination = 4

        if self.genetic_generation > self.unchanged_elite_num_generations_cutoff + 2:
            gen = self.genetic_generation
            recent_generations = list(
                range(
                    max(
                        0, 
                        gen - self.unchanged_elite_num_generations_cutoff
                    ), 
                    gen+1
                )
            )
            recent_elite_models = [
                self.most_elite_models_by_generation[g] for g in recent_generations
            ]
            unchanged = np.all( 
                np.array(recent_elite_models) == self.most_elite_models_by_generation[gen]
            )
            if unchanged:
                self.best_model_unchanged = True
            self.log_print(
                [
                    "Elite model unchanged in last {} generations: {}. \nCurrently: {} with f-score {}".format(
                        self.unchanged_elite_num_generations_cutoff, 
                        self.best_model_unchanged,
                        self.most_elite_models_by_generation[gen],
                        self.chromosome_f_score(
                            self.map_model_to_chromosome(
                                self.most_elite_models_by_generation[gen]
                            )
                        )
                    )
                ]
            )
        return elite_models

    ######################
    # Processing given fitness to 
    # selection probabilities
    ######################

    def get_selection_probabilities(
        self, 
        **kwargs
    ):
        r""" 
        Wrapper for user-defined probability processing function.

        Current iteration truncates and includes only top half of models
        """
        return self.truncate_to_top_half(**kwargs)


    def truncate_to_top_half(
        self, 
        model_fitnesses, 
        **kwargs
    ):
        ranked_models = sorted(
            model_fitnesses,
            key=model_fitnesses.get,
            reverse=True
        )
        num_models = len(ranked_models)
        truncation_cutoff = int(num_models/2)
        if num_models <= 4:
            truncated_model_list = ranked_models 
        else:
            self.log_print(
                [
                    "Truncating model to include only {} models".format(
                        truncation_cutoff
                    )
                ]
            )
            truncated_model_list = ranked_models[:truncation_cutoff]

        truncated_model_fitnesses = {
            mod : model_fitnesses[mod] 
            for mod in truncated_model_list
        }

        sum_fitnesses = np.sum(list(truncated_model_fitnesses.values()))
        self.log_print(
            [
                "Truncated model list:\n", truncated_model_list, 
                "\nTruncated model fitnesses:\n", truncated_model_fitnesses, 
                "\nsum fitnesses:", sum_fitnesses
            ]    
        )
        model_probabilities = {
            self.chromosome_string(self.map_model_to_chromosome(mod)) : (truncated_model_fitnesses[mod] / sum_fitnesses)
            for mod in truncated_model_list
        }
        self.log_print(
            [
                "Chromosome Selection probabilities:\n", model_probabilities
            ]
        )
        return model_probabilities


    ######################
    # Implement entire genetic algorithm iteration
    ######################

    def genetic_algorithm_step(
        self,
        model_fitnesses,
        **kwargs
    ):
        input_models = list(model_fitnesses.keys())
        num_models_for_next_generation = len(input_models)
        self.log_print(
            [
                "Num models reqd for generation:", num_models_for_next_generation
            ]
        )

        elite_models = self.get_elite_models(
            model_fitnesses = model_fitnesses,
            num_elites = 2
        )
        proposed_chromosomes = [
            self.chromosome_string(
                self.map_model_to_chromosome(
                    mod
                )
            ) 
            for mod in elite_models
        ] # list of chromosome strings to return

        chromosome_selection_probabilities = self.get_selection_probabilities(
            model_fitnesses = model_fitnesses
        )
        self.unique_pair_combinations_considered = []
        while len(proposed_chromosomes) < num_models_for_next_generation:
            selected_pair_chromosomes = self.selection(
                chromosome_selection_probabilities = chromosome_selection_probabilities
            )
            self.log_print(
                [
                    "Selected pair of chromosomes:", selected_pair_chromosomes
                ]
            )
            suggested_chromosomes = self.crossover(
                selected_pair_chromosomes
            )
            suggested_chromosomes = self.mutation(
                suggested_chromosomes
            )
            c0_str = self.chromosome_string( suggested_chromosomes[0] )
            c1_str = self.chromosome_string( suggested_chromosomes[1] )

            if (
                c0_str not in proposed_chromosomes
                and 
                c1_str not in proposed_chromosomes
                and
                c0_str != self.all_zero_chromosome_string 
                and
                c1_str != self.all_zero_chromosome_string
            ):
                proposed_chromosomes.append(c0_str)
                proposed_chromosomes.append(c1_str)
                self.log_print(
                    [
                        "num proposed chromosome now: {} of {}".format(
                            len(proposed_chromosomes),
                            num_models_for_next_generation
                        )
                    ]
                )
            else: 
                self.log_print(
                    [
                        "{} or {} already present in {}".format(c0_str, c1_str, proposed_chromosomes)
                    ]
                )

        # chop extra chromosomes if generated
        proposed_chromosomes = proposed_chromosomes[:num_models_for_next_generation]
        self.log_print(
            [
                "Proposed chromosome list now has {} elements.".format(
                    len(proposed_chromosomes)
                )
            ]
        )
        self.previously_considered_chromosomes.extend([
            self.chromosome_string(r) for r in proposed_chromosomes
            ]
        )
        self.genetic_generation += 1
        # self.delta_f_by_generation[self.genetic_generation] = delta_f_score
        self.chromosomes_at_generation[self.genetic_generation] = [
            self.chromosome_string(r) for r in proposed_chromosomes
        ]
        new_models = [
            self.map_chromosome_to_model(mod) 
            for mod in proposed_chromosomes
        ]
        self.log_print(
            [
                "Genetic alg num new models:{}".format(len(new_models)),
                "({} unique)".format(len(set(list(new_models))))
            ]
        )
        return new_models



