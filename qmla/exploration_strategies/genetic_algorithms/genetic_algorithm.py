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

sys.path.append("/home/bf16951/QMD")
import qmla

import qmla.model_building_utilities

class GeneticAlgorithmQMLA():
    r"""
    Standalone genetic algorithm implementation for integration with :class:`qmla.QuantumModelLearningAgent`. 

    This class works with the :class:`~qmla.exploration_strategies.ExplorationStrategy`
    to construct models according to the genetic strategy. 

    :param list genes: individual terms which can be combined to form chromosomes
    :param int num_sites: maximum dimension permitted in model search
    :param str true_model: target model. if None, set at random from space of valid models.
    :param list base_terms: deprecated TODO remove
    :param str selection_method: mechanism through which to select chromosomes as parents. 
        Currently only 'roulette' available, but the framework should facilitate 
        alternatives. 
    :param str crossover_method : mechanism through which parent chromosomes are combined 
        to form offspring. 
        Currently only 'one_point' available, but the framework should facilitate 
        alternatives. 
    :param str mutation_method: mechanism through which to perform chromosome mutation
        Currently only 'element_wise' available, but the framework should facilitate 
        alternatives. 
    :param float mutation_probability: rate with which the mutation mechanism incurs mutation. 
    :param float selection_truncation_rate: fraction of models to retain as viable parents 
        to the subsequent generation; the lower-rated other models are discarded. 
    :param int num_protected_elite_models: number of models to automatically admit to the 
        subsequent generation. 
    :param int unchanged_elite_num_generations_cutoff: after this number of generations, 
        if the top model has not changed, the model search is terminated. 
    :param str log_file: path of QMLA instance's log file.

    """

    def __init__(
        self,
        genes,
        num_sites,
        true_model=None,
        base_terms=['x', 'y', 'z'],
        selection_method='roulette', 
        crossover_method='one_point',
        mutation_method='element_wise', 
        mutation_probability=0.1,
        selection_truncation_rate = 0.5, 
        num_protected_elite_models = 2, 
        unchanged_elite_num_generations_cutoff = 5,
        log_file=None, 
        **kwargs
    ):
        self.num_sites = num_sites
        self.base_terms = base_terms
        self.genes = list(sorted(genes))
        self.get_base_chromosome()
       
        if true_model is None: 
            r = random.randint(1, 2**self.num_terms-1)
            r = format(r, '0{}b'.format(self.num_terms))
            self.true_model = self.map_chromosome_to_model(r)
        else:
            self.true_model = true_model

        self.true_chromosome = self.map_model_to_chromosome(self.true_model)
        self.true_chromosome_string = self.chromosome_string(
            self.true_chromosome
        )
        self.all_zero_chromosome_string = '0'*self.num_terms
        self.addition_str = '+'
        self.mutation_probability = mutation_probability
        self.mutation_count = 0
        self.previously_considered_chromosomes = []
        self.chromosomes_at_generation = {}
        self.delta_f_by_generation = {}
        self.genetic_generation = 1
        self.log_file = log_file
        self.log_print([
            "Genes: {} \n Base chromosome: {} \n log file: {}".format(self.genes, self.basic_chromosome, self.log_file)
        ])
        self.f_score_change_by_generation = {}
        self.fitness_at_generation = {}
        self.models_ranked_by_fitness = {}
        self.most_elite_models_by_generation = {}
        self.num_protected_elite_models = num_protected_elite_models
        self.terminate_early_if_top_model_unchanged = True
        self.best_model_unchanged = False
        self.unchanged_elite_num_generations_cutoff = unchanged_elite_num_generations_cutoff
        self.selection_truncation_rate = selection_truncation_rate
        self.gene_pool = pd.DataFrame(columns=[
            'model', 'chromosome', 'f_score', 'probability', 'generation'
        ])
        self.elite_models = pd.DataFrame(columns=[
            'model', 'chromosome', 'f_score', 'generation', 'elite_position'
        ])

        # specifying which functionality to use
        self.selection_method = self.select_from_pair_df_remove_selected
        self.mutation_method = self.element_wise_mutation
        self.crossover_method = self.one_point_crossover
        
        available_selection_methods = {
            'roulette' : self.select_from_pair_df_remove_selected,
        }
        available_mutation_methods = {
            'element_wise' : self.element_wise_mutation
        }
        available_crossover_methods = {
            'one_point' : self.one_point_crossover
        }

        self.selection_method = available_selection_methods[selection_method]
        self.mutation_method = available_mutation_methods[mutation_method]
        self.crossover_method = available_crossover_methods[crossover_method]


        
    def get_base_chromosome(self):
        r"""
        Creates basic chromosome, i.e. with all genes set to 0. 
        """
        
        self.num_terms = len(self.genes)
        self.basic_chromosome = np.array([0]  * self.num_terms)        
        self.chromosome_description = self.genes
        self.chromosome_description_array = np.array(self.genes)

    def map_chromosome_to_model(
        self,
        chromosome,
    ):
        r"""
        Given a chromosome, get the corresponding model. 
        
        :param np.array chromosome: chromosome representing a candidate model
        :returns str model_string: name of the corresponding model
        """

        if isinstance(chromosome, str):
            chromosome = list(chromosome)
            chromosome = np.array([int(i) for i in chromosome])
        assert \
            len(chromosome) == self.num_terms, \
            "Chromosome must be of length {}".format(self.num_terms)
            
        nonzero_postions = chromosome.nonzero()
        present_terms = list(
            self.chromosome_description_array[nonzero_postions]
        )

        model_string = '+'.join(present_terms)
        return model_string
        
    def map_model_to_chromosome(
        self,
        model
    ):
        r"""
        Given a model, get the corresponding chromosome. 

        :param str model: name of candidate model
        :returns np.array chromosome: array of ones and zeros indicating which genes are active in the model
        """

        terms = qmla.model_building_utilities.get_constituent_names_from_name(model)
        assert \
            np.all([ t in self.chromosome_description for t in terms]), \
            "Cannot map some term(s) to any available gene. Terms: {} \n Genes".format(terms, self.chromosome_description)
            
        locs = [ self.chromosome_description.index(t) for t in terms]
        chromosome = copy.copy(self.basic_chromosome)
        chromosome[np.array(locs)] = 1
        return chromosome
           
    def model_f_score(
        self, 
        model_name
    ):
        r"""
        Get the F score of a candidate model. 

        :param str model_name: name of candidate model
        :returns float f_score: F score, between 0 and 1, indicating how many terms overlap 
            between the candidate and target models.
        """

        model_as_chromosome = self.map_model_to_chromosome(model_name)
        return self.chromosome_f_score(model_as_chromosome)

    def chromosome_string(
        self,
        c
    ):
        r"""Map a chromosome array to a string."""

        b = [str(i) for i in c]
        s = ''.join(b)
        if s == '1000000000':
            # TODO generaalise
            # 1 followed by num_terms 0's can be generated and is not permitted
            self.log_print([
                "Unallowed chromosome string {} for {}".format(b, c)
            ])
        return s

    def chromosome_f_score(
        self, 
        chromosome, 
    ):
        r"""
        Get the F score of a candidate model from its chromosome representation. 

        :param np.array chromosome: representation of candidate model
        :returns float f_score: F score, between 0 and 1, indicating how many terms overlap 
            between the candidate and target models.
        
        """

        if not isinstance(chromosome, np.ndarray):            
            chromosome = np.array([int(a) for a in list(chromosome)])
        
        return sklearn.metrics.f1_score(
            chromosome, 
            self.true_chromosome
        )

    def log_print(self, to_print_list):
        r"""Wrapper for :func:`~qmla.print_to_log`"""
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'GA gen {}'.format(self.genetic_generation)
        )


    def random_initial_models(
        self,
        num_models=5
    ):
        r"""
        Generate random models from the space of valid candidates. 

        :param int num_models: number of candidates to generate
        :returns list new_models: the randomly generated model names
        """

        if num_models > 2**self.num_terms:
            self.log_print([
                "Number of models requested > number of possible models ({})".format(
                    2**self.num_terms
                ),
                "Reducing by half until < half available"
            ])

            while num_models > (2**self.num_terms)/2:
                num_models = int(num_models/2)
        new_models = []
        self.initial_number_models = num_models
        self.chromosomes_at_generation[0] = []
        self.previously_considered_chromosomes = []
        self.birth_register = pd.DataFrame(
            columns=[
                'child', 'chromosome_child', 
                'parent_a', 'parent_b', 
                'chromosome_parent_a', 'chromosome_parent_b', 
                'generation', 'f_score'
            ]
        ) # TODO this is awful - this stuff shouldn't be initialised in this function

        while len(new_models) < num_models:
            # generate random number and 
            # format as binary string, i.e. chromosome
            r = random.randint(1, 2**self.num_terms-1)
            r = format(r, '0{}b'.format(self.num_terms)) 

            if (
                self.chromosome_string(r)
                not in self.previously_considered_chromosomes
            ):
                r = list(r)
                r = np.array([int(i) for i in r])
                mod = self.map_chromosome_to_model(r)
                chrom  = self.chromosome_string(r)
                f = self.chromosome_f_score(chrom)
                self.previously_considered_chromosomes.append(
                    chrom
                )
                self.chromosomes_at_generation[0].append(
                    chrom
                )
                new_models.append(mod)

                birth = pd.Series({
                    'child' : mod, 
                    'chromosome_child' : chrom, 
                    'generation' : 1, 
                    'f_score' : f,
                })
                self.birth_register.loc[len(self.birth_register)] = birth

        return new_models

    def rand_model_f(self):
        r"""
        Generate a random model chromosome and evaluate its F score. 
        """
        
        r = 0
        while r == 0 :
            r = np.random.randint(2**self.num_terms)
        
        b = bin(r)[2:].zfill(self.num_terms)
        b_array = np.array([int(i) for i in list(b)])
        f = sklearn.metrics.f1_score(
            b_array, 
            self.true_chromosome
        )
        return f, b_array

    def random_models_sorted_by_f_score(
        self,
        num_models=14, 
    ):
        r"""
        Generate a set of random models and sort them by F score. 
        """

        n_runs = 1e3 # first sample ~1000 random numbers 
        some_models = [
            self.rand_model_f() for _ in range(int(n_runs))
        ]
        f_scores = np.array(some_models)[:, 0]
        chromosomes =  np.array(some_models)[:,1]

        # then choose from those randomly generated models
        random_chroms = np.random.choice(chromosomes, num_models)
        random_models = [self.map_chromosome_to_model(c) for c in random_chroms]
        models_w_f = list(zip(random_models, [self.model_f_score(m) for m in random_models]))
        sorted_by_f = sorted(models_w_f, key = lambda x: x[1])
        sorted_models = np.array(sorted_by_f)[:, 0]
        sorted_models = list(sorted_models)
        just_f = np.array(models_w_f)[:, 1]
        just_f = [float(a) for a in just_f]

        return sorted_models        

    ######################
    # Selection functions
    ######################

    def selection(
        self,
        **kwargs
    ):
        r"""
        Wrapper for user's selected selection method. 

        Whatever method is called must return
            * prescribed_chromosomes
            * chromosomes_for_crossover - pairs
        """

        return self.selection_method(**kwargs)

    def select_from_pair_df_remove_selected(
        self,
        **kwargs
    ):
        # normalise so pairs' probabilities sum to 1
        self.chrom_pair_df.probability = self.chrom_pair_df.probability.astype(float)
        self.chrom_pair_df.probability = self.chrom_pair_df.probability / self.chrom_pair_df.probability.sum()
        pair_ids = list(self.chrom_pair_df.index)
        pair_probs = [ self.chrom_pair_df.loc[i].probability for i in pair_ids]
        self.log_print( ["Number available pairs:", len(pair_ids)] )

        # randomly select a pair from list of pairs
        selected_id = np.random.choice(
            a = pair_ids, 
            p = pair_probs
        )
        selected_entry = self.chrom_pair_df.loc[selected_id]
        # Drop so it can't be chosen again
        self.chrom_pair_df.drop(selected_id, inplace=True)
        self.log_print(["chrom pair df has {} options remaining".format(len(self.chrom_pair_df))])

        selection = {
            'chromosome_1' : selected_entry['c1'], 
            'chromosome_2' : selected_entry['c2'],
            'other_data' : { 
                'cut' : int(selected_entry['cut1']),
                'force_mutation' : bool(selected_entry['force_mutation'])
            }
        }
        return selection


    def basic_pair_selection(
        self,
        chromosome_selection_probabilities,
        **kwargs
    ):
        r"""
        Mechanism for selecting two models from the database of potential parents. 

        :param pd.DataFrame chromosome_selection_probabilities: 
            database indicating the probability that every valid pair of 
            parents should be selected. 
        :return tuple selected_chromosomes: two models
        """

        chromosomes = list(chromosome_selection_probabilities.keys())
        probabilities = [chromosome_selection_probabilities[c] for c in chromosomes]
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
        **kwargs
    ):
        r"""
        Wrapper for crossover mechanism. 

        This method assumes only 2 chromosomes to crossover
        and passes them to the method set as self.crossover_method, which can be easily replaced
        to facilitate alternative crossover schemes. 
        """

        return self.crossover_method(**kwargs)


    def one_point_crossover(
        self, 
        **kwargs
    ):
        r"""
        Crossover two chromosomes about a single gene. 

        Input two chromosomes, and selection (a dict) in kwargs. 
        selection contains ``chromosome_1`` and ``chromosome_2``,
        as well as a dict called  ``other_data`` containing ``cut``, 
        which is the position about which to crossover the two chromosomes. 
        """

        selection = kwargs['selection']
        c1 = np.array(list(selection['chromosome_1']))
        c2 = np.array(list(selection['chromosome_2']))
        x = selection['other_data']['cut']
        tmp = c2[:x].copy()
        c2[:x], c1[:x] = c1[:x], tmp

        return c1, c2

    ######################
    # Mutation functions
    ######################

    def mutation(
        self, 
        **kwargs
    ):
        r"""
        Wrapper for mutation mechanism. 
        All input arguments to the mutation method are passed directly to 
        the nominated mutation function, set as self.mutation_method.
        """

        return self.mutation_method(**kwargs)

    def element_wise_mutation(
        self,
        **kwargs
    ):
        r"""
        Probabilistically mutate each gene independently. 
        """
        
        chromosomes = kwargs['chromosomes']
        force_mutation = kwargs['force_mutation']

        copy_chromosomes = copy.copy(chromosomes)
        mutated_chromosomes = []
        for c in copy_chromosomes:
            try:
                if np.all(c == 0):
                    self.log_print([
                        "Input chomosome {} has no interactions -- forcing mutation".format(c)
                    ])
                    mutation_probability = 1.0
                else:
                    mutation_probability = self.mutation_probability
            except:
                self.log_print(["Can't compare all w/ 0 :", c])
                mutation_probability = self.mutation_probability

            if (
                np.random.rand() < mutation_probability
                or 
                force_mutation 
            ):
                num_mutations_to_perform = max(1, force_mutation)
                self.mutation_count += 1
                idx = np.random.choice(range(len(c)))
                # print("Flipping idx {}".format(idx))
                if int(c[idx]) == 0:
                    c[idx] = '1'
                elif int(c[idx]) == 1:
                    c[idx] = '0'
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
        Wrapper for elite model selection method, 
            here set to self.elite_ranking_top_n_models.        
        """

        return self.elite_ranking_top_n_models(
            **kwargs
        )


    def elite_ranking_top_n_models(
        self, 
        model_fitnesses,
        **kwargs
    ):        
        r"""
        Get the top N models, and store info on the elite models to date. 
        """

        elite_models = \
            self.models_ranked_by_fitness[self.genetic_generation][:self.num_protected_elite_models]
        self.log_print([
            "Elite models at generation {}: {}".format(
                self.genetic_generation, elite_models
            )
        ])
        for m in elite_models:
            self.elite_models = self.elite_models.append(
                pd.Series({
                    'model' : m, 
                    'generation' : self.genetic_generation, 
                    'elite_position' : elite_models.index(m) + 1,
                    'chromosome' : self.map_model_to_chromosome(m),
                    'f_score' : self.model_f_score(m)                    
                }),
                ignore_index=True
            )
        self.most_elite_models_by_generation[self.genetic_generation] = \
            self.models_ranked_by_fitness[self.genetic_generation][0]

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
                np.array(recent_elite_models) 
                == self.most_elite_models_by_generation[gen]
            )
            if unchanged and self.terminate_early_if_top_model_unchanged:
                # TODO this allows for unusual case where top model unchanged in 5 generations, 
                # but is improved upon in the subsequent generation.
                # but since 5 generations are unchanged, termination is triggered and the new generation champion is winner
                self.best_model_unchanged = True
                self.log_print([
                    "Setting best_model_unchanged to {}".format(self.best_model_unchanged)
                ])
            self.log_print([
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
            ])
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
        Wrapper for parent selection function, here set to self.truncate_to_top_half.
        """
        return self.truncate_to_top_half(**kwargs)


    def truncate_to_top_half(
        self, 
        model_fitnesses, 
        **kwargs
    ):
        r"""
        Retain only the top-performing half of models considered at this generation, 
        for consideration as parents to offspring on the subsequent generation. 

        """

        ranked_models = sorted(
            model_fitnesses,
            key=model_fitnesses.get,
            reverse=True
        )
        num_models = len(ranked_models)
        self.log_print([
            "Considering truncation for {} models. Truncation rate = {}".format(
                num_models,
                self.selection_truncation_rate
            ),
        ])
        for m in ranked_models: 
            self.log_print(["fitness = {} \t Model={} ".format(model_fitnesses[m], m )])
        
        truncation_cutoff = max( int(num_models*self.selection_truncation_rate), 4) # either consider top half, or top 4 if too small
        truncation_cutoff = min( truncation_cutoff, num_models )
        truncated_model_list = ranked_models[:truncation_cutoff]

        truncated_model_fitnesses = {
            mod : model_fitnesses[mod] 
            for mod in truncated_model_list
        }

        # keep the others with zero fitness, so the gene pool reflect them
        for m in ranked_models[truncation_cutoff:]:
            self.log_print([
                "Setting fitness to 0 for {} as it is {}th in rankings".format(
                    m, ranked_models.index(m)
                )
            ])
            truncated_model_fitnesses[m] = 0 

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
            for mod in truncated_model_fitnesses.keys()
        }
        self.log_print([
                "Chromosome Selection probabilities:\n", model_probabilities
        ])
        return model_probabilities


    def prepare_chromosome_pair_dataframe(
        self, 
        chromosome_probabilities,
        force_mutation=False,
    ):
        r"""
        Given a set of individual chromosome fitnesses, generate database of pairs of 
        parent chromosomes, with probability proportional to the fitness of both parents. 
        
        """

        self.log_print([
            "Setting up chromosome pair dataframe with initial probabilities", 
            chromosome_probabilities
        ])
        if len(chromosome_probabilities) == 1:
            self.log_print([
                "There is only one chromosome; not constructing selection database."
            ])
            return

        # Register gene pool
        for c in chromosome_probabilities:
            model = self.map_chromosome_to_model(c)
            gene_probability = pd.Series({
                'model' : model, 
                'chromosome' : c, 
                'f_score' : self.model_f_score(model), 
                'probability' : chromosome_probabilities[c],
                'generation' : self.genetic_generation,
            })
            self.gene_pool.loc[len(self.gene_pool)] = gene_probability

        # Construct df of pairs of chromosomes from the gene pool, where the probability of that 
        # pair being selected is the product of their individual fitnesses
        t2 = time.time()
        chromosome_combinations = list(
            itertools.combinations(list(chromosome_probabilities.keys()), 2)
        )
        eg_combo = chromosome_combinations[0]
        min_cut_pt = int(len(eg_combo[0])*0.25)
        max_cut_pt = int(len(eg_combo[0])*0.75) + 1
        self.log_print([
            "example chrom combination : {}. \n min/max cut locations = {}/{}".format(
                eg_combo, min_cut_pt, max_cut_pt
            )
        ])

        pair_data = []
        count_good_pairs = 0 
        for c1,c2 in chromosome_combinations:
            pair_prob = chromosome_probabilities[c1] * chromosome_probabilities[c2] # TODO better way to get pair prob?
            # for cut1 in range(1, len(c1)-2):
            if pair_prob > 0:
                count_good_pairs += 1
                self.log_print(["Nonzero prob pair: {} & {}, prob = {}".format(c1, c2, pair_prob)])
                for cut1 in range(min_cut_pt, max_cut_pt):
                    this_pair_df = {
                        'c1' : c1, 
                        'c2' : c2, 
                        'probability' : pair_prob, # np.round(pair_prob, 2), 
                        'cut1' : cut1, 
                        'c1_prob' : chromosome_probabilities[c1], 
                        'c2_prob' : chromosome_probabilities[c2],
                        'force_mutation' : force_mutation
                    }
                    pair_data.append(this_pair_df)
        self.chrom_pair_df = pd.DataFrame.from_dict(pair_data)    

        # normalise probabilities
        try:
            self.chrom_pair_df.probability = self.chrom_pair_df.probability.astype(float)
            self.chrom_pair_df.probability = self.chrom_pair_df.probability / self.chrom_pair_df.probability.sum()
        except:
            self.log_print(["Failing at final generation. chrom pair df:", self.chrom_pair_df])

        self.log_print([
            "starting chromosome pair dataframe setup. {} combinations in total from {} non-zero prob pairs. took {} sec and has len {}".format(
                len(chromosome_combinations),
                count_good_pairs, 
                np.round(time.time() - t2, 3),
                len(self.chrom_pair_df)
            )
        ])
        self.log_print([
            "Probs after preparing df:", 
            self.chrom_pair_df[
                ["c1", "c2", "probability"]
            ]
        ])

    def get_pair_selection_order(self):
        r"""
        Use the probabilities of parental selection to define the order in which to generate offspring. 
        It is cheaper to perform this once than call the database repeatedly. 

        :return list pair_selection_order: list of tuples of the order in which to pass 
            the model pairs to the crossover mechanism to generate offspring
        """

        pair_idx = self.chrom_pair_df.index.values
        probabilities = self.chrom_pair_df.probability.values
        # only keep nonzero probs
        pair_idx = pair_idx[probabilities > 0]
        probabilities = probabilities[probabilities > 0] 
        self.log_print([
            "get_pair_selection_order probabilities: ", probabilities, 
            "\n {} distinct".format(len(probabilities)), 
            "\n sum:", np.sum(probabilities)
        ])
        probabilities /= np.sum(probabilities)

        n_samples = len(probabilities)
        self.log_print(["Getting {} samples from chromosome probabilities".format(n_samples)])
        t1 = time.time()
        pair_selection_order = np.random.choice(
            a = pair_idx,
            size = n_samples, 
            p = probabilities,
            replace=False
        )
        self.log_print([
            "after {} s, pair_selection_order has {} elements ({} unique): \n {}".format(
                np.round(time.time() - t1, 3), 
                len(pair_selection_order), 
                len(set(pair_selection_order)),
                repr(pair_selection_order)
            ) 
        ])
        return pair_selection_order

    ######################
    # Implement entire genetic algorithm iteration
    ######################

    def consolidate_generation(
        self, 
        model_fitnesses, 
        **kwargs
    ):
        r"""
        Following the training of all models on a generation, consolidate that generation. 

        This involves determining the strongest models from the generation, 
        and constructing the database of parent-pairs and their associated selection probabilities. 
        """

        self.fitness_at_generation[self.genetic_generation] = model_fitnesses
        self.models_ranked_by_fitness[self.genetic_generation] = sorted(
            model_fitnesses,
            key=model_fitnesses.get,
            reverse=True
        )
        self.log_print([
            "GA step. model ranked by fitness:", self.models_ranked_by_fitness[self.genetic_generation]
        ])

        self.get_elite_models(
            model_fitnesses = model_fitnesses,
            num_protected_elite_models = 2
        )

        self.chromosome_selection_probabilities = self.get_selection_probabilities(
            model_fitnesses = model_fitnesses,
        )
        t_init = time.time()
        self.prepare_chromosome_pair_dataframe(
            chromosome_probabilities=self.chromosome_selection_probabilities
        )

    def genetic_algorithm_step(
        self,
        model_fitnesses,
        **kwargs
    ):
        r"""
        Perform a complete step of the genetic algorithm, assuming all of the required steps have been performed. 
        That is, the database for parent selection must already be available. 

        :param dict model_fitnesses: the fitness of each model in this generation according to the 
            chosen objective function. 
        :returns list new_models: set of models to place on the next generation. 
        """

        # get the order to iterate through chromosome pairs
        self.log_print(["Genetic algorithm step {}".format(self.genetic_generation)])
        pair_selection_order = self.get_pair_selection_order()
        init_num_chrom_pairs = len(pair_selection_order)
        pair_selection_order = iter(pair_selection_order)

        elite_models = list(self.elite_models[
            self.elite_models.generation == self.genetic_generation
        ].model)
        self.log_print([
            "elite models to start off with:", elite_models
        ])
        proposed_chromosomes = [
            self.chromosome_string(
                self.map_model_to_chromosome(mod)
            ) for mod in elite_models
        ] # list of chromosome strings to return

        input_models = list(model_fitnesses.keys())
        num_models_for_next_generation = len(input_models)
        self.log_print([
            "Num models reqd for generation:", num_models_for_next_generation
        ])

        num_loops_to_find_new_chromosome = 0
        force_mutation = False
        num_genes_to_force_mutate = 0
        t_init = time.time()
        while len(proposed_chromosomes) < num_models_for_next_generation:
            # selection = self.selection()
            try:
                selected_id = next(pair_selection_order)
            except:
                self.log_print([
                    "no pairs remaining." #  TODO now what?
                ])
                raise
            selected_entry = self.chrom_pair_df.loc[selected_id]
            selection = {
                'chromosome_1' : selected_entry['c1'], 
                'chromosome_2' : selected_entry['c2'],
                'other_data' : { 
                    'cut' : int(selected_entry['cut1']),
                    'force_mutation' : bool(selected_entry['force_mutation'])
                }
            }

            suggested_chromosomes = self.crossover(
                selection = selection
            )
            suggested_chromosomes = self.mutation(
                chromosomes = suggested_chromosomes,
                force_mutation=selection['other_data']['force_mutation']
            )
            c0_str = self.chromosome_string( suggested_chromosomes[0] )
            c1_str = self.chromosome_string( suggested_chromosomes[1] )

            for c in [c0_str, c1_str]:
                if (c not in proposed_chromosomes and c != self.all_zero_chromosome_string):
                    proposed_chromosomes.append(c)
                    self.log_print([
                        "num proposed chromosome now: {} of {}".format(
                            len(proposed_chromosomes),
                            num_models_for_next_generation
                        ),
                        "new chromosome:", c
                    ])
                    birth = pd.Series({
                        'child' : self.map_chromosome_to_model(c), 
                        'chromosome_child' : c, 
                        'chromosome_parent_a' : selection['chromosome_1'], 
                        'chromosome_parent_b' : selection['chromosome_2'], 
                        'parent_a' : self.map_chromosome_to_model( selection['chromosome_1']),
                        'parent_b' : self.map_chromosome_to_model( selection['chromosome_2']),
                        'generation' : self.genetic_generation,
                        'f_score' : self.chromosome_f_score(c)
                    })
                    self.birth_register.loc[len(self.birth_register)] = birth
                    self.log_print([
                        "Registering birth"
                    ])

            if len(self.chrom_pair_df) == 0 :
                # already tried every available pair 
                num_genes_to_force_mutate += 1 # TODO increase number of genes to flip to diversify population when repetitive
                self.log_print([
                    "Redrawing chromosome pair selection dataframe, enforcing mutation on {} genes".format(num_genes_to_force_mutate)
                ])
                self.prepare_chromosome_pair_dataframe(
                    chromosome_probabilities=self.chromosome_selection_probabilities,
                    force_mutation=True
                    # force_mutation=num_genes_to_force_mutate
                )

        # chop extra chromosomes if generated
        proposed_chromosomes = proposed_chromosomes[:num_models_for_next_generation]
        self.previously_considered_chromosomes.extend([
            self.chromosome_string(r) for r in proposed_chromosomes
            ]
        )
        
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

        self.genetic_generation += 1
        return new_models


class GeneticAlgorithmFullyConnectedLikewisePauliTerms(GeneticAlgorithmQMLA):
    r"""
    Exact structure of :class:`~qmla.GeneticAlgorithmQMLA`, where the avaiable terms 
    are assumed to follow conventional pauliSet format,
    and all sites are connected. 
    e.g. terms of the form 
    pauliSet_1J2_xJx_d2, pauliSet_1J2_yJy_d2, pauliSet_1J2_zJz_d2,

    :param int num_sites: dimension to permit model search within
    :param list base_terms: terms to use with pauliSet-type terms 
    """ 
    def __init__(self, num_sites, base_terms=['x', 'y', 'z'], **kwargs):
                
        terms = []
        for i in range(1, 1 + num_sites):
            for j in range(i + 1, 1 + num_sites):
                for t in base_terms:
                    new_term = 'pauliSet_{i}J{j}_{o}J{o}_d{N}'.format(
                        i= i, j=j, o=t, N=num_sites, 
                    )
                    terms.append(new_term)


        super().__init__(
            genes = terms, 
            num_sites = num_sites, 
            **kwargs
        )
        
def multidimensional_shifting(num_samples, sample_size, elements, probabilities):
    # replicate probabilities as many times as `num_samples`
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]
