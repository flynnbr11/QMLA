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
import NVCentreLargeSpinBath
import NVGrowByFitness
import SpinProbabilistic
import ConnectedLattice


class hopping_probabilistic(
    ConnectedLattice.connected_lattice
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
        
        self.lattice_dimension = 2
        self.initial_num_sites = 2
        self.lattice_connectivity_max_distance = 1
        self.lattice_connectivity_linear_only = True
        self.lattice_full_connectivity = True

        self.true_operator = 'h_1h2_d3PPPh_2h3_d3'
        self.true_operator = DataBase.alph(self.true_operator)
        self.qhl_models = [self.true_operator]
        self.base_terms = [
            'x', 
            'y', 
            'z'
        ]
        # fitness calculation parameters. fitness calculation inherited.
        self.num_top_models_to_build_on = 1 # 'all' # at each generation Badassness parameter
        self.model_generation_strictness = 0 #1 #-1 
        self.fitness_win_ratio_exponent = 3

        self.generation_DAG = 1
        self.max_num_sites = 4
        self.tree_completed_initially = False
        self.num_processes_to_parallelise_over = 10
        self.max_num_models_by_shape = {
            'other' : 10
        }

        self.min_param = 0
        self.max_param = 10

        self.setup_growth_class()

        
    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        individual_terms = DataBase.get_constituent_names_from_name(name)
        latex_term = ''

        hopping_terms = []
        interaction_energy = False
        for constituent in individual_terms:
            components = constituent.split('_')
            for term in components: 
                if term != 'h': #ie entire term not just 'h'
                    if 'h' in term: # ie a hopping term eg 1_1h2_d3, hopping sites 1-2, total num sites 3
                        split_term = term.split('h')
                        hopping_terms.append(split_term)

                    elif 'e' in term:
                        interaction_energy = True
                    elif 'd' in term:
                        dim = int(term.replace('d', ''))

        hopping_latex = 'H_{'
        for site_pair in hopping_terms:
            hopping_latex += str(
                '({},{})'.format(
                    str(site_pair[0]), 
                    str(site_pair[1])
                )
            )

        hopping_latex += '}'

        if hopping_latex != 'H_{}':
            latex_term += hopping_latex
            latex_term += str(
                "^{ \otimes"
                 + str(dim) + "}"
            )

        if interaction_energy is True:
            latex_term += str(
                '\sigma_{z}^{\otimes'
                +str(dim)
                +'}'
            )

        latex_term = str('$' + latex_term + '$')
        return latex_term


        
    def generate_terms_from_new_site(
        self,
        connected_sites, 
        base_terms, 
        num_sites,
    ):
        return generate_hopping_models_from_connected_sites(
            connected_sites = connected_sites,
            base_terms = base_terms, 
            num_sites = num_sites
        )
    
def generate_hopping_models_from_connected_sites(
    connected_sites, 
    base_terms, 
    num_sites
):
    
    new_terms = []
    
    for sites in connected_sites:
        site_1 = sites[0]
        site_2 = sites[1]
        
        new_term = "h_{}h{}_d{}".format(
            site_1, 
            site_2, 
            num_sites
        )
        new_terms.append(new_term)
    
    return new_terms
    


class hopping_predetermined(
    hopping_probabilistic
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
        self.true_operator = 'h_1h2_d4PPPPh_1h3_d4PPPPh_2h3_d4PPPPh_3h4_d4'
        self.tree_completed_initially = True
        if self.tree_completed_initially == True:
            # to manually fix the models to be considered
            self.initial_models = [
                'h_1h2_d2',
                'h_1h2_d3PPPh_1h3_d3',
                'h_1h2_d3PPPh_2h3_d3',
                'h_1h2_d3PPPh_1h3_d3PPPh_2h3_d3',
                'h_1h2_d4PPPPh_1h3_d4PPPPh_2h3_d4PPPPh_2h4_d4',
                'h_1h2_d4PPPPh_1h3_d4PPPPh_2h3_d4PPPPh_3h4_d4',
                'h_1h2_d4PPPPh_1h3_d4PPPPh_1h4_d4PPPPh_2h3_d4',
                'h_1h2_d4PPPPh_1h3_d4PPPPh_1h4_d4PPPPh_2h3_d4PPPPh_2h4_d4',
                'h_1h2_d4PPPPh_1h3_d4PPPPh_1h4_d4PPPPh_2h3_d4PPPPh_3h4_d4',
                'h_1h2_d4PPPPh_1h3_d4PPPPh_1h4_d4PPPPh_2h3_d4PPPPh_2h4_d4PPPPh_3h4_d4'
            ]

        