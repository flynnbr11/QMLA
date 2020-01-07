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
import FermiHubbard

flatten = lambda l: [item for sublist in l for item in sublist]  # flatten list of lists



class fermi_hubbard_probabilistic(
    FermiHubbard.fermi_hubbard_predetermined
):
    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        self.true_operator = 'FHhop_1h2_down_d2+FHhop_1h2_up_d2+FHonsite_1_d2+FHonsite_2_d2' # for testing  
        # self.true_operator = 'FHhop_1h2_up_d2+FHonsite_1_d2+FHonsite_2_d2' # for testing  
        # self.true_operator = 'FHhop_1h2_down_d2+FHhop_1h2_up_d2' # for testing  
        # self.probe_generation_function = ProbeGeneration.fermi_hubbard_single_spin_n_sites
        # self.probe_generation_function = ProbeGeneration.fermi_hubbard_random_half_filled_superpositions
        # self.probe_generation_function = ProbeGeneration.fermi_hubbard_random_half_filled_bases
        # self.probe_generation_function = ProbeGeneration.singular_fill_bases
        self.probe_generation_function = ProbeGeneration.separable_fermi_hubbard_half_filled
        self.max_num_sites = 3
        self.max_num_probe_qubits = self.max_num_sites
        self.max_num_qubits = self.max_num_sites
        self.num_probes = 20
        self.lattice_dimension = 1
        self.tree_completed_initially = False
        self.num_top_models_to_build_on = 1
        self.model_generation_strictness = 0
        self.fitness_win_ratio_exponent = 1
        self.qhl_models = [
            'FHhop_1h2_down_d3+FHonsite_3_d3'
        ]

        self.true_params = {
            # term : true_param
            # 'FHhop_1h2_up_d2' : 1,
        }
        self.max_num_models_by_shape = {
            'other' : 4
        }

        self.setup_growth_class()

    def check_model_validity(
        self, 
        model, 
        **kwargs
    ):
        # possibility that some models not valid; not needed by default but checked for general case
        terms = DataBase.get_constituent_names_from_name(model)

        if np.all(['FHhop' in a for a in terms]):
            return  True
        elif np.all(['FHonsite' in a for a in terms]):
            # onsite present in all terms: discard
            # self.log_print(
            #     ["Rejecting model", model, "b/c all onsite terms"]
            # )
            return False
        else:
            hopping_sites = []
            number_term_sites = []
            chemical_sites = []
            num_sites = DataBase.get_num_qubits(model)
            for term in terms:
                constituents = term.split('_')
                constituents.remove('d{}'.format(num_sites))
                if 'FHhop' in term:
                    constituents.remove('FHhop')
                    for c in constituents:
                        if 'h' in c:
                            hopping_sites.extend(c.split('h'))
                elif 'FHonsite' in term:
                    constituents.remove('FHonsite')
                    number_term_sites.extend(constituents)
                elif 'FHchemical' in term:
                    constituents.remove('FHchemical')
                    chemical_sites.extend(constituents)

    #         print("hopping_sites:", hopping_sites)
    #         print('number term sites:', number_term_sites)
            hopping_sites = set(hopping_sites)
            number_term_sites = set(number_term_sites)
            overlap = number_term_sites.intersection(hopping_sites)

            if number_term_sites.issubset(hopping_sites):
                return True
            else:
                # no overlap between hopping sites and number term sites
                # so number term will be constant
                self.log_print(
                    [
                        "Rejecting model", model, 
                        "bc number terms present"
                        "which aren't present in kinetic term"
                    ]
                )
                return False

    def match_dimension(
        self,
        mod_name,
        num_sites, 
        **kwargs
    ):
        dimension_matched_name = match_dimension_hubbard(
            mod_name, 
            num_sites,
        )    
        return dimension_matched_name

    def generate_terms_from_new_site(
        self, 
        **kwargs 
    ):
        
        return generate_new_terms_hubbard(**kwargs)

    def combine_terms(
        self, 
        terms, 
    ):
        addition_string = '+'
        terms = sorted(terms)
        return addition_string.join(terms)


def generate_new_terms_hubbard( 
    connected_sites,
    num_sites,
    new_sites, 
    **kwargs     
):
    new_terms = []
    for pair in connected_sites:
        i = pair[0]
        j = pair[1]
        for spin in ['up', 'down']:
            hopping_term = "FHhop_{}h{}_{}_d{}".format(
                i, j, spin, num_sites 
            )
            new_terms.append(hopping_term)

    for site in new_sites:
        onsite_term = "FHonsite_{}_d{}".format(
            site, num_sites
        )
        
        new_terms.append(onsite_term)
        
    return new_terms

def match_dimension_hubbard(
    model_name, 
    num_sites,
    **kwargs
):
    redimensionalised_terms = []
    terms = model_name.split('+')
    for term in terms:
        parts = term.split('_')
        for part in parts:
            if part[0] == 'd' and part not in ['down', 'double']:
                parts.remove(part)

        parts.append("d{}".format(num_sites))
        new_term = "_".join(parts)
        redimensionalised_terms.append(new_term)
    new_model_name = "+".join(redimensionalised_terms)
    return new_model_name
    
    