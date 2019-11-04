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

flatten = lambda l: [item for sublist in l for item in sublist]  # flatten list of lists


class fermi_hubbard(
    ConnectedLattice.connected_lattice
    # hopping_probabilistic
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
        self.true_operator = 'hop_1h2_down_d2+hop_1h2_up_d2+hop_1_double_d2+hop_2_double_d2' 
        # self.true_operator = 'h_1h2_d2'
        self.tree_completed_initially = True
        self.min_param = 0
        self.max_param = 1
        self.initial_models = [
            self.true_operator
        ]
        self.max_time_to_consider = 20
        self.num_processes_to_parallelise_over = 6
        self.max_num_models_by_shape = {
            1 : 0,
            2 : 0,
            4 : 10, 
            'other' : 0
        }

    def latex_name(
        self,
        name, 
        **kwargs
    ):  
        # TODO gather terms in list, sort alphabetically and combine for latex str
        basis_vectors = {
            'vac' : np.array([1,0,0,0]),
            'down' : np.array([0,1,0,0]),
            'up' : np.array([0,0,1,0]),
            'double' : np.array([0,0,0,1])
        }

        basis_latex = {
            'vac' : 'V',
            'up' : r'\uparrow',
            'down' : r'\downarrow',
            'double' : r'\updownarrow'
        }

        latex_str = ""
        terms = name.split('+')
        for term in terms:
            constituents = term.split('_')
            for c in constituents:
                if c == 'hop':
                    continue # do nothing - just registers what type of matrix to construct
                elif c in list(basis_vectors.keys()):
                    spin_type = c
                elif c[0] == 'd':
                    num_sites = int(c[1:])
                else:
                    sites = [str(s) for s in c.split('h')]        


            if spin_type == 'double':
                term_latex = "\hat{{N}}_{{{}}}".format(sites[0])
            else:
                term_latex = '\hat{{H}}_{{{}}}^{{{}}}'.format(
                    ",".join(sites),  # subscript site indices
                    basis_latex[spin_type] # superscript which spin type
                )
            latex_str += term_latex
        latex_str = "${}$".format(latex_str)
        return latex_str



class fermi_hubbard_predetermined(
    fermi_hubbard
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
        self.true_operator = 'hop_1h2_down_d2+hop_1h2_up_d2+hop_1_double_d2+hop_2_double_d2' 
        self.tree_completed_initially = True
        self.initial_models = [
            'hop_1h2_down_d2+hop_1_double_d2',
        ]
        self.max_num_sites = 2
        if self.true_operator not in self.initial_models:
            self.initial_models.append(self.true_operator)



class fermi_hubbard_probabilistic(
    fermi_hubbard_predetermined
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
        self.true_operator = 'hop_1h2_down_d2+hop_1h2_up_d2+hop_1_double_d2+hop_2_double_d2' 
        self.tree_completed_initially = False
        self.num_top_models_to_build_on = 1 # 'all'
        self.setup_growth_class()




    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):

        model_points = kwargs['branch_model_points']
        branch_models = list(model_points.keys())
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

        # self.generation_champs[self.generation_DAG][self.sub_generation_idx] = models_to_build_on
        self.generation_champs[self.generation_DAG][self.sub_generation_idx] = [
            kwargs['model_names_ids'][models_to_build_on[0]]
        ]

        self.counter+=1
        new_models = []

        if self.spawn_stage[-1] == None:
            # new models given by models_to_build_on plus terms in available_terms (greedy)
            
            for mod_id in models_to_build_on:
                mod_name = kwargs['model_names_ids'][mod_id]
                present_terms = DataBase.get_constituent_names_from_name(mod_name)
                print("[fermi hubbard] model {} has present terms {}".format(mod_name, present_terms))
                available_terms = list(
                    set(self.available_mods_by_generation[self.generation_DAG])
                    - set(present_terms)
                )

                if len(available_terms) == 0:
                    # this dimension exhausted; return dimension branch champs
                    self.spawn_stage.append('make_new_generation')
                    new_models = [
                        self.generation_champs[self.generation_DAG][k] for k in 
                        list(self.generation_champs[self.generation_DAG].keys())
                    ]
                    new_models = flatten(new_models)
                    new_models = [new_models[0]] # hack to force non-crash for single generation
                    self.log_print(
                        [
                            "No remaining available terms. Completing generation",
                            "generation:", self.generation_DAG, 
                            "Models:", new_models
                        ]
                    )
                    if self.generation_DAG == self.max_num_generations:
                        # this was the final generation to learn.
                        # instead of building new generation, skip straight to Complete stage
                            
                        self.log_print(
                            [
                                "Completing growth rule"
                            ]
                        )

                        self.spawn_stage.append('Complete')

                    return new_models

                # if some terms are still availble (i.e. len(avail_terms)!=0)
                for term in available_terms:
                    new_mod = "{}+{}".format(
                        mod_name, 
                        term    
                    )
                    # new_mod = DataBase.alph(new_mod) # TODO fix DataBase.alph to take + like terms
                    new_models.append(new_mod)

        return new_models




    def generate_terms_from_new_site(
        self, 
        **kwargs 
    ):
        
        return generate_new_terms_hubbard(**kwargs)

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
            hopping_term = "hop_{}h{}_{}_d{}".format(
                i, j, spin, num_sites 
            )
            new_terms.append(hopping_term)

    for site in new_sites:
        onsite_term = "hop_{}_double_d{}".format(
            site, num_sites
        )
        
        new_terms.append(onsite_term)
        
    return new_terms
            