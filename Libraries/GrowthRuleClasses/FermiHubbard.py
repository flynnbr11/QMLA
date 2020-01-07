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
        # self.true_operator = 'hop_1h2_down_d2+hop_1h2_up_d2+hop_1_double_d2+hop_2_double_d2' 
        self.true_operator = 'FHhop_1h2_up_d2'
        self.tree_completed_initially = True
        self.min_param = 0
        self.max_param = 1
        self.initial_models = [
            self.true_operator
        ]
        self.probe_generation_function = ProbeGeneration.separable_fermi_hubbard_half_filled
        self.simulator_probe_generation_function = self.probe_generation_function # unless specifically different set of probes required
        self.shared_probes = True # i.e. system and simulator get same probes for learning
        self.plot_probe_generation_function = ProbeGeneration.fermi_hubbard_half_filled_superposition
        # self.plot_probe_generation_function = ProbeGeneration.fermi_hubbard_single_spin_n_sites


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

        number_counting_terms = []
        hopping_terms = []
        chemical_terms = []
        terms = name.split('+')
        for term in terms:
            constituents = term.split('_')
            for c in constituents:
                if c[0:2] == 'FH':
                    term_type = c[2:]
                    continue # do nothing - just registers what type of matrix to construct
                elif c in list(basis_vectors.keys()):
                    spin_type = c
                elif c[0] == 'd':
                    num_sites = int(c[1:])
                else:
                    sites = [str(s) for s in c.split('h')]        


            if term_type == 'onsite':
                term_latex = "\hat{{N}}_{{{}}}".format(sites[0])
                number_counting_terms.append(term_latex)
            elif term_type == 'hop':
                term_latex = '\hat{{H}}_{{{}}}^{{{}}}'.format(
                    ",".join(sites),  # subscript site indices
                    basis_latex[spin_type] # superscript which spin type
                )
                hopping_terms.append(term_latex)
            elif term_type == 'chemical':
                term_latex = "\hat{{C}}_{{{}}}".format(sites[0])
                chemical_terms.append(term_latex)

        
        latex_str = ""
        for term_latex in (
            sorted(hopping_terms) + sorted(number_counting_terms) + sorted(chemical_terms)
        ):
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
        self.true_operator = 'FHhop_1h2_up_d2' 
        self.tree_completed_initially = True
        self.initial_models = [
            'FHhop_1h2_up_d2',
        ]
        self.max_num_sites = 2
        if self.true_operator not in self.initial_models:
            self.initial_models.append(self.true_operator)



