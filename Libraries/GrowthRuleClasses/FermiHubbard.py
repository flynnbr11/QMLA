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


    def latex_name(
        self,
        name, 
        **kwargs
    ):  
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
