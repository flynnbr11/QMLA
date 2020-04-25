import sys
import os
import itertools

from qmla.growth_rules.nv_centre_spin_characterisation import nv_centre_full_access
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.expectation_values
from qmla import database_framework


class SimulatedNVCentre(
    nv_centre_full_access.ExperimentFullAccessNV  # inherit from this
):
    # Uses some of the same functionality as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out
    # and different model generation

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.true_model = spin_system_model(
            num_sites = 4,
            # core_terms = ['x'], 
            include_transverse_terms=False
        )
        self.tree_completed_initially = True
        self.initial_models=None
        self.expectation_value_function = \
            qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        self.model_heuristic_function = \
            qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic            
        self.max_time_to_consider = 100
        self.true_model_terms_params = {
            'pauliSet_1_x_d4' : 8.2450565,
            'pauliSet_1_y_d4' : 12.00664336,
            'pauliSet_1_z_d4' : 5.65998543,
            'pauliSet_1J2_xJx_d4' : 0.15,
            'pauliSet_1J2_yJy_d4' : 0.015,
            'pauliSet_1J2_zJz_d4' : 0.25, #0.7654868,
            'pauliSet_1J3_xJx_d4' : 0.15,
            'pauliSet_1J3_yJy_d4' : 0.15, 
            'pauliSet_1J3_zJz_d4' : 0.15,
            'pauliSet_1J4_xJx_d4' : 0.15,
            'pauliSet_1J4_yJy_d4' : 0.15,
            'pauliSet_1J4_zJz_d4' : 0.15,
        }

    def generate_models(self, model_list, **kwargs):
        if self.spawn_stage[-1]==None:
            self.spawn_stage.append('Complete')
            return [self.true_model]
    
    def check_tree_completed(self, **kwargs):
        if self.spawn_stage[-1] == 'Complete':
            return True
        else:
            return False


    def latex_name(
        self,
        name,
        **kwargs
    ):
        # print("[latex name fnc] name:", name)
        core_operators = list(sorted(database_framework.core_operator_dict.keys()))
        num_sites = database_framework.get_num_qubits(name)
        p_str = '+'
        separate_terms = name.split(p_str)

        site_connections = {}
        # for c in list(itertools.combinations(list(range(num_sites + 1)), 2)):
        #     site_connections[c] = []

        # term_type_markers = ['pauliSet', 'transverse']
        transverse_axis = None
        ising_axis = None
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
                # sites = tuple([int(a) for a in sites])
                sites = ','.join([str(a) for a in sites])
                # assumes like-like pauli terms like xx, yy, zz
                op = operators[0]
                try:
                    site_connections[sites].append(op)
                except:
                    site_connections[sites] = [op]
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

        latex_term = "${}$".format(latex_term)
        return latex_term


def spin_system_model(
    num_sites = 2, 
    full_parameterisation=True, 
    include_transverse_terms = False,
    core_terms = ['x', 'y', 'z']
):
    spin_terms = [
        'pauliSet_1_{op}_d{N}'.format(op = op, N=num_sites)
        for op in core_terms
    ]
    hyperfine_terms = [
        'pauliSet_1J{k}_{op}J{op}_d{N}'.format(
            k = k, 
            op = op, 
            N = num_sites
        )
        for k in range(2, num_sites+1)
        for op in core_terms
    ]
    transverse_terms = [
        'pauliSet_1J{k}_{op1}J{op2}_d{N}'.format(
            k = k, 
            op1 = op1,
            op2 = op2,
            N = num_sites
        )
        for k in range(2, num_sites+1)
        for op1 in core_terms
        for op2 in core_terms
        if op1 < op2 # only xJy, not yJx 
    ]
    
    all_terms = []
    all_terms.extend(spin_terms)
    all_terms.extend(hyperfine_terms)
    if include_transverse_terms: all_terms.extend(transverse_terms)
    
    model = '+'.join(all_terms)
    model = qmla.database_framework.alph(model)
    print("Spin system used:", model)
    return model
    
    
