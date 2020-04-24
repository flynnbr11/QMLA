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
        self.log_print([
            "GR called. probe fnc:", self.probe_generation_function
        ])
        self.true_model = spin_system_model(
            num_sites = 1,
            core_terms = ['x'], 
            include_transverse_terms=False
        )
        self.tree_completed_initially = True
        self.initial_models=None
        self.expectation_value_function = \
            qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        self.model_heuristic_function = \
            qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic            
        self.log_print([
            "GR initialised"
        ])

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
        self.log_print(["Getting latex for ", name])
        latex_term = "${}$".format(name)
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
    
    
