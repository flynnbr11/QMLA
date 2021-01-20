from __future__ import absolute_import
import sys
import os
import random

from qmla.exploration_strategies.exploration_strategy import ExplorationStrategy
import qmla.shared_functionality.experiment_design_heuristics
from qmla.construct_models import alph, get_num_qubits

__all__ = [
    'FullAccessNVCentre'
]

class FullAccessNVCentre(
    ExplorationStrategy
):
    r"""
    Exploration strategy for NV system described in experimental paper, 
    assuming full access to the state so the likelihood is based on 
    :math`\bra{++} e^{ -i\hat{H(\vec{x})} t } \ket{++}`. 

    This is the base class for results presented in the experimental paper, 
    namely Fig 2. 
    The same model generation strategy is used in each case (i), (ii), (iii):
    this ES is for (i) pure simulation. 

    """

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        # self.true_model = 'xTz'
        self.true_model = 'xTi+yTi+zTi+zTz'
        self.initial_models = ['xTi', 'yTi', 'zTi']
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.nv_centre_SAT
        self.max_num_parameter_estimate = 9
        self.max_spawn_depth = 8
        self.prune_completed_initially = False
        self.max_num_probe_qubits = 2
        self.tree_completed_initially = False
        self.fixed_axis_generator = False
        self.fixed_axis = 'z'  # e.g. transverse axis
        self.check_champion_reducibility = True
        self.learned_param_limit_for_negligibility = 0.05
        self.reduce_champ_bayes_factor_threshold = 1e1

        self.min_param = 0
        self.max_param = 10
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0 # the last half of models' training data are deterministally the same anyway
        self.fraction_particles_for_bf = 0.1 # testing whether reduced num particles for BF can work 


        if self.true_model == 'xTi+yTi+zTi+zTz':
            self.true_model_terms_params = {  # from Jul_05/16_40
                'xTi': 0.92450565,
                'yTi': 6.00664336,
                'zTi': 1.65998543,
                'zTz': 0.76546868,
            }
        else:
            self.true_model_terms_params = {
                # Decohering param set
                # From 3000exp/20000prt, BC SelectedRuns/Nov_28/15_14/results_049
                'xTi' : -0.98288958683093952,
                'xTx' : 6.7232235286284681,  
                'yTi' : 6.4842202054983122,  
                'yTy' : 2.7377867056770397,  
                'zTi' : 0.96477790489201143, 
                'zTz' : 1.6034234519563935, 
                'xTy' : 1.5,
                'xTz' : 7, 
                'yTz' : 2 
            }
        self.gaussian_prior_means_and_widths = {
            'xTi': [4.0, 1.5],
            'yTi': [4.0, 1.5],
            'zTi': [4.0, 1.5],
            'xTx': [4.0, 1.5],
            'yTy': [4.0, 1.5],
            'zTz': [4.0, 1.5],
            'xTy': [4.0, 1.5],
            'xTz': [4.0, 1.5],
            'yTz': [4.0, 1.5],
        }

        self.num_processes_to_parallelise_over = 4
        self.timing_insurance_factor = 0.5

        self.max_num_models_by_shape = {
            1: 0,
            2: 18,
            'other': 0
        }

    def generate_models(
        self,
        model_list,
        spawn_step,
        model_dict,
        **kwargs
    ):
        # TODO replace with newer greedy add mechanism (as in TieredGreedySearchNVCentre)
        # TODO then remove model_dict from qmla's call to next_layer
        self.log_print([
            "Generating. input model list:", model_list
        ])
        single_qubit_terms = ['xTi', 'yTi', 'zTi']
        nontransverse_terms = ['xTx', 'yTy', 'zTz']
        transverse_terms = ['xTy', 'xTz', 'yTz']
        all_two_qubit_terms = (single_qubit_terms + nontransverse_terms
                               + transverse_terms
                               )
        model = model_list[0]
        present_terms = model.split('+')
        p_str = '+'

        new_models = []
        if spawn_step in [1, 2]:
            for term in single_qubit_terms:
                if term not in present_terms:
                    new_model = model + p_str + term
                    new_models.append(new_model)
        elif spawn_step in [3, 4, 5]:
            for term in nontransverse_terms:
                if term not in present_terms:
                    new_model = model + p_str + term
                    new_models.append(new_model)

        elif spawn_step == 6:
            i = 0
            while i < 3:
                term = random.choice(transverse_terms)

                if term not in present_terms:
                    new_model = model + p_str + term
                    if (
                        check_model_in_dict(
                            new_model, model_dict) == False
                        and new_model not in new_models
                    ):

                        new_models.append(new_model)
                        i += 1
        elif spawn_step == 7:
            i = 0
            while i < 2:
                term = random.choice(transverse_terms)

                if term not in present_terms:
                    new_model = model + p_str + term
                    if (
                        check_model_in_dict(
                            new_model, model_dict) == False
                        and new_model not in new_models
                    ):

                        new_models.append(new_model)
                        i += 1

        elif spawn_step == 8:
            i = 0
            while i < 1:
                term = random.choice(transverse_terms)

                if term not in present_terms:
                    new_model = model + p_str + term
                    if (
                        check_model_in_dict(
                            new_model, model_dict) == False
                        and new_model not in new_models
                    ):

                        new_models.append(new_model)
                        i += 1
        return new_models



def check_model_in_dict(name, model_dict):
    """
    Check whether the new model, name, exists in all previously considered models,
        held in model_lists.
    [previously in construct_models]
    If name has not been previously considered, False is returned.
    """
    # Return true indicates it has not been considered and so can be added

    al_name = alph(name)
    n_qub = get_num_qubits(name)

    if al_name in model_dict[n_qub]:
        return True  # todo -- make clear if in legacy or running db
    else:
        return False
