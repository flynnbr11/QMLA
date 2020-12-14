from __future__ import absolute_import
import sys
import os
import random

from qmla.exploration_strategies.exploration_strategy import ExplorationStrategy
import qmla.shared_functionality.experiment_design_heuristics
from qmla.construct_models import alph, get_num_qubits

__all__ = [
    'ExperimentFullAccessNV'
]

class ExperimentFullAccessNV(
    ExplorationStrategy
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        # print("[Exploration Strategies] init nv_spin_experiment_full_tree")
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        # self.true_model = 'xTz'
        self.true_model = 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz'
        self.initial_models = ['xTi', 'yTi', 'zTi']
        self.qhl_models = [
            'xTiPPxTxPPxTyPPyTiPPyTyPPzTiPPzTz',
            'xTiPPxTxPPxTzPPyTiPPyTyPPzTiPPzTz',
            # 'xTiPPxTxPPxTyPPxTzPPyTiPPyTyPPyTzPPzTiPPzTz',
            'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
            # 'zTi'
        ]
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.nv_centre_SAT
        self.max_num_parameter_estimate = 9
        self.max_spawn_depth = 8
        self.prune_completed_initially = False
        # self.max_num_qubits = 3
        self.max_num_probe_qubits = 8
        self.tree_completed_initially = False
        # self.experimental_dataset = 'NVB_rescale_dataset.p'
        # self.measurement_type = 'full_access'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z'  # e.g. transverse axis

        self.min_param = 0
        self.max_param = 10

        self.max_num_models_by_shape = {
            1: 0,
            2: 18,
            'other': 1
        }

        self.true_model_terms_params = {
            # Decohering param set
            # From 3000exp/20000prt, BC SelectedRuns/Nov_28/15_14/results_049
            'xTi': -0.98288958683093952,  # -0.098288958683093952
            'xTx': 6.7232235286284681,  # 0.67232235286284681,
            'yTi': 6.4842202054983122,  # 0.64842202054983122, #
            'yTy': 2.7377867056770397,  # 0.27377867056770397,
            'zTi': 0.96477790489201143,  # 0.096477790489201143,
            'zTz': 1.6034234519563935,  # 0.16034234519563935,
        }

    def generate_models(
        self,
        model_list,
        spawn_step,
        model_dict,
        **kwargs
    ):
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
