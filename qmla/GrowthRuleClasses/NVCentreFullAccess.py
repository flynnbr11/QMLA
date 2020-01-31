from __future__ import absolute_import
import sys
import os

from qmla.GrowthRuleClasses.SuperClassGrowthRule import growth_rule_super_class
from qmla import experiment_design_heuristics
from qmla import expectation_values
from qmla import database_framework

__all__ = [
    'nv_centre_spin_full_access'
]

class nv_centre_spin_full_access(
    # SuperClassGrowthRule.growth_rule_super_class
    growth_rule_super_class
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        # if self.use_experimental_data == True:
        #     import probe_set_generation
        #     # self.probe_generation_function = probe_set_generation.NV_centre_ising_probes_plus
        #     self.probe_generation_function = probe_set_generation.restore_dec_13_probe_generation

        # self.true_operator = 'xTz'
        self.true_operator = 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz'
        self.initial_models = ['xTi', 'yTi', 'zTi']
        self.qhl_models = [
            'xTiPPxTxPPxTyPPyTiPPyTyPPzTiPPzTz',
            'xTiPPxTxPPxTzPPyTiPPyTyPPzTiPPzTz',
            # 'xTiPPxTxPPxTyPPxTzPPyTiPPyTyPPyTzPPzTiPPzTz',
            'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',

            # 'zTi'
        ]
        self.heuristic_function = experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        self.max_num_parameter_estimate = 9
        self.max_spawn_depth = 8
        # self.max_num_qubits = 3
        self.tree_completed_initially = False
        self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.measurement_type = 'full_access'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z'  # e.g. transverse axis

        self.min_param = 0
        self.max_param = 10

        self.max_num_models_by_shape = {
            1: 0,
            2: 18,
            'other': 1
        }

        self.true_params = {
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
        log_file,
        **kwargs
    ):
        print("[Growth Rules] NV MODEL GENERATION")
        import random
        single_qubit_terms = ['xTi', 'yTi', 'zTi']
        nontransverse_terms = ['xTx', 'yTy', 'zTz']
        transverse_terms = ['xTy', 'xTz', 'yTz']
        all_two_qubit_terms = (single_qubit_terms + nontransverse_terms
                               + transverse_terms
                               )
        if len(model_list) > 1:
            log_print(["Only one model required for transverse Ising growth."],
                      log_file
                      )
            return False
        else:
            model = model_list[0]

        present_terms = model.split('PP')

        new_models = []
        if spawn_step in [1, 2]:
            for term in single_qubit_terms:
                if term not in present_terms:
                    new_model = model + 'PP' + term
                    new_models.append(new_model)
        elif spawn_step in [3, 4, 5]:
            for term in nontransverse_terms:
                if term not in present_terms:
                    new_model = model + 'PP' + term
                    new_models.append(new_model)

        elif spawn_step == 6:
            i = 0
            while i < 3:
                term = random.choice(transverse_terms)

                if term not in present_terms:
                    new_model = model + 'PP' + term
                    if (
                        database_framework.check_model_in_dict(
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
                    new_model = model + 'PP' + term
                    if (
                        database_framework.check_model_in_dict(
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
                    new_model = model + 'PP' + term
                    if (
                        database_framework.check_model_in_dict(
                            new_model, model_dict) == False
                        and new_model not in new_models
                    ):

                        new_models.append(new_model)
                        i += 1
        return new_models

    def latex_name(
        self,
        name
    ):
        if name == 'x' or name == 'y' or name == 'z':
            return '$' + name + '$'

        num_qubits = database_framework.get_num_qubits(name)
        terms = name.split('PP')
        rotations = ['xTi', 'yTi', 'zTi']
        hartree_fock = ['xTx', 'yTy', 'zTz']
        transverse = ['xTy', 'xTz', 'yTz', 'yTx', 'zTx', 'zTy']

        present_r = []
        present_hf = []
        present_t = []

        for t in terms:
            if t in rotations:
                present_r.append(t[0])
            elif t in hartree_fock:
                present_hf.append(t[0])
            elif t in transverse:
                string = t[0] + t[-1]
                present_t.append(string)
            # else:
            #     print("Term",t,"doesn't belong to rotations, Hartree-Fock or transverse.")
            #     print("Given name:", name)
        present_r.sort()
        present_hf.sort()
        present_t.sort()

        r_terms = ','.join(present_r)
        hf_terms = ','.join(present_hf)
        t_terms = ','.join(present_t)

        latex_term = ''
        if len(present_r) > 0:
            latex_term += r'\hat{S}_{' + r_terms + '}'
        if len(present_hf) > 0:
            latex_term += r'\hat{A}_{' + hf_terms + '}'
        if len(present_t) > 0:
            latex_term += r'\hat{T}_{' + t_terms + '}'

        final_term = '$' + latex_term + '$'
        if final_term != '$$':
            return final_term

        else:
            plus_string = ''
            for i in range(num_qubits):
                plus_string += 'P'
            individual_terms = name.split(plus_string)
            individual_terms = sorted(individual_terms)

            latex_term = '+'.join(individual_terms)
            final_term = '$' + latex_term + '$'
            return final_term
