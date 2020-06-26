from __future__ import absolute_import
import sys
import os

from qmla.growth_rules.growth_rule import GrowthRule
import qmla.shared_functionality.experiment_design_heuristics
from qmla import construct_models

__all__ = [
    'NVCentreNQubitBath'
]

class NVCentreNQubitBath(
    GrowthRule
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):

        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        
        # Choose functions 
        self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution_double_time_reverse
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probe_generation_function = self.probe_generation_function
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic

        # True model configuration        
        self.true_model = 'pauliSet_1_x_d2+pauliSet_1_y_d2+pauliSet_1_z_d2+pauliSet_1J2_zJz_d2'
        self.true_model_terms_params = {
            'pauliSet_1_x_d2': 0.92450565,
            'pauliSet_1_y_d2': 6.00664336,
            'pauliSet_1_z_d2': 1.65998543,
            'pauliSet_1J2_zJz_d2': 0.76546868,
        }


        # QMLA and model learning configuration
        self.initial_models = None
        self.qhl_models =  ['pauliSet_1_x_d1', 'pauliSet_1_y_d1', 'pauliSet_1_z_d1']
        self.prune_completed_initially = True # pruning is implicit in the spawn rule

        self.min_param = 0
        self.max_param = 10
        self.num_probes = 1 # |++'>
        self.max_num_qubits = 3
        self.probe_maximum_number_qubits = 5
        self.include_transverse_terms = False

        self.fraction_opponents_experiments_for_bf = 0
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_particles_for_bf = 1

        # Logistics
        self.max_num_models_by_shape = {
            1 : 3,
            'other': 6
        }
        self.num_processes_to_parallelise_over = 6
        self.timing_insurance_factor = 1


    # Model generation / QMLA progression

    def generate_models(self, model_list, **kwargs):

        self.log_print(["Model list:", model_list])

        try:
            top_model = model_list[0]
            present_terms = top_model.split('+')
        except:
            top_model = ''
            present_terms = []

        one_qubit_terms = ['pauliSet_1_x_d1', 'pauliSet_1_y_d1', 'pauliSet_1_z_d1']
        if self.spawn_stage[-1] == None:
            # Start of tree - very first models
            new_models = one_qubit_terms
            self.log_print([
                "Start of GR. Models=", new_models
            ])
            self.spawn_stage.append('1_qubit')
        elif self.spawn_stage[-1] == '1_qubit':
            # 1 qubit: rotation terms on the spin
            unused_terms = set(one_qubit_terms) - set (present_terms)
            if len(unused_terms) == 0:
                # no new terms to greedily add -> finish 1 qubit section
                new_models = ['pauliSet_1_x_d1'] # champions of 1 qubit branches
                self.log_print(["No unused terms; GR complete."])
                self.spawn_stage.append('1_qubit_complete')
                self.spawn_stage.append('2_qubit_coupling')
            else:
                self.log_print([
                    "New terms: ", unused_terms
                ])
                new_models = [
                    "{}+{}".format(top_model, new_term)
                    for new_term in unused_terms
                ]
        elif "_qubit_coupling" in self.spawn_stage[-1]:
            
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_coupling'))
            coupling_terms = [ # coupling electron with this qubit
                'pauliSet_1J{q}_{p}J{p}_d{N}'.format(
                    q = num_qubits, 
                    p = pauli_term, 
                    N = num_qubits
                )
                for pauli_term in ['x', 'y', 'z']
            ]
            self.log_print(["Coupling terms:", coupling_terms])
            unused_terms = list(set(coupling_terms) - set(present_terms))

            if len(unused_terms) == 0:
                self.spawn_stage.append(
                    "{N}_qubits_coupling_complete".format(N=num_qubits)
                )
                
                if self.include_transverse_terms:
                    self.spawn_stage.append(
                        "{N}_qubits_transverse".format(N=num_qubits)
                    )
                elif num_qubits == self.max_num_qubits:
                    self.spawn_stage.append('Complete')
                
                new_models = ['pauliSet_1_x_d2']
                

                # TODO new_models = N qubit coupling branch champions
            else:
                new_models = [
                    "{}+{}".format(top_model, new_term)
                    for new_term in unused_terms
                ]
        elif '_qubits_transverse' in self.spawn_stage[-1]:
            num_qubits = int(self.spawn_stage[-1].strip('_qubits_transverse'))

            transverse_terms = [
                'pauliSet_1J{N}_xJy_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_xJz_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_yJz_d{N}'.format(N = num_qubits)
            ]
            unused_terms = list(set(transverse_terms) - set(present_terms))

            if len(unused_terms) == 0:
                self.spawn_stage.append(
                    "{N}_qubits_transverse_complete".format(N=num_qubits)
                )

                if num_qubits == self.max_num_qubits:
                    self.spawn_stage.append['Complete']
                else:
                    new_num_qubits = num_qubits + 1
                    self.spawn_stage.append(
                        "{N}_qubits_coupling".format( N = new_num_qubits)
                    )

                # TODO new_models = N qubit transverse branch champions
            else:
                new_models = [
                    "{}+{}".format(top_model, new_term)
                    for new_term in unused_terms
                ]

        return new_models

            
    def check_tree_completed(
        self,
        spawn_step,
        **kwargs
    ):
        if self.tree_completed_initially:
            return True
        
        if self.spawn_stage[-1] == 'Complete':
            return True
        
        return False

    def check_tree_pruned(self, prune_step, **kwargs):
        return True

