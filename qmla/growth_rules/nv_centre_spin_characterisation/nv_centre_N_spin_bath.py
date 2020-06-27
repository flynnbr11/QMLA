from __future__ import absolute_import
import sys
import os

from qmla.growth_rules.growth_rule import GrowthRule
import qmla.shared_functionality.experiment_design_heuristics
import qmla.utilities
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
        self.true_model = 'pauliSet_1_x_d2+pauliSet_1_z_d2+pauliSet_2_y_d2+pauliSet_1J2_zJz_d2'
        self.true_model_terms_params = {
            'pauliSet_1_x_d2': 0.92450565,
            'pauliSet_1_y_d2': 6.00664336,
            'pauliSet_1_z_d2': 1.65998543,
            'pauliSet_2_y_d2' : 2, 
            'pauliSet_1J2_zJz_d2': 0.76546868,
        }


        # QMLA and model learning configuration
        self.initial_models = None
        self.qhl_models =  ['pauliSet_1_x_d1', 'pauliSet_1_y_d1', 'pauliSet_1_z_d1']
        self.prune_completed_initially = True # pruning is implicit in the spawn rule

        self.min_param = 0
        self.max_param = 10
        self.num_probes = 1 # |++'>
        self.champion_models_by_spawn_stage = {}

        non_spin_qubit_contributions = ['rotation', 'coupling']
        self.stages_by_num_qubits = {
            1 : iter(['rotation']),
            2 : iter( non_spin_qubit_contributions ),
            3 : iter( non_spin_qubit_contributions ),            
        }

        self.max_num_qubits = int(max( self.stages_by_num_qubits ))
        self.all_layer_champions_by_num_qubits = {
            i : []
            for i in range(1, self.max_num_qubits+1)
        }
        self.layer_champions = {
            num_qubits :  {}
            for num_qubits in range(1, self.max_num_qubits+1)
        }
        self.num_qubits_champion = {}
        self.stage_champions = {
            i : {} for i in range(1, self.max_num_qubits+1)
        }
        self.current_stage = None
        self.spawn_stage = ['stage_complete']
        
        self.probe_maximum_number_qubits = 5
        self.include_transverse_terms = True

        self.fraction_opponents_experiments_for_bf = 0.5
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_particles_for_bf = 0.5

        # Logistics
        self.max_num_models_by_shape = {
            1 : 3,
            'other': 6
        }
        self.num_processes_to_parallelise_over = 6
        self.timing_insurance_factor = 1


    # Model generation / QMLA progression

    def generate_models(self, model_list, **kwargs):

        self.log_print([
            "Generating models. \nModel list={} \nSpawn stage:{}\n Current stage={}".format(model_list, self.spawn_stage[-1], self.current_stage)
        ])

        try:
            top_model = model_list[0]
            num_qubits = qmla.construct_models.get_num_qubits(top_model)
            present_terms = top_model.split('+')
        except:
            top_model = None
            num_qubits = 1
            present_terms = []

        # Move to next major stage if finished this one
        if '_qubit_champion_selection' in self.spawn_stage[-1]:
            # Record the previous round champion as that qubit-number champ
            # increase num qubits for this round
            self.log_print([""])
            self.num_qubits_champion[num_qubits] = top_model
            self.stage_champions[num_qubits]['overall'] = top_model
            
            if self.max_num_qubits == num_qubits:
                self.log_print(["Spawning complete"])
                self.spawn_stage.append('Complete')

                new_models = [
                    self.stage_champions[num_qubits]['overall']
                    for num_qubits in self.stage_champions
                ] # all champions by num qubits
                return new_models
            else:              
                num_qubits += 1
                self.spawn_stage.append('stage_complete')
        else:
            # record previous round champion as a layer champion 
            # within this number of qubits
            # and within this stage
            if top_model is not None:
                self.all_layer_champions_by_num_qubits[num_qubits].append(top_model)
                try:
                    self.layer_champions[num_qubits][self.current_stage].append(top_model)
                except:
                    self.layer_champions[num_qubits][self.current_stage] = [top_model]


        if self.spawn_stage[-1] == 'stage_complete':
            self.log_print([
                "Stage complete. num qubits={} current stage={}".format(
                    num_qubits, 
                    self.current_stage
                )
            ])
            self.stage_champions[num_qubits][self.current_stage] = top_model
            self.log_print([
                "Stage champion[{}][{}] : {}".format(
                    num_qubits, self.current_stage, 
                    self.stage_champions[num_qubits][self.current_stage]
                )
            ])

            try:
                # Check if a new stage for this number of qubits is available
                self.current_stage = next(self.stages_by_num_qubits[num_qubits])
                new_spawn_stage = "{N}_qubit_{S}".format(N=num_qubits, S=self.current_stage)
                self.spawn_stage.append(new_spawn_stage)
                self.log_print(["New spawn stage:", new_spawn_stage])

            except:
                # No stages left for this number of qubits
                # -> get layer champions from this stage
                new_models = self.layer_champions[num_qubits][self.current_stage]
                self.log_print([
                    "Setting new models as selecting champion for {} qubits: {}".format(
                        num_qubits, new_models
                    )
                ])
                self.spawn_stage.append(
                    "{}_qubit_champion_selection".format(num_qubits)
                )
                return new_models

        # Greedily add terms from batch of available terms, determined by stage of GR
        if 'rotation' in self.spawn_stage[-1]:
            self.log_print(["Available: rotation terms"])
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_rotation'))
            available_terms = [
                'pauliSet_{N}_{p}_d{N}'.format(p=pauli_term, N=num_qubits)
                for pauli_term in 
                ['x', 'y', 'z']
            ]
        elif 'coupling' in self.spawn_stage[-1]:
            self.log_print(["Available: coupling terms"])
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_coupling'))
            available_terms = [ # coupling electron with this qubit
                'pauliSet_1J{q}_{p}J{p}_d{N}'.format(
                    q = num_qubits, 
                    p = pauli_term, 
                    N = num_qubits
                )
                for pauli_term in ['x', 'y', 'z']
            ]
        elif 'transverse' in self.spawn_stage[-1]:
            self.log_print(["Available: transverse terms"])
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_transverse'))
            available_terms =  [
                'pauliSet_1J{N}_xJy_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_xJz_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_yJz_d{N}'.format(N = num_qubits)
            ]

        # Select new models
        unused_terms = list(
            set(available_terms) - set(present_terms)
        )
        self.log_print(["Available terms:", unused_terms])
        new_models = []
        for term in unused_terms:
            new_model_terms = list(
                set(present_terms + [term] )
            )
            new_model = '+'.join(new_model_terms)
            new_models.append(new_model)

        if len(new_models) <= 1:
            # Greedy addition of terms exhausted
            # -> move to next stage
            self.log_print(["Few new models - completing stage"])
            self.spawn_stage.append('stage_complete')
        
        self.log_print(["Designed new models:", new_models])

        new_models = [
            qmla.utilities.ensure_consisten_num_qubits_pauli_set(
                model
            ) for model in new_models
        ]
        return new_models



    def alt_generate_models(self, model_list, **kwargs):

        self.log_print([
            "Generating models. \nModel list={} \nSpawn stage:{}\n kwargs:{}".format(model_list, self.spawn_stage[-1], kwargs)
        ])

        try:
            top_model = model_list[0]
            num_qubits = qmla.construct_models.get_num_qubits(top_model)
            present_terms = top_model.split('+')
        except:
            top_model = None
            present_terms = []

        if self.spawn_stage[-1] is None:
            self.spawn_stage.append('1_qubit_rotation')

        if '_qubit_champion_selection' in self.spawn_stage[-1]:
            self.num_qubits_champion[num_qubits] = top_model
            
            # Move to next spawn stage
            if num_qubits == self.max_num_qubits:
                self.log_print(["Spawning complete"])
                self.spawn_stage.append('Complete')

                new_models = list(self.num_qubits_champion.values()) # all champions by num qubits
                return new_models
            else:
                new_num_qubits = num_qubits + 1
                new_spawn_stage = "{N}_qubit_coupling".format(N=new_num_qubits)
                self.spawn_stage.append(new_spawn_stage)
        else:
            if top_model is not None:
                self.all_layer_champions_by_num_qubits[num_qubits].append(top_model)


        # Process spawn stages
        if '_qubit_complete' in self.spawn_stage[-1]:
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_complete'))

        elif 'qubit_rotation' in self.spawn_stage[-1]:
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_rotation'))
            available_terms = [
                'pauliSet_1_{p}_d{N}'.format(p=pauli_term, N=num_qubits)
                for pauli_term in ['x', 'y', 'z']
            ]
        elif 'coupling' in self.spawn_stage[-1]:
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_coupling'))
            available_terms = [ # coupling electron with this qubit
                'pauliSet_1J{q}_{p}J{p}_d{N}'.format(
                    q = num_qubits, 
                    p = pauli_term, 
                    N = num_qubits
                )
                for pauli_term in ['x', 'y', 'z']
            ]
        elif 'transverse' in self.spawn_stage[-1]:
            num_qubits = int(self.spawn_stage[-1].strip('_qubit_transverse'))
            available_terms =  [
                'pauliSet_1J{N}_xJy_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_xJz_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_yJz_d{N}'.format(N = num_qubits)
            ]

        # Select new models
        if '_qubit_complete' in self.spawn_stage[-1]:
            new_models = self.all_layer_champions_by_num_qubits[num_qubits]
            
            new_spawn_stage = "{}_qubit_champion_selection".format(num_qubits)
            self.spawn_stage.append(new_spawn_stage)        

        else:
            unused_terms = list(
                set(available_terms) - set(present_terms)
            )

            new_models = []
            for term in unused_terms:
                new_model_terms = list(
                    set(present_terms + [term] )
                )
                new_model = '+'.join(new_model_terms)
                new_models.append(new_model)

            if len(new_models) <= 1:
                # Greedy addition of terms exhausted
                # -> move to next stage
                self.spawn_stage.append('{N}_qubit_complete'.format(N=num_qubits))


        self.log_print(["Models before correcting dimension:", new_models])
        new_models = [
            qmla.utilities.ensure_consisten_num_qubits_pauli_set(
                model
            ) for model in new_models
        ]
        self.log_print(["Models after correcting dimension:", new_models])

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

