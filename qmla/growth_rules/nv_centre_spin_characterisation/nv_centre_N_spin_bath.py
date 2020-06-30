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
    r"""
    Staged greedy term addition for NV centre in spin bath.
    """

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
        # self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution_double_time_reverse
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probe_generation_function = self.probe_generation_function
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        # self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic

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
        self.num_probes = 25 # |++'>

        non_spin_qubit_contributions = [
            'rotation', 
            'coupling',
            # 'transverserse'
        ]
        self.stages_by_num_qubits = {
            1 : iter(['rotation']),
            # 2 : iter( non_spin_qubit_contributions ),
            # 3 : iter( non_spin_qubit_contributions ),            
            # 4 : iter( non_spin_qubit_contributions )
        }

        self.max_num_qubits = int(max( self.stages_by_num_qubits ))
        self.log_print(["Max num qubits:", self.max_num_qubits])
        
        self.substage_layer_champions =  { i : {} for i in range(1, self.max_num_qubits+1) }
        self.substage_champions_by_stage = { i : [] for i in range(1, self.max_num_qubits+1) }
        self.stage_champions = {}

        self.substage = 'rotation'
        self.spawn_stage = [None]        
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
        self.timing_insurance_factor = 0.4

        # Test: a few hand picked models to see if true model wins
        self.test_preset_models = True
        if self.test_preset_models:

            self.initial_models = [
                'pauliSet_1_x_d1', 
                'pauliSet_1_y_d1', 
                'pauliSet_1_z_d1', 
                'pauliSet_1_x_d1+pauliSet_1_y_d1', 
                'pauliSet_1_x_d1+pauliSet_1_z_d1', 
                'pauliSet_1_y_d1+pauliSet_1_z_d1', 
                'pauliSet_1_x_d1+pauliSet_1_y_d1+pauliSet_1_z_d1', 

                # 'pauliSet_1_x_d2+pauliSet_1_z_d2+pauliSet_2_y_d2+pauliSet_1J2_zJz_d2', 
                # 'pauliSet_1_x_d2+pauliSet_1_y_d2+pauliSet_1_z_d2', 
                # 'pauliSet_1_x_d2+pauliSet_2_y_d2+pauliSet_1J2_zJz_d2', 
                # 'pauliSet_1_z_d2+pauliSet_2_z_d2+pauliSet_1J2_zJz_d2', 
                # 'pauliSet_1_x_d2+pauliSet_1_y_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2+pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2+pauliSet_1J2_zJz_d2',
                # 3 qubits
                # 'pauliSet_1_z_d3+pauliSet_2_z_d3+pauliSet_3_z_d3+pauliSet_1J2_zJz_d3+pauliSet_1J3_zJz_d3', 
                # 'pauliSet_1_x_d3+pauliSet_1_y_d3+pauliSet_1_z_d3+pauliSet_2_x_d3+pauliSet_2_y_d3+pauliSet_2_z_d3+pauliSet_3_x_d3+pauliSet_3_y_d3+pauliSet_3_z_d3', 

            ]
            self.initial_models = [
                qmla.construct_models.alph(m) for m in self.initial_models
            ]
            self.tree_completed_initially = True
            if self.tree_completed_initially:
                self.max_spawn_depth = 1
            self.max_num_models_by_shape = {
                2 : 6,
                1 : 7,
                'other': 0
            }
            self.num_processes_to_parallelise_over = len(self.initial_models)
            self.timing_insurance_factor = 0.25


    # Model generation / QMLA progression

    def greedy_add(self, model_list,  num_qubits, substage):
        try:
            top_model = model_list[0]
        except:
            top_model = None
        
        self.log_print(["Greedily adding terms. N qubits = {}, substage={}".format(num_qubits, substage)])
        if substage == 'rotation':
            self.log_print(["Available: rotation terms"])
            available_terms = [
                'pauliSet_{N}_{p}_d{N}'.format(p=pauli_term, N=num_qubits)
                for pauli_term in 
                # ['x', 'y']
                ['x', 'y', 'z']
            ]
        elif substage == 'coupling':
            self.log_print(["Available: coupling terms"])
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
            available_terms =  [
                'pauliSet_1J{N}_xJy_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_xJz_d{N}'.format(N = num_qubits),
                'pauliSet_1J{N}_yJz_d{N}'.format(N = num_qubits)
            ]

        # Select new models
        
        if top_model is not None: 
            present_terms = top_model.split('+')
        else:
            present_terms = []

        unused_terms = list(
            set(available_terms) - set(present_terms)
        )
        
        self.log_print([
            "Present terms:", present_terms, 
            "\nAvailable terms:", unused_terms
        ])
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

    def get_layer_champions_by_substage(self, num_qubits, substage):
        self.log_print(["Getting layer champions for {} qubits by substage {}".format(num_qubits, substage)])
        self.log_print(["Substage layer champions:", self.substage_layer_champions])
        return self.substage_layer_champions[num_qubits][substage]

    def get_substage_champions_by_stage(self, num_qubits):
        self.log_print(["Getting substage champions for {} qubits".format(num_qubits)])
        return self.substage_champions_by_stage[num_qubits]
    
    def get_all_stage_champions(self):
        self.log_print(["Getting all stage champions"])
        return list(self.stage_champions.values())


    def generate_models(self, model_list, **kwargs):
        try:
            top_model = model_list[0]
            num_qubits = qmla.construct_models.get_num_qubits(top_model)
        except:
            top_model = None
            num_qubits = 1
        self.log_print([
            "Generating models. \nModel list={} \nSpawn stage:{}\n N qubits={} \nsub stage={}".format(
                model_list, self.spawn_stage[-1], num_qubits, self.substage)
        ])

        signal = self.spawn_stage[-1]
        
        if signal is None:
            # very start
            self.log_print(["No signal - at start of generate models lifecycle"])
            num_qubits = 1
            self.substage = next(self.stages_by_num_qubits[num_qubits])
            self.log_print(["New stage found for {} qubits: {}. Retuning to greedy add".format(num_qubits, self.substage)])
            self.spawn_stage.append('default')
        
        elif signal == 'finding_substage_champion':
            self.substage_champions_by_stage[num_qubits].append( top_model )
            top_model = None # so it isn't recorded in the next substage
            try:
                self.substage = next(self.stages_by_num_qubits[num_qubits])
                self.log_print(["New stage found for {} qubits: {}. Retuning to greedy add".format(num_qubits, self.substage)])
                self.spawn_stage.append('default')
            except:
                # no substage left for this num qubits
                self.log_print(["No further stage found for {} qubits. Declaring stage complete".format(num_qubits)])
                self.spawn_stage.append('stage_complete')

        elif signal == 'finding_stage_champion':
            self.stage_champions[num_qubits] = top_model 
            top_model = None # so it isn't recorded in the next substage

            if num_qubits == self.max_num_qubits:
                self.log_print('Num qubits already maximum so not increasing.')
                self.spawn_stage.append('all_stages_complete')
            else: 
                # increase number of qubits
                num_qubits += 1
                self.log_print(["Trying to get first substage of {} qubits. Stages: {}".format(
                    num_qubits, self.stages_by_num_qubits)])
                self.substage = next(self.stages_by_num_qubits[num_qubits])
                self.log_print(["Increasing # qubits. Now {} with substage {}".format(num_qubits, self.substage)])
                self.spawn_stage.append('default')

        signal = self.spawn_stage[-1]
        if signal == 'default':
            if top_model is not None:
                try:
                    self.substage_layer_champions[num_qubits][self.substage].append(top_model)
                except:
                    self.substage_layer_champions[num_qubits][self.substage] = [top_model]
            new_models = self.greedy_add(
                model_list = model_list, 
                num_qubits = num_qubits, 
                substage = self.substage
            )
            if len(new_models) == 1:
                self.spawn_stage.append('substage_complete')
        
        elif signal == 'substage_complete':
            try:
                self.substage_layer_champions[num_qubits][self.substage].append(top_model)
            except:
                self.substage_layer_champions[num_qubits][self.substage] = [top_model]

            new_models = self.get_layer_champions_by_substage(num_qubits=num_qubits, substage = self.substage) 
            self.spawn_stage.append('finding_substage_champion')

        
        elif signal == 'stage_complete':
            new_models = self.get_substage_champions_by_stage(num_qubits)

            self.spawn_stage.append('finding_stage_champion')
            
        elif signal == 'all_stages_complete':
            new_models = self.get_all_stage_champions()
            self.spawn_stage.append('Complete')

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

