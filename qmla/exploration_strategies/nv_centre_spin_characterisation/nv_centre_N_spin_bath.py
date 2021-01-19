from __future__ import absolute_import
import sys
import os
import itertools

from qmla.exploration_strategies.exploration_strategy import ExplorationStrategy
import qmla.shared_functionality.experiment_design_heuristics
import qmla.utilities
from qmla import construct_models


__all__ = [
    'NVCentreNQubitBath'
]

class NVCentreNQubitBath(
    ExplorationStrategy
):
    r"""
    Staged greedy term addition for NV centre in spin bath.
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

        # Choose functions 
        # self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution_double_time_reverse
        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution
        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.tomographic_basis
        self.plot_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_probes_dict
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.SampleOrderMagnitude
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.TimeList


        # QMLA and model learning configuration
        self.initial_models = None
        self.prune_completed_initially = True # pruning is implicit in the spawn rule

        self.min_param = 0
        self.max_param = 10
        self.num_probes = 25 # TODO restore 25 # |++'>

        non_spin_qubit_contributions = [
            'rotation', 
            # 'coupling',
            # 'transverserse'
        ]
        self.stages_by_num_qubits = {
            1 : iter(['rotation']),
            # 2 : iter( non_spin_qubit_contributions ),
            # 3 : iter( non_spin_qubit_contributions ),            
            # 4 : iter( non_spin_qubit_contributions )
        }

        # self.max_num_qubits = int(max( self.stages_by_num_qubits ))
        self.max_num_qubits = 5
        self.log_print(["Max num qubits:", self.max_num_qubits])
        
        self.substage_layer_champions =  { i : {} for i in range(1, self.max_num_qubits+1) }
        self.substage_champions_by_stage = { i : [] for i in range(1, self.max_num_qubits+1) }
        self.stage_champions = {}
        # self.greedy_mechanism = 'greedy_single_terms'
        self.greedy_mechanism = 'all_combinations'
        self.num_processes_to_parallelise_over = 7

        self.substage = 'rotation'
        self.spawn_stage = [None]        
        self.probe_maximum_number_qubits = 5
        self.include_transverse_terms = True

        self.fraction_opponents_experiments_for_bf = 0.5
        self.fraction_own_experiments_for_bf = self.fraction_opponents_experiments_for_bf
        self.fraction_particles_for_bf = 0.5

        # True model configuration        
        self._set_true_params()
        self.qhl_models =  [
            # 'pauliSet_1_x_d1',
            'pauliSet_1_y_d1',
            # 'pauliSet_1_z_d1',

            # 'pauliSet_1_x_d1+pauliSet_1_z_d1',
            # 'pauliSet_1_x_d1+pauliSet_1_y_d1',
            # 'pauliSet_1_y_d1+pauliSet_1_z_d1',

            # 'pauliSet_1_x_d1+pauliSet_1_y_d1+pauliSet_1_z_d1',
        ]
        # self.true_model = 'pauliSet_1_y_d2'
        # self.num_probes = 1
        # self.hard_fix_resample_effective_sample_size = 0
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.FixedTimeTest

        # Logistics
        self.max_num_models_by_shape = {
            # 1 : 3,
            'other': 1
        }
        self.num_processes_to_parallelise_over = 8
        self.timing_insurance_factor = 0.3

        # Test: a few hand picked models to see if true model wins
        self.test_preset_models = True
        if self.test_preset_models:
            self._setup_preset_models_test()


    def _setup_true_model_4_qubit_approx(self,):
        self.true_model_terms_params =  {
            # spin rotation
            'pauliSet_1_z_d4' : 7255832515, 
            # coupling
            'pauliSet_1J2_zJz_d4' : 197814, 
            'pauliSet_1J3_zJz_d4' : 577277, 
            'pauliSet_1J4_zJz_d4' : 643980, 
            # nuclear rotations
            'pauliSet_2_x_d4' : 72321, 
            'pauliSet_2_y_d4' : 67684, 
            'pauliSet_2_z_d4' : 15466, 
            'pauliSet_3_x_d4' : 27479, 
            'pauliSet_3_y_d4' : 63002,
            'pauliSet_3_z_d4' : 61635, 
            'pauliSet_4_x_d4' : 47225, 
            'pauliSet_4_y_d4' : 38920, 
            'pauliSet_4_z_d4' : 50960
        }


    def _setup_true_model_2_qubit_approx(self,):

        n_qubits = 2
        self.true_model_terms_params = {
            # spin
            'pauliSet_1_z_d{}'.format(n_qubits) : 2e9,
            
            # coupling with 2nd qubit
            'pauliSet_1J2_zJz_d{}'.format(n_qubits) : 0.2e6, 
            # 'pauliSet_1J2_yJy_d{}'.format(n_qubits) : 0.4e6, 
            # 'pauliSet_1J2_xJx_d{}'.format(n_qubits) : 0.2e6, 

            # carbon nuclei - 2nd qubit
            'pauliSet_2_x_d{}'.format(n_qubits) : 66e3,
            'pauliSet_2_y_d{}'.format(n_qubits) : 66e3,
            'pauliSet_2_z_d{}'.format(n_qubits) : 15e3,
        }


    def _set_true_params(self):

        # set target model
        # self._setup_true_model_2_qubit_approx()
        self._setup_true_model_4_qubit_approx()

        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)
        self.availalbe_pauli_terms  = ['x', 'y', 'z']

        self.max_time_to_consider = 100e-6
        self.plot_time_increment = 0.5e-6

        # max_num_qubits = 5
        test_prior_info = {}      
        paulis_to_include = self.availalbe_pauli_terms

        for pauli in paulis_to_include:

            for num_qubits in range(1, 1+self.max_num_qubits):
        
                spin_rotation_term = 'pauliSet_1_{p}_d{N}'.format(
                    p=pauli, N=num_qubits)
                test_prior_info[spin_rotation_term] = (5e9, 2e9)

                for j in range(2, 1+num_qubits):

                    nuclei_rotation = 'pauliSet_{j}_{p}_d{N}'.format(
                        j = j, 
                        p = pauli, 
                        N = num_qubits
                    )
                    test_prior_info[nuclei_rotation] = (5e4, 2e4)

                    coupling_w_spin = 'pauliSet_1J{j}_{p}J{p}_d{N}'.format(
                        j = j, 
                        p = pauli,
                        N = num_qubits
                    )
                    test_prior_info[coupling_w_spin] = (5e5, 2e5)

                    # TODO add transverse terms

        self.gaussian_prior_means_and_widths = test_prior_info


    def _setup_preset_models_test(self):
        self.initial_models = [
            # secular_approximation(2),
            secular_approximation(3), 
            secular_approximation(4),
            secular_approximation(5),
            # secular_approximation(6),

        ]
        self.initial_models = [
            qmla.construct_models.alph(m) for m in self.initial_models
        ]
        self.tree_completed_initially = True
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.max_num_models_by_shape = {
            # 2 : 6,
            # 1 : 7,
            'other': 1
        }
        self.num_processes_to_parallelise_over = len(self.initial_models)+1


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
                for pauli_term in self.availalbe_pauli_terms
                # ['x', 'y']
                # ['x', 'y', 'z']
            ]
        elif substage == 'coupling':
            self.log_print(["Available: coupling terms"])
            available_terms = [ # coupling electron with this qubit
                'pauliSet_1J{q}_{p}J{p}_d{N}'.format(
                    q = num_qubits, 
                    p = pauli_term, 
                    N = num_qubits
                )
                for pauli_term in self.availalbe_pauli_terms
                # ['x', 'y', 'z']
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
        
        if self.greedy_mechanism == 'greedy_single_terms':
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

        elif self.greedy_mechanism == 'all_combinations':
            new_models = []

            for i in range(1, len(available_terms) + 1 ):
                combinations = list(itertools.combinations(available_terms, i))
                model_lists = [ present_terms + list(a) for a in combinations ]

                # these_models = [qmla.utilities.flatten(ml) for ml in model_lists]
                # self.log_print([
                #     "Model lists to generate", these_models
                # ])
                these_models = ['+'.join(m) for m in model_lists]
                
                new_models.extend(these_models)

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
                self.log_print(['Num qubits already maximum so not increasing.'])
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
            if len(new_models) == 1 or self.greedy_mechanism == 'all_combinations':
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



def secular_approximation(num_qubits):
#     num_qubits = self.target_num_qubits

    available_terms = [
        # electron spin rotation terms
        'pauliSet_1_z_d{}'.format(num_qubits), 
    ]

    for k in range(2, num_qubits+1):

        coupling_terms = [
            'pauliSet_1J{k}_zJz_d{n}'.format(
                k = k,
                n=num_qubits
            )
        ]
        rotation_terms = [
            'pauliSet_{k}_{p}_d{n}'.format(
                k = k, 
                p = pauli_term, 
                n = num_qubits
            )
            for pauli_term in ['x', 'y', 'z']
        ]

        available_terms.extend(coupling_terms)
        available_terms.extend(rotation_terms)

    secular_approx = '+'.join(available_terms)
    secular_approx = qmla.construct_models.alph(secular_approx)
    return secular_approx

