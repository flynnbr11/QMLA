import sys
import os
import itertools

from qmla.exploration_strategies.nv_centre_spin_characterisation import nv_centre_full_access
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.expectation_value_functions
from qmla import construct_models


class SimulatedNVCentre(
    nv_centre_full_access.ExperimentFullAccessNV  # inherit from this
):
    # Uses some of the same functionality as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out
    # and different model generation

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        # Set up true model
        B = 11e-3 # Tesla
        g = 2 # 
        bohr_magneton = 9.274e-24 # J T^-1
        hbar = 1.05e-34 # m^2 kg s^-1
        nuclear_magneton = 5.05e-27 # J T^-1 
        gamma_n = 0.307e6 / 1e-6 # from Seb's thesis
        gamma = 10.705e6 # T^-1 s^-1 # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5226623/
        self.true_model_terms_params = {
            # 'x' : 0.55,
            # test
            # 'pauliSet_1_x_d1' : 4, # TEST - kHz
            # 'pauliSet_1_y_d1' : 4, # TEST - kHz
            # 'pauliSet_1_z_d1' : 4, # TEST - kHz
            # spin - test MHz terms
            # 'pauliSet_1_x_d3' : 1e-3*B*g*bohr_magneton/hbar, # ~1.943 GHz = 1943123809.5238094
            # 'pauliSet_1_y_d3' : 1e-3*B*g*bohr_magneton/hbar,
            # 'pauliSet_1_z_d3' : 1e-3*B*g*bohr_magneton/hbar,
            # spin
            # 'pauliSet_1_x_d3' : B*g*bohr_magneton/hbar, # ~1.943 GHz = 1943123809.5238094
            'pauliSet_1_y_d3' : B*g*bohr_magneton/hbar,
            'pauliSet_1_z_d3' : B*g*bohr_magneton/hbar,
            # nitrogen nuclei
            # 'pauliSet_2_x_d3' : B*gamma_n , # ~3.37GHz
            # 'pauliSet_2_y_d3' : B*gamma_n ,
            # 'pauliSet_2_z_d3' : B*gamma_n ,
            # # carbon nuclei
            # 'pauliSet_3_x_d3' : B * gamma , # ~117KHz
            # 'pauliSet_3_y_d3' : B * gamma ,
            # 'pauliSet_3_z_d3' : B * gamma ,
            # # interactions: spin with nitrogen nuclei
            # 'pauliSet_1J2_xJx_d3' : 2.7e6, # 2.7MHz
            # 'pauliSet_1J2_yJy_d3' : 2.7e6,
            # 'pauliSet_1J2_zJz_d3' : 2.14e6,
            # # interactions: spin with carbon nuclei
            # 'pauliSet_1J3_xJx_d3' : 2.4e6, # 2.4MHz
            # 'pauliSet_1J3_yJy_d3' : 2.4e6, 
            # 'pauliSet_1J3_zJz_d3' : 2.4e6,
        }
        

        self.gaussian_prior_means_and_widths = {
            # 'x' : (0.5, 0.2),
            # test
            'pauliSet_1_x_d1' : (5, 2), # TEST
            'pauliSet_1_y_d1' : (5, 2), # TEST
            'pauliSet_1_z_d1' : (5, 2), # TEST
            # spin w v thin prior

            # values found in a QHL for GHz terms
            # 'pauliSet_1_x_d3' : (1.85197880e+09,   7.28778717e+07), # ~1.943 GHz
            # 'pauliSet_1_y_d3' : (1.97413207e+09,   7.77490072e+07),
            # 'pauliSet_1_z_d3' : (1.99779857e+09,  3.95883908e+07),

            # 'pauliSet_1_x_d3' : (B*g*bohr_magneton/hbar - 0.4e7, 5e7), # ~1.943 GHz
            # 'pauliSet_1_y_d3' : (B*g*bohr_magneton/hbar + 0.5, 5e7),
            # 'pauliSet_1_z_d3' : (B*g*bohr_magneton/hbar - 0.5, 5e7),
            # 'pauliSet_1_z_d3' : (B*g*bohr_magneton/hbar - 50, 100),
            # spin
            'pauliSet_1_x_d3' : (5e9, 2e9), # ~1.943 GHz
            'pauliSet_1_y_d3' : (5e9, 2e9),
            'pauliSet_1_z_d3' : (5e9, 2e9),
            # nitrogen nuclei
            'pauliSet_2_x_d3' : (5e9, 2e9) , # ~3.37GHz
            'pauliSet_2_y_d3' : (5e9, 2e9) ,
            'pauliSet_2_z_d3' : (5e9, 2e9) ,
            # carbon nuclei
            'pauliSet_3_x_d3' : (1e6, 2e5) , # ~117KHz
            'pauliSet_3_y_d3' : (1e6, 2e5) ,
            'pauliSet_3_z_d3' : (1e6, 2e5) ,
            # interactions: spin with nitrogen nuclei
            'pauliSet_1J2_xJx_d3' : (5e6, 2e6), # 2.7MHz
            'pauliSet_1J2_yJy_d3' : (5e6, 2e6),
            'pauliSet_1J2_zJz_d3' : (5e6, 2e6),
            # interactions: spin with carbon nuclei
            'pauliSet_1J3_xJx_d3' : (5e6, 2e6), # 2.4MHz
            'pauliSet_1J3_yJy_d3' : (5e6, 2e6), 
            'pauliSet_1J3_zJz_d3' : (5e6, 2e6),

        }

        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)
        
        self.tree_completed_initially = True
        self.initial_models=None
        self.expectation_value_subroutine = \
            qmla.shared_functionality.expectation_value_functionsn_qubit_hahn_evolution_double_time_reverse
        self.timing_insurance_factor = 2/3
        self.num_probes = 20
        time_basis = 1e-9 # nanoseconds
        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.eigenbasis_of_first_qubit
        self.max_time_to_consider = 50 * time_basis # 50 microseconds 
        self.plot_time_increment = 0.5 * time_basis # 0.5 microseconds
        self.track_quadratic_loss = True
        # self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.default_expectation_value
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MultiParticleGuessHeuristic
        self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.SampleOrderMagnitude
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.SampledUncertaintyWithConvergenceThreshold

    def generate_models(self, model_list, **kwargs):
        if self.spawn_stage[-1]==None:
            self.spawn_stage.append('Complete')
            return [self.true_model]
    
    def check_tree_completed(self, **kwargs):
        if self.spawn_stage[-1] == 'Complete':
            return True
        else:
            return False


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
    model = qmla.construct_models.alph(model)
    print("Spin system used:", model)
    return model
    
    


class TestSimulatedNVCentre(
    SimulatedNVCentre  # inherit from this
):
    # Uses some of the same functionality as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out
    # and different model generation

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        

        # Set up true model
        B = 11e-3 # Tesla
        g = 2 # 
        bohr_magneton = 9.274e-24 # J T^-1
        hbar = 1.05e-34 # m^2 kg s^-1
        nuclear_magneton = 5.05e-27 # J T^-1 
        gamma_n = 0.307e6 / 1e-6 # from Seb's thesis
        gamma = 10.705e6 # T^-1 s^-1 # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5226623/


        order_mag = -1
        self.true_model_terms_params = {
            # spin
            # 'pauliSet_1_x_d3' : B*g*bohr_magneton/hbar, # ~1.943 GHz = 1943123809.5238094
            # 'pauliSet_1_y_d3' : B*g*bohr_magneton/hbar,
            # 'pauliSet_1_z_d3' : B*g*bohr_magneton/hbar,

            'pauliSet_1_x_d1' : 4.2431238095238094 * (10**order_mag),
            'pauliSet_1_y_d1' : 5.9431238095238094 * (10**order_mag),
            # 'pauliSet_1_z_d1' : 5.9431238095238094 * (10**order_mag),

            # 'pauliSet_1_x_d2' : 3.2431238095238094 * (10**order_mag),
            # 'pauliSet_1_y_d2' : 6.9431238095238094 * (10**order_mag),
            # 'pauliSet_1_x_d2' : 1.9431238095238094 * (10**order_mag),
            # 'pauliSet_1_y_d2' : 8.9431238095238094 * (10**order_mag),
            # 'pauliSet_1_z_d2' : 3.9431238095238094 * (10**order_mag),
            # 'pauliSet_2_x_d2' : 4.9431238095238094 * (10**order_mag),
            # 'pauliSet_2_y_d2' : 7.9431238095238094 * (10**order_mag),
            # 'pauliSet_2_z_d2' : 6.9431238095238094 * (10**order_mag),
        }

        self.gaussian_prior_means_and_widths = {
            # start accurate with very small prior
            # 'pauliSet_1_x_d1' : ( 3.243 * (10**order_mag) , 0.001 * (10**order_mag) ),
            # 'pauliSet_1_y_d1' : ( 6.943 * (10**order_mag) , 0.001 * (10**order_mag) ) ,
            # 'pauliSet_1_z_d1' : (5* (10**order_mag), 2* (10**order_mag)),

            # spin
            'pauliSet_1_x_d1' : (5* (10**order_mag), 2* (10**order_mag)),
            'pauliSet_1_y_d1' : (5* (10**order_mag), 2* (10**order_mag)),
            'pauliSet_1_z_d1' : (5* (10**order_mag), 2* (10**order_mag)),

            'pauliSet_1_x_d2' : (5* (10**order_mag), 2* (10**order_mag)),
            'pauliSet_1_y_d2' : (5* (10**order_mag), 2* (10**order_mag)),
            'pauliSet_1_z_d2' : (5* (10**order_mag), 2* (10**order_mag)),

            'pauliSet_2_x_d2' : (5* (10**order_mag), 2* (10**order_mag)),
            'pauliSet_2_y_d2' : (5* (10**order_mag), 2* (10**order_mag)),
            'pauliSet_2_z_d2' : (5* (10**order_mag), 2* (10**order_mag)),

        }

        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)
        self.qinfer_resampler_threshold = 0.15
        self.qinfer_resampler_a = 0.98
        self.iqle_mode = False
        self.hard_fix_resample_effective_sample_size = 1000

        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.default_expectation_value
        
        # Choose heuristic
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MultiParticleGuessHeuristic
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.RandomTimeUpperBounded
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.VolumeAdaptiveParticleGuessHeuristic
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.FixedNineEighthsToPowerK


        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.eigenbasis_of_first_qubit
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.manual_set_probes
        time_basis = 1/10**order_mag # nanoseconds
        # self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.eigenbasis_of_first_qubit
        self.max_time_to_consider = 50 * time_basis # 50 microseconds 
        self.plot_time_increment = 5 * time_basis # 0.5 microseconds
        self.timing_insurance_factor = 0.25