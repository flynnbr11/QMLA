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
            # spin
            'pauliSet_1_x_d3' : B*g*bohr_magneton/hbar, # ~1.943 GHz
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
            'pauliSet_1J2_xJx_d3' : 2.7e6, # 2.7MHz
            'pauliSet_1J2_yJy_d3' : 2.7e6,
            'pauliSet_1J2_zJz_d3' : 2.14e6,
            # # interactions: spin with carbon nuclei
            'pauliSet_1J3_xJx_d3' : 2.4e6, # 2.4MHz
            'pauliSet_1J3_yJy_d3' : 2.4e6, 
            'pauliSet_1J3_zJz_d3' : 2.4e6,
        }
        

        self.gaussian_prior_means_and_widths = {
            # 'x' : (0.5, 0.2),
            # test
            'pauliSet_1_x_d1' : (5, 2), # TEST
            'pauliSet_1_y_d1' : (5, 2), # TEST
            'pauliSet_1_z_d1' : (5, 2), # TEST
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
        self.true_model = qmla.database_framework.alph(self.true_model)
        
        self.tree_completed_initially = True
        self.initial_models=None
        self.expectation_value_function = \
            qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        self.model_heuristic_function = \
            qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic            
        self.timing_insurance_factor = 0.5
        time_basis = 1e-9 # nanoseconds
        self.max_time_to_consider = 50 * time_basis # 50 microseconds 
        self.plot_time_increment = 0.5 * time_basis # 0.5 microseconds
        # self.expectation_value_function = qmla.shared_functionality.expectation_values.default_expectation_value
        # self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MultiParticleGuessHeuristic
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.SampleOrderMagnitude

    def generate_models(self, model_list, **kwargs):
        if self.spawn_stage[-1]==None:
            self.spawn_stage.append('Complete')
            return [self.true_model]
    
    def check_tree_completed(self, **kwargs):
        if self.spawn_stage[-1] == 'Complete':
            return True
        else:
            return False

    def _latex_name(
        self,
        name,
        **kwargs
    ):
        return "${}$".format(name)

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
    
    
