import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, growth_rule
from qmla.growth_rules.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla.shared_functionality import topology_predefined
from qmla import construct_models

class FermiHubbardLatticeSet(
    fixed_lattice_set.LatticeSet
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        self.true_lattice = topology_predefined._2_site_chain
        # self.true_lattice = topology_predefined._3_site_chain
        # self.true_lattice = topology_predefined._4_site_square
        self.onsite_terms_present = True
        self.true_model = self.model_from_lattice(self.true_lattice)
        # TEST:
        # self.true_model = 'FH-hopping-sum_down_1h2_d2+FH-onsite-sum_1_2_d2'
        
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = self.true_model,
            **kwargs
        )        
        self.log_print(["True model is:", self.true_model])

        self.available_lattices = [
            self.true_lattice, 
            topology_predefined._3_site_chain_fully_connected, 
            topology_predefined._4_site_chain,
            topology_predefined._4_site_square,
        ]

        # self.quantisation = 'first'
        self.quantisation = 'second'  
        if self.quantisation == 'first':
            # need a probe transformer
            self.probe_transformer = qmla.shared_functionality.probe_transformer.FirstQuantisationToJordanWigner(max_num_qubits = 7)
            # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.test_probes_first_quantisation
            self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict

        elif self.quantisation == 'second':
            # Default for FH
            
            self.probe_transformer = qmla.shared_functionality.probe_transformer.ProbeTransformation()        
            # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.test_probes_second_quantisation
            self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_fermi_hubbard_half_filled

        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_occupation_basis_down_in_first_site
        # self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_half_filled_superposition

        self.num_sites_true = construct_models.get_num_qubits(self.true_model)
        self.num_qubits_true = 2*self.num_sites_true # FH uses 2 qubits per sites (up and down spin) 
        self.num_probes = 25

        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.TimeList
        self.max_time_to_consider = 25
        self.max_num_qubits = 5
        self.max_num_probe_qubits = self.max_num_qubits
        # self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_occupation_basis_up_in_first_site
        # self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_occupation_basis_down_in_all_sites
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.lattice_set_fermi_hubbard
        self.timing_insurance_factor = 15
        self.min_param = 0.25
        self.max_param = 0.75
        self.true_model_terms_params = {
            # 3 sites
            'FH-hopping-sum_down_1h2_2h3_d3' : 0.25,
            'FH-hopping-sum_up_1h2_2h3_d3' : 0.75,
            'FH-onsite-sum_1_2_3_d3': 0.55,
            
            # 2 sites
            'FH-hopping-sum_down_1h2_d2' : 0.25,
            'FH-hopping-sum_up_1h2_d2' : 0.75,
            'FH-onsite-sum_1_2_d2': 0.55,
        }

    def model_from_lattice(self, lattice):
        connected_sites = lattice.get_connected_site_list()
        conn_list = [list(str(c) for c in conn) for conn in connected_sites]
        conn_string = '_'.join(['h'.join(c) for c in conn_list])
        lattice_dimension = lattice.num_sites()

        individual_operators = [
            'FH-hopping-sum_{spin}_{connections}_d{N}'.format(
                spin = s,
                connections = conn_string,
                N = lattice_dimension
            )
            for s in ['up', 'down']
        ]

        if self.onsite_terms_present: 
            site_string = '_'.join(
                [str(i) for i in range(1, 1+lattice_dimension)]
            )
            onsite_sum = 'FH-onsite-sum_{sites}_d{N}'.format(
                sites = site_string, 
                N = lattice_dimension
            )
            individual_operators.append(onsite_sum)

        complete_model = '+'.join(individual_operators)
        # complete_model = qmla.construct_models.alph(complete_model)
        return complete_model

    def expectation_value(self, **kwargs):
        r"""
        Transform probe to the Jordan Wigner basis before computing expectation value. 
        """

        try:
            ex_val = self.expectation_value_function(**kwargs)
            method = 'default'
        except:
            # transform - e.g. probe was from a different starting basis
            probe = kwargs['state']
            transformed_probe = self.probe_transformer.transform(probe = probe)
            kwargs['state'] = transformed_probe
            
            method = 'transform'

            ex_val = self.expectation_value_function(**kwargs)

        # self.log_print([
        #     "Expectation value method: {}".format(method)
        # ])
        return ex_val




        
