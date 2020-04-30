import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, growth_rule
from qmla.growth_rules.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined
from qmla import database_framework

class FermiHubbardLatticeSet(
    fixed_lattice_set.LatticeSet
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
        self.true_lattice = topology_predefined._3_site_chain
        # self.true_lattice = topology_predefined._4_site_square
        self.onsite_terms_present = True
        self.true_model = self.model_from_lattice(self.true_lattice)

        self.available_lattices = [
            self.true_lattice, 
            topology_predefined._3_site_chain_fully_connected, 
            topology_predefined._4_site_chain,
            topology_predefined._4_site_square,
        ]
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_fermi_hubbard_half_filled
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_half_filled_superposition

        self.num_sites_true = database_framework.get_num_qubits(self.true_model)
        self.num_qubits_true = 2*self.num_sites_true # FH uses 2 qubits per sites (up and down spin) 
        self.max_num_qubits = 5
        self.max_num_probe_qubits = self.max_num_qubits
        self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_occupation_basis_down_in_first_site
        # self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_occupation_basis_up_in_first_site
        # self.plot_probe_generation_function = qmla.shared_functionality.probe_set_generation.fermi_hubbard_occupation_basis_down_in_all_sites
        self.timing_insurance_factor = 15
        self.min_param = 0.25
        self.max_param = 0.75
        self.true_model_terms_params = {
            'FH-hopping-sum_down_1h2_2h3_d3' : 0.25,
            'FH-hopping-sum_up_1h2_2h3_d3' : 0.75,
            'FH-hopping-sum_down_1h2_d2' : 0.25,
            'FH-hopping-sum_up_1h2_d2' : 0.75
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
            for s in [
                'up', 
                'down'
            ]
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
        # complete_model = qmla.database_framework.alph(complete_model)
        return complete_model

    def latex_name(self, name):
        separate_terms = name.split('+')

        all_terms = []
        for term in separate_terms:
            components = term.split('_')
            if 'FH-hopping-sum' in components:
                components.remove('FH-hopping-sum')
                connected_sites = ""
                for c in components:
                    if c in ['down', 'up']:
                        spin_type = c
                    elif c[0] == 'd':
                        num_sites = int(c[1:])
                    else:
                        sites = [int(s) for s in c.split('h')]
                        connected_sites += str(
                            "({},{})".format(sites[0], sites[1])
                        )
                
                if spin_type == 'up':
                    spin_label = str("\\uparrow")
                elif spin_type == 'down':
                    spin_label = str("\\downarrow")
                new_term = str(
                    "\hat{H}^{" + spin_label + "}"
                    + "_{"
                    + connected_sites
                    + "}"
                )
                all_terms.append(new_term)
            elif 'FH-onsite-sum' in components:
                components.remove('FH-onsite-sum')
                sites = []
                for c in components:
                    if c[0] == 'd':
                        num_sites = int(c[1:])
                    else:
                        sites.append(int(c))
                sites = sorted(sites)
                sites = ','.join([str(i) for i in sites])
                sites_not_present =  list(
                    set(range(1, num_sites+1))
                    - set(sites)
                )
                if len(sites_not_present) > 0:
                    new_term = str(
                        "\hat{N" + "}^{" 
                        + str(num_sites)
                        + "}_{" + sites + "}"
                    )
                else:
                    new_term = str(
                        "\hat{N" + "}^{" 
                        + str(num_sites)
                        + "}"
                    )
                all_terms.append(new_term)

        model_string = '+'.join(all_terms)
        return "${}$".format(model_string)
                



        
