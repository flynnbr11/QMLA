import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, growth_rule
import qmla.shared_functionality.probe_set_generation
from qmla import database_framework

class LatticeSet(
    # connected_lattice.ConnectedLattice
    growth_rule.GrowthRule
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
        self.tree_completed_initially = True # fixed lattice set considered
        self.available_lattices = [
            qmla.shared_functionality.topology.GridTopology(
                dimension=1, num_sites = 3
            ), # 3 site chain
            qmla.shared_functionality.topology.GridTopology(
                dimension=2, 
                num_sites=4,
            ) # square
        ]
        self.base_terms = ['x']
        self.initial_models = None # so that QMLA will call generate_models first
        # self.true_model = self.model_from_lattice(self.available_lattices[0])
        self.transverse_field = None
        self.timing_insurance_factor = 5

    def model_from_lattice(
        self, 
        lattice
    ):
        connected_sites = lattice.get_connected_site_list()
        conn_list = [list(str(c) for c in conn) for conn in connected_sites]
        conn_string = '_'.join(['J'.join(c) for c in conn_list])
        lattice_dimension = lattice.num_sites()

        individual_operators = [
            'pauliLikewise_l{}_{}_d{}'.format(
                op, 
                conn_string, 
                lattice_dimension
            )
            for op in self.base_terms
        ]
        complete_model = '+'.join(individual_operators)

        if self.transverse_field is not None:
            transverse_string = (
                '_'.join(list(str(s) for s in range(1, lattice_dimension+1)))
            )
            transverse_term = 'pauliLikewise_l{}_{}_d{}'.format(
                self.transverse_field, 
                transverse_string, 
                lattice_dimension
            )
            complete_model += '+{}'.format(transverse_term)
        return complete_model


    def generate_models(
        self, 
        model_list, 
        **kwargs 
    ):
        model_set = [
            self.model_from_lattice(lattice)
            for lattice in self.available_lattices
        ]
        self.log_print([
            "Generate models returning ", model_set
        ])

        return model_set


    def latex_name(
        self,
        name,
        **kwargs
    ):
        separate_terms = name.split('+')
        latex_term = ""
        latex_terms = {}
        for term in separate_terms:
            components = term.split('_')
            try:
                components.remove('pauliLikewise')
            except:
                print("Couldn't remove pauliLikewise from", name)
            this_term_connections = []
            for l in components:
                if l[0] == 'd':
                    dim = int(l.replace('d', ''))
                elif l[0] == 'l':
                    operator = str(l.replace('l', ''))
                else:
                    n_sites = len(l.split('J'))
                    sites = l.replace('J', ',')
                    # if n_sites > 1: sites = "(" + str(sites) + ")"
                    this_term_connections.append(sites)

            # limits for sum
            lower_limit = str(
                "i \in "
                +",".join(this_term_connections)
            )
            operator_string = str("\sigma_{ i }^{" + str(operator) + "}")
            if n_sites == 1: 
                sites_not_present = list(
                    set([int(i) for i in this_term_connections]) 
                    - set(range(1, dim+1))
                )
                if len(sites_not_present) == 0:
                    lower_limit = "i=1"
            elif n_sites == 2:
                nns = [(str(n), str(n+1) ) for n in range(1, dim)]
                nns = [','.join(list(nn)) for nn in nns]
                nns = set(nns)
                sites_not_present = list(
                    set(this_term_connections)
                    - nns
                )
                if len(sites_not_present) == 0:
                    lower_limit = "i"
                    operator_string = str(
                        "\hat{\sigma}_{i}^{" + str(operator) + "}"
                        + "\hat{\sigma}_{i+1}^{" + str(operator) + "}"
                    )
                else: 
                    this_term_connections = [
                        "({})".format(c) for c in this_term_connections
                    ]
                    lower_limit = str(
                        "i \in "
                        +",".join(this_term_connections)
                    )

            upper_limit = str(
                "N={}".format(dim)
            )
            new_term = str(
                "\sum"
                + "_{"
                    + lower_limit
                + "}"
                + "^{"
                    + upper_limit
                + "}"
                + operator_string
                # + "\hat{\sigma}^{"
                # + "}_{" + "i" + "}"
            )
            
            if n_sites not in latex_terms:
                latex_terms[n_sites] = {}
            if operator not in latex_terms[n_sites]:
                latex_terms[n_sites][operator] = new_term


        site_numbers = sorted(latex_terms.keys())
        all_latex_terms = []
        latex_model = ""
        for n in site_numbers:
            for term in latex_terms[n]:
                all_latex_terms.append(latex_terms[n][term])
        latex_model = "+".join(all_latex_terms)
        return "${}$".format(latex_model)

class IsingLatticeSet(LatticeSet):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        self.true_lattice = qmla.shared_functionality.topology.GridTopology(
                dimension=1, num_sites = 4
        )
        self.base_terms = ['z']
        self.transverse_field = 'x'
        self.true_model = self.model_from_lattice(self.true_lattice)

        self.available_lattices = [
            self.true_lattice, 
            qmla.shared_functionality.topology.GridTopology(
                dimension=1, num_sites = 3
            ), # 3 site chain
            qmla.shared_functionality.topology.GridTopology(
                dimension=1, num_sites = 5
            ), # 5 site chain
            qmla.shared_functionality.topology.GridTopology(
                dimension=1, num_sites = 6
            ), # 6 site chain
        ]
        self.true_model_terms_params = {
            'pauliLikewise_lz_1J2_2J3_3J4_d4' : 0.78,
            'pauliLikewise_lx_1_2_3_4_d4' : 0.12,
        }
        self.num_processes_to_parallelise_over = 4
        self.max_num_models_by_shape = {
            3 : 2, 
            4 : 2, 
            5 : 2, 
            6 : 2, 
            'other' : 0
        }
        self.max_time_to_consider = 50
        


class HeisenbergLatticeSet(LatticeSet):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.transverse_field = None # 'x'
        self.true_lattice = qmla.shared_functionality.topology.GridTopology(
                dimension=2, num_sites = 2
        ) # square
        self.true_model = self.model_from_lattice(self.true_lattice)

        self.available_lattices = [
            self.true_lattice, 
            qmla.shared_functionality.topology.GridTopology(
                dimension=1, num_sites = 3
            ), # 3 site chain
            qmla.shared_functionality.topology.GridTopology(
                dimension=2, num_sites = 6
            ), # 6 site grid
            qmla.shared_functionality.topology.GridTopology(
                dimension=2, num_sites=5, all_sites_connected=True
            ), # fully connected 5 site
            qmla.shared_functionality.topology.GridTopology(
                dimension=2, num_sites=2, all_sites_connected=True
            ) # fully connected square
        ]
        self.base_terms = ['x', 'z']
        


