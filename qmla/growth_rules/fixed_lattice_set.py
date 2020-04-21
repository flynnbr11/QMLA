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
        self.true_model = self.model_from_lattice(self.available_lattices[0])


    def model_from_lattice(
        self, 
        lattice
    ):
        # shared field type model

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
        all_connections = []
        latex_term = ""
        connections_terms = {}
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
                    sites = l.split('J')
                    this_term_connections.append(sites)
            for s in this_term_connections:
                # con = "({},{})".format(s[0], s[1])
                con = ",".join(list(s))
                con = "({})".format(con)
                try:
                    connections_terms[con].append(operator)
                except:
                    connections_terms[con] = [operator]

            latex_term = ""
            for c in list(sorted(connections_terms.keys())):
                connection_string = str(
                    "\sigma_{"
                    + str(c)
                    + "}^{"
                    + str(",".join(connections_terms[c]))
                    + "}"
                )
                latex_term += connection_string

        return "${}$".format(latex_term)


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
        self.base_terms = ['z']

    def model_from_lattice(
        self, 
        lattice
    ):
        # shared field on z-axis with transverse x-field
        transverse_field = 'x'

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

        transverse_string = '_'.join(list(str(s) for s in range(1, lattice_dimension+1)))
        transverse_term = 'pauliLikewise_l{}_{}_d{}'.format(
            transverse_field, 
            transverse_string, 
            lattice_dimension
        )
        complete_model = '+'.join(individual_operators)
        complete_model += '+{}'.format(transverse_term)
        return complete_model

