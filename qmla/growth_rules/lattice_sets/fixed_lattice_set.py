import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, growth_rule
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla import construct_models

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
        self.base_terms = ['x', 'z']
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.lattice_set_grouped_pauli
        self.initial_models = None # so that QMLA will call generate_models first
        # self.true_model = self.model_from_lattice(self.available_lattices[0])
        self.max_time_to_consider = 45
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
        return construct_models.alph(complete_model)


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


