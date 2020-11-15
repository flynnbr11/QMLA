import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy
from qmla.exploration_strategies.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined
from qmla import construct_models
import qmla.shared_functionality.topology_predefined as topologies

class ThesisLatticeDemo(
    exploration_strategy.ExplorationStrategy
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):

        self.base_terms = ['z']
        self.transverse_field = 'x'
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.lattice_set_grouped_pauli
        self.true_lattice_name = '_4_site_lattice_fully_connected'
        self.true_lattice = topologies.__getattribute__(self.true_lattice_name)
        self.true_model = self.model_from_lattice(self.true_lattice)


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


    def exploration_strategy_specific_plots(
        self,
        save_directory,
        qmla_id=0, 
    ):
        save_file = os.path.join(
            save_directory, 
            'true_lattice.png'
        )
        self.log_print([
            "GR plots for fixed lattice. Save file:", save_file
        ])
        self.true_lattice.draw_topology(
            save_to_file = save_file
        )

    def exploration_strategy_finalise(
        self
    ):        
        self.storage.lattice_record =  pd.DataFrame(
            columns=['true_lattice', 'model_type',],
            data =np.array([[self.true_lattice_name, self.exploration_rules]])
        )
        



