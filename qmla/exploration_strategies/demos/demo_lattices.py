import numpy as np
import itertools
import sys
import os
import pandas as pd

from qmla.exploration_strategies import exploration_strategy
from qmla.exploration_strategies.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
from qmla.shared_functionality import topology_predefined
from qmla import model_building_utilities
import qmla.shared_functionality.topology_predefined as topologies

class DemoLattice(
    exploration_strategy.ExplorationStrategy
):
    r"""
    Demo of how lattices can be incorporated in the ES. 
    """
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
        self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.lattice_pauli_likewise_concise
        self.true_lattice_name = '_3_site_chain_fully_connected'
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
        return model_building_utilities.alph(complete_model)


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
        plot_level=2, 
        **kwargs
    ):
        save_file = os.path.join(
            save_directory, 
            'true_lattice.png'
        )
        self.log_print([
            "ES plots for fixed lattice. Save file:", save_file
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
        
class DemoIsing(DemoLattice):
    r"""
    Demo of lattices where the Ising formalism is used to define the true and considered models. 
    """
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):

        self.base_terms = ['z']
        self.transverse_field = 'x'
        self.true_model = "pauliLikewise_lx_1_2_3_d3+pauliLikewise_lz_1J2_1J3_2J3_d3"
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.true_model = "pauliLikewise_lx_1_2_3_d3+pauliLikewise_lz_1J2_1J3_2J3_d3"
        self.qhl_models = [
            "pauliSet_zJz_1J2_d3+pauliSet_zJz_1J3_d3+pauliSet_zJz_2J3_d3+pauliSet_x_1_d3+pauliSet_x_2_d3+pauliSet_x_3_d3",
            "pauliLikewise_lx_1_2_3_d3+pauliLikewise_lz_1J2_1J3_2J3_d3"
        ]
        self.qhl_models = [qmla.model_building_utilities.alph(m) for m in self.qhl_models]
        self.initial_models = self.qhl_models
        self.true_model_terms_params = {
            "pauliLikewise_lx_1_2_3_d3" : 0.2,
            "pauliLikewise_lz_1J2_1J3_2J3_d3" : 0.8
        }
        self.tree_completed_initially = True

    def latex_name(
        self,
        name,
        **kwargs
    ):
        if 'pauliLikewise' in name:
            return qmla.shared_functionality.latex_model_names.lattice_pauli_likewise_concise(name, **kwargs)
        elif 'pauliSet' in name:
            return qmla.shared_functionality.latex_model_names.pauli_set_latex_name(name, **kwargs)
            

class DemoIsingFullyParameterised(DemoIsing):
    
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):

        self.base_terms = ['z']
        self.transverse_field = None # neglecting transverse field here
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.transverse_field = None # neglecting transverse field here
        self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        self.true_model = self.model_from_lattice(self.true_lattice)

    def model_from_lattice(
        self, 
        lattice
    ):
        connected_sites = lattice.get_connected_site_list()
        connections = ['{s1}J{s2}'.format(s1=c[0], s2=c[1]) for c in connected_sites]
        conn_list = [list(str(c) for c in conn) for conn in connected_sites]
        conn_string = '_'.join(['J'.join(c) for c in conn_list])
        lattice_dimension = lattice.num_sites()

        individual_operators = [
            'pauliSet_{o}J{o}_{c}_d{N}'.format(
                o = op, 
                c = c, 
                N = lattice_dimension
            )
            for op in self.base_terms for c in connections
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
        return model_building_utilities.alph(complete_model)


