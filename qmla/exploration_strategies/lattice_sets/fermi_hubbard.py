import numpy as np
import itertools
import sys
import os

from qmla.exploration_strategies.lattice_sets import fixed_lattice_set
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
from qmla.shared_functionality import topology_predefined
import qmla.shared_functionality.model_constructors
from qmla import model_building_utilities


__all__ = [
    "FermiHubbardLatticeSet", 
    "HubbardReducedLatticeSet"
]

class FermiHubbardLatticeSet(
    fixed_lattice_set.LatticeSet
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
    
        self.onsite_terms_present = True
        super().__init__(
            exploration_rules=exploration_rules,
            # true_model = self.true_model,
            **kwargs
        )
        self.model_constructor = qmla.shared_functionality.model_constructors.FermilibModel

        self._lattice_names = [
            '_2_site_chain', 
            '_3_site_chain', 
            '_3_site_lattice_fully_connected', 
            '_4_site_lattice_fully_connected',
            '_4_site_square',
        ] # TODO excluding 4 sites models for tests against other ESs -- reinstate afterwards
        self.setup_models()


    def setup_models(self):

        self.available_lattices_by_name = {
            k : topology_predefined.__getattribute__(k)
            for k in self.lattice_names
        }
        self.available_lattices = [
            topology_predefined.__getattribute__(k)
            for k in self.lattice_names
        ]
        if self._shared_true_parameters:
            lattice_idx = -1
        else:
            lattice_idx = self.qmla_id % len(self.available_lattices)  
        self.true_lattice_name = self.lattice_names[ lattice_idx ]

        self.true_lattice = self.available_lattices_by_name[self.true_lattice_name]
        self.true_model = self.model_from_lattice(self.true_lattice)
        # self.log_print(["QMLA {} using lattice {} (lattice idx {}) has model {}".format(self.qmla_id, self.true_lattice_name, lattice_idx, self.true_model)])
        self.max_num_qubits = 8
        self.max_num_probe_qubits = 8

        self.qhl_models = [
            self.model_from_lattice(l)
            for l in self.available_lattices
        ]

        self.quantisation = 'first' # 'second
        if self.quantisation == 'first':
            # probe transformer between formalisms - not used currently
            self.probe_transformer = qmla.shared_functionality.probe_transformer.ProbeTransformation()        
            self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.separable_probe_dict

        elif self.quantisation == 'second':
            # Default for FH
            self.probe_transformer = qmla.shared_functionality.probe_transformer.ProbeTransformation()        
            self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.separable_fermi_hubbard_half_filled
            
        # TODO plot probe dict with meaningful probes wrt FH model 
        self.plot_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_probes_dict

        self.num_sites_true = model_building_utilities.get_num_qubits(self.true_model)
        self.num_qubits_true = 2*self.num_sites_true # FH uses 2 qubits per sites (up and down spin) 
        self.num_probes = 25

        self.max_time_to_consider = 25
        self.max_num_qubits = 8
        self.max_num_probe_qubits = self.max_num_qubits
        # self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.lattice_set_fermi_hubbard
        # self.timing_insurance_factor = 0.8
        self.min_param = 0.25
        self.max_param = 0.75
        self.true_model_terms_params = {
            # 3 sites
            # 'FH-hopping-sum_down_1h2_2h3_d3' : 0.25,
            # 'FH-hopping-sum_up_1h2_2h3_d3' : 0.75,
            # 'FH-onsite-sum_1_2_3_d3': 0.55,
            
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
        complete_model = qmla.model_building_utilities.alph(complete_model)
        return complete_model


class HubbardReducedLatticeSet(FermiHubbardLatticeSet):

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        r"""
        Only consider 3-site models. 
        For use in multi-exploration-strategy tests. 
        """
    
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        self._lattice_names = [
            '_2_site_chain', 
            '_3_site_chain', 
            '_3_site_lattice_fully_connected', 
        ] 
        self.setup_models()

        # self.timing_insurance_factor = 0.2
        self.max_num_models_by_shape = {
            4: 2,
            6: 2,
            'other': 0
        } # test


        
