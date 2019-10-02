import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelGeneration

import IsingMultiAxis



class ising_2D(
    # SuperClassGrowthRule.GrowthRuleSuper
    IsingMultiAxis.ising_chain_multi_axis
):

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )

        self.true_operator =  'pauliSet_zJz_1J2_d4PPPPpauliSet_zJz_1J3_d4PPPPpauliSet_zJz_2J4_d4PPPPpauliSet_zJz_3J4_d4'
        self.initial_models = [
            self.true_operator
        ]
        self.max_num_qubits = 8

        self.topology = ModelGeneration.initialise_topology_2x2_square_lattice()
        self.spawn_stage = [None]
        self.consider_transverse_axis = False

    def generate_models(
        self, 
        **kwargs
    ):

        new_models = []
        if self.spawn_stage[-1] == None:
            # now get new model name and update topology in same step
            new_mod = self.new_ising_2d_model_square_lattice()
            new_models.append(new_mod)

            # self.spawn_stage.append('Complete')
            if DataBase.get_num_qubits(new_mod) >= self.max_num_qubits:
                self.spawn_stage.append('Nontransverse complete')
                if self.consider_transverse_axis == False:
                    self.spawn_stage.append('Complete')

        elif self.spawn_stage[-1] == 'Nontransverse complete':
            champs = kwargs['current_champs']
            new_models = [
                IsingMultiAxis.add_transverse_name(
                    c, 
                    transverse_axis = self.transverse_axis
                ) for c in champs
            ]
            self.spawn_stage.append('Complete')

        return new_models


    def new_ising_2d_model_square_lattice(
        self
    ):
        ModelGeneration.add_sites_to_topology(self.topology)
        nearest_neighbours_list = ModelGeneration.get_nearest_neighbour_list(self.topology)
        nearest_neighbours_list = [list(a) for a in nearest_neighbours_list]
        num_qubits = len(self.topology['coordinates'])

        new_model_dict = {
            'sites' : nearest_neighbours_list, 
            'dim' : num_qubits
        }

        new_model = generate_ising2d_term(
            new_model_dict, 
            include_transverse_term = False
        )
        print("[2dIsing] new model returned:", new_model)
        return new_model



def generate_ising2d_term(
    model_dict,
    include_transverse_term=False
):
    # NOTE this generation function ONLY generates Ising models along a single axis
    # this should be okay assuming the standard definition where only sigma_z is present
    # The transverse axis is defined within the ising2d class


    sites_list = model_dict['sites']
    dim = model_dict['dim']
    
    if type(sites_list[0]) != list:
        sites_list = [sites_list]
    p_str = 'P'*dim
    overall_term = ''

    first_term = True
    pauli_marker = 'pauliSet'
    for sites in sites_list:
        if first_term is False:
            overall_term += p_str 
        else:
            first_term = False   
        overall_term += str(
            pauli_marker
            + "_{}J{}".format(
                str(sites[0]), 
                str(sites[1])
            ) 
            + "_{}J{}".format(
                'z',
                'z'
            ) 
            + "_d{}".format(
                str(dim)
            )
        )
    if include_transverse_term == True:
        overall_term = IsingMultiAxis.add_transverse_name(
            overall_term,
            transverse_axis = self.transverse_axis
        )
    return overall_term
