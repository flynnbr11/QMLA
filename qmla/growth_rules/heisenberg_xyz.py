import numpy as np
import itertools
import sys
import os

from qmla.growth_rules import connected_lattice, connected_lattice_probabilistic
import qmla.shared_functionality.probe_set_generation
from qmla import database_framework


class HeisenbergXYZProbabilistic(
    connected_lattice_probabilistic.ConnectedLatticeProbabilistic
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        self.lattice_dimension = 2
        self.initial_num_sites = 4
        self.lattice_connectivity_max_distance = 1
        self.lattice_connectivity_linear_only = True
        self.lattice_full_connectivity = False
        self.max_num_sites = 4
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.pauli_eigenvector_based_probes

        self.three_site_chain_xxz = 'pauliSet_1J2_xJx_d3+pauliSet_2J3_zJz_d3'
        self.four_site_xxz_chain = 'pauliSet_1J2_xJx_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_3J4_zJz_d4'
        self.four_site_xxz = 'pauliSet_1J2_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4'
        
        self.true_model = self.four_site_xxz_chain
        self.true_model = database_framework.alph(self.true_model)
        self.qhl_models = [
            self.true_model,
            'pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4'
        ]
        self.base_terms = [
            'x',
            # 'y',
            'z'
        ]
        # self.num_probes = 50
        # self.max_time_to_consider = 50
        # fitness calculation parameters. fitness calculation inherited.
        # 'all' # 'all' # at each generation Badassness parameter

        self.num_top_models_to_build_on =  1 # to test strict generation to ensure sensible paths occuring
        # self.num_top_models_to_build_on =  'all'
        self.model_generation_strictness = 0  # 1 #-1
        self.fitness_win_ratio_exponent = 1
        self.fitness_minimum = 1
        self.fitness_maximum = 1.0
        self.check_champion_reducibility = True
        self.generation_DAG = 1
        # self.true_model_terms_params = {
        #     'pauliSet_1J2_xJx_d4': 0.27044671107574969, 
        #     'pauliSet_1J3_zJz_d4': 1.1396665426731736, 
        #     'pauliSet_2J4_xJx_d4': 0.38705331216054806, 
        #     'pauliSet_3J4_xJx_d4': 0.46892509638460805, 
        #     'pauliSet_3J4_zJz_d4': 0.45440765993845578
        # }

        self.tree_completed_initially = False
        self.num_processes_to_parallelise_over = 10
        self.max_num_models_by_shape = {
            # 1 : 0,
            # 2: 10,
            # 3: 10,
            2: 4,
            3: 4,
            # 4: 50,
            'other': 0
        }

        self.setup_growth_class()


class HeisenbergXYZPredetermined(
    HeisenbergXYZProbabilistic
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        # self.max_time_to_consider = 100
        self.tree_completed_initially = True
        self.num_processes_to_parallelise_over = 8
        self.max_num_models_by_shape = {
            # Note dN here requires 2N qubits so d3 counts as shape 6
            1: 0,
            2: 1,
            4: 3,
            5: 2,
            6: 2,
            'other': 0
        }
        self.max_num_qubits = 3
        self.max_num_sites = 6
        self.setup_growth_class()

        # self.true_model_terms_params = {
        #     'pauliSet_1J2_zJz_d4': 0.7070533314487537, 
        #     'pauliSet_2J4_zJz_d4': 0.46976922504656166, 
        #     'pauliSet_1J3_xJx_d4': 0.5445604382949641, 
        #     'pauliSet_3J4_zJz_d4': 0.9825508924850654, 
        #     'pauliSet_1J2_xJx_d4': 0.268227093044357, 
        #     'pauliSet_3J4_xJx_d4': 0.7849629571198494, 
        #     'pauliSet_2J4_xJx_d4': 0.48907849424521366, 
        #     'pauliSet_1J3_zJz_d4': 0.1972300936609916
        # }

        if self.tree_completed_initially == True:
            # to manually fix the models to be considered
            # heis_xxz_4site = 'pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4'
            # self.true_model = heis_xxz_4site
            models = []
            list_connections = [
                [(1, 2)],  # pair of sites
                [(1, 2), (2, 3)],  # chain length 3
                [(1, 2), (1, 3), (3, 4), (2, 4)],  # square,
                [(1, 2), (2, 3), (3, 4)],  # chain,
                [(1, 2), (2, 3), (3, 4), (4, 5)],  # chain,
                # [(1,2), (2,3), (3,4), (4,5), (5,6)], # chain,
                [(1, 2), (1, 4), (2, 3), (2, 5),
                 (3, 6), (4, 5), (5, 6)]  # 3x2 grid
            ]
            for connected_sites in list_connections:

                system_size = max(max(connected_sites))
                if system_size  > self.max_num_sites: 
                    self.max_num_sites = system_size + 1
                terms = connected_lattice.pauli_like_like_terms_connected_sites(
                    connected_sites=connected_sites,
                    base_terms=self.base_terms, 
                    num_sites=system_size
                )

                p_str = 'P' * system_size
                models.append(p_str.join(terms))

            self.initial_models = models
            self.true_model = self.initial_models[2]

            ###########
            # testing that subsystem is better than random alternative
            ##########

            # i.e. 1 qubit model containing correct subsystem wins 1 qubit generation
            # self.true_model = 'pauliSet_1J2_xJx_d3+pauliSet_2J3_zJz_d3'
            # self.initial_models = [
            #     'pauliSet_1J2_xJx_d3',
            #     'pauliSet_1J2_zJz_d3',
            #     'pauliSet_1J2_yJy_d3',
            #     'pauliSet_1J2_xJx_d3+pauliSet_1J2_yJy_d3',
            #     'pauliSet_1J2_xJx_d3+pauliSet_1J2_zJz_d3',
            #     'pauliSet_1J2_zJz_d3+pauliSet_1J2_yJy_d3',
            #     'pauliSet_1J2_xJx_d3+pauliSet_1J2_yJy_d3+pauliSet_1J2_zJz_d3',
            # ]


            # self.true_model_terms_params = {
            #     'pauliSet_1J2_xJx_d3': 0.27044671107574969, 
            #     # 'pauliSet_2J3_zJz_d3': 0, 
            #     'pauliSet_2J3_zJz_d3': 1.1396665426731736, 
            #     'pauliSet_1J3_zJz_d3': 1.1396665426731736, 
            #     'pauliSet_1J2_xJx_d4': 0.27044671107574969, 
            #     'pauliSet_1J3_zJz_d4': 1.1396665426731736, 
            #     'pauliSet_2J4_xJx_d4': 0.38705331216054806, 
            #     'pauliSet_3J4_xJx_d4': 0.46892509638460805, 
            #     'pauliSet_3J4_zJz_d4': 0.45440765993845578
            # }

            if self.true_model not in self.initial_models:
                self.log_print("Adding true operator to initial model list")
                self.initial_models.append(self.true_model)
        


        self.max_num_models_by_shape = {
            2 : 6, 
            3 : 6, 
            4: 6,
            'other': 0
        }


class HeisenbergSharedField(
    HeisenbergXYZPredetermined
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        # self.true_model = 'Heis_ix_d2PPHeis_iy_d2PPHeis_iz_d2PPHeis_tz_d2'
        # self.true_model = 'likewisePauliSum_lx_1J2_1J3_2J4_3J4_d4+likewisePauliSum_lz_1J2_1J3_2J4_3J4_d4'
        self.true_model = 'pauliLikewise_lx_1J2_1J3_2J4_3J4_d4+pauliLikewise_ly_1J2_1J3_2J4_3J4_d4+pauliLikewise_lz_1J2_1J3_2J4_3J4_d4'
        self.initial_models = [
            'pauliLikewise_lx_1J2_1J3_2J4_3J4_d4+pauliLikewise_ly_1J2_1J3_2J4_3J4_d4+pauliLikewise_lz_1J2_1J3_2J4_3J4_d4',
            'pauliLikewise_lx_1J2_2J3_3J4_d4+pauliLikewise_ly_1J2_2J3_3J4_d4+pauliLikewise_lz_1J2_2J3_3J4_d4',
            'pauliLikewise_lx_1J2_1J3_2J3_d3+pauliLikewise_ly_1J2_1J3_2J3_d3+pauliLikewise_lz_1J2_1J3_2J3_d3'
        ]
        if self.true_model not in self.initial_models:
            self.initial_models.append(self.true_model)
        self.qhl_models = self.initial_models
        self.tree_completed_initially=True

    def latex_name(
        self,
        name,
        **kwargs
    ):
        return "${}$".format(name)


    # def latex_name(
    #     self,
    #     name,
    #     **kwargs
    # ):
    #     # print("[latex name fnc] name:", name)
    #     core_operators = list(sorted(database_framework.core_operator_dict.keys()))
    #     num_sites = database_framework.get_num_qubits(name)
    #     try:
    #         p_str = 'P' * num_sites
    #         separate_terms = name.split(p_str)
    #     except:
    #         p_str = '+'
    #         separate_terms = name.split(p_str)

    #     site_connections = {}
    #     for c in list(itertools.combinations(list(range(num_sites + 1)), 2)):
    #         site_connections[c] = []

    #     # term_type_markers = ['pauliSet', 'transverse']
    #     transverse_axis = None
    #     heis_axis = None
    #     for term in separate_terms:
    #         components = term.split('_')
    #         if 'Heis' in components:
    #             components.remove('Heis')
    #             for l in components:
    #                 if l[0] == 'd':
    #                     dim = int(l.replace('d', ''))
    #                 elif l[0] == 'i':
    #                     heis_axis = str(l.replace('i', ''))
    #                 elif l[0] == 't':
    #                     transverse_axis = str(l.replace('t', ''))

    #     latex_term = ""
    #     if heis_axis is not None:
    #         this_term = r"\sigma_{"
    #         this_term += str(heis_axis)
    #         this_term += "}^{\otimes"
    #         this_term += str(dim)
    #         this_term += "}"
    #         latex_term += this_term
    #     if transverse_axis is not None:
    #         this_term = r"T_{"
    #         this_term += str(transverse_axis)
    #         this_term += "}^{\otimes"
    #         this_term += str(dim)
    #         this_term += "}"
    #         latex_term += this_term

    #     if latex_term == "":
    #         print("Heisenberg shared field could not generate latex string for ", name)
    #     latex_term = "${}$".format(latex_term)
    #     return latex_term
