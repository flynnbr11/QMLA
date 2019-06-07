import sys, os
import numpy as np
sys.path.append(os.path.abspath('..'))
import DataBase

from SuperClassGrowthRule import GrowthRuleSuper


class hubbardSquare(GrowthRuleSuper):
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

        self.true_operator = 'h_1h2_1h3_2h4_3h4_d4PPPPh_e_d4'
        self.qhl_models = [
            'h_1h2_1h3_2h4_3h4_d4PPPPh_e_d4'
        ] 
        self.initial_models = [
            'h_1h2_1h3_2h4_3h4_d4PPPPh_e_d4'
        ]

        self.max_num_parameter_estimate = 2
        self.max_spawn_depth = 10
        self.max_num_qubits = 6
        
        self.max_num_models_by_shape = {
            4 : 2,
            6 : 2, 
            8 : 2, 
            9 : 2,
            'other' : 0
        }



    def generate_models(
        self,
        model_list, 
        **kwargs
    ):
        # from UserFunctions import initial_models, max_num_qubits_info, fixed_axes_by_generator
        # growth_generator = kwargs['generator']
        # model_list = kwargs['model_list']
        # from ModelGeneration import new_hubbard_model_from_square_lattice
        spawn_stage = kwargs['spawn_stage']
        max_num_qubits = self.max_num_qubits
        # max_num_qubits = max_num_qubits_info[growth_generator]
        
        new_models = []
        misc = kwargs['miscellaneous']
        if spawn_stage[-1] == None:
            topology = initialise_topology_2x2_square_lattice()
            misc['topology'] = topology
            
            # now get new model name and update topology in same step
            new_mod = new_hubbard_model_from_square_lattice(misc['topology'])
            new_models.append(new_mod)
            spawn_stage.append('topology_generated')
        elif spawn_stage[-1] == 'topology_generated':
            print("generating new topology. Starting from:\n", misc['topology'])
            new_mod = new_hubbard_model_from_square_lattice(misc['topology'])
            new_models.append(new_mod)
        
        if np.any(
            np.array([DataBase.get_num_qubits(mod) for mod in new_models]) >= 
            max_num_qubits
        ):
            print("Max num qubits {} reached".format(max_num_qubits))
            spawn_stage.append('Complete')

        return new_models


    def check_tree_completed(
        self, 
        current_num_qubits,
        **kwargs
    ):
        if (
            current_num_qubits 
            == 
            self.max_num_qubits
        ):
            return True 
        else:
            return False


    def name_branch_map(
        self,
        latex_mapping_file, 
        **kwargs
    ):
        import ModelNames
        name_map = ModelNames.branch_is_num_dims(
            latex_mapping_file = latex_mapping_file,
            **kwargs
        )
        return name_map

    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        individual_terms = DataBase.get_constituent_names_from_name(name)
        latex_term = ''

        hopping_terms = []
        interaction_energy = False
        for constituent in individual_terms:
            components = constituent.split('_')
            for term in components: 
                if term != 'h': #ie entire term not just 'h'
                    if 'h' in term: # ie a hopping term eg 1_1h2_d3, hopping sites 1-2, total num sites 3
                        split_term = term.split('h')
                        hopping_terms.append(split_term)

                    elif 'e' in term:
                        interaction_energy = True
                    elif 'd' in term:
                        dim = int(term.replace('d', ''))

        hopping_latex = 'H_{'
        for site_pair in hopping_terms:
            hopping_latex += str(
                '({},{})'.format(
                    str(site_pair[0]), 
                    str(site_pair[1])
                )
            )

        hopping_latex += '}'

        if hopping_latex != 'H_{}':
            latex_term += hopping_latex

        if interaction_energy is True:
            latex_term += str(
                '\sigma_{z}^{\otimes'
                +str(dim)
                +'}'
            )

        latex_term = str('$' + latex_term + '$')
        return latex_term



def initialise_topology_2x2_square_lattice():
    # Initialises a 2x2 square lattice
    topology = {
        'lattice_dimension' : 2,
        'span' : [0,0],
        'occupation' : {
            'rows' : {
                1 : [1, 2],
                2 : [1, 2]
            },
            'cols' : {
                1 : [1, 2], 
                2 : [1, 2]
            }
        },
        'coordinates' : {
            1 : [1,1],
            2 : [1,2], 
            3 : [2,1],
            4 : [2,2]
        },
        'nearest_neighbours' : {
            1 : [2,3],
            2 : [1,4], 
            3 : [1,4], 
            4 : [2,3]
        }
    }
    
    return topology

def new_hubbard_model_from_square_lattice(topology):

    add_sites_to_topology(topology)
    nearest_neighbours_list = get_nearest_neighbour_list(topology)
    nearest_neighbours_list = [list(a) for a in nearest_neighbours_list]
    num_qubits = len(topology['coordinates'])

    hubbard_model_dict = {
        'sites' : nearest_neighbours_list, 
        'dim' : num_qubits
    }

    new_model = generate_hopping_term(
        hubbard_model_dict, 
        include_interaction_energy=True, 
    )
    return new_model

    
def get_nearest_neighbour_list(topology):
    coordinates = topology['coordinates']
    site_indices = list(coordinates.keys())
    nearest_neighbours = []
    
    for i in range(len(site_indices)):
        idx_1 = site_indices[i]
        for j in range(i, len(site_indices)):
            idx_2 = site_indices[j]
            nn = check_nearest_neighbour_sites(
                site_1 = coordinates[idx_1],
                site_2 = coordinates[idx_2],
            )
            if nn is True:
                nearest_neighbours.append( (idx_1, idx_2) )
                
    return nearest_neighbours

def check_nearest_neighbour_sites(site_1, site_2):
    # simply checks whether sites are adjacent (or comptues distance)
    # assumes Cartesian coordinates
    if len(site_1) != len(site_2):
        print(
            "Site distance calculation: both sites must have same number of dimensions.",
            "Given:", site_1, site_2
        )
        raise NameError('Unequal site dimensions.')
    
    dim = len(site_1)
    dist = 0 
    for d in range(dim):
        dist += np.abs(site_1[d] - site_2[d])
        
    if dist == 1:
        return True
    else:
        return False

def add_sites_to_topology(topology):
    all_sites_greater_than_2_nearest_neighbours = False
    while all_sites_greater_than_2_nearest_neighbours == False:
        new_site_idx = add_new_coordinate_2d_lattice(topology)
        site_indices = list(topology['coordinates'].keys())

        new_coords  = topology['coordinates'][new_site_idx] 
        topology['nearest_neighbours'][new_site_idx] = []

        for i in site_indices:
            other_coords = topology['coordinates'][i] 

            nearest_neighbour = check_nearest_neighbour_sites(
                site_1 = new_coords,
                site_2 = other_coords
            )

            if nearest_neighbour is True:
                if i not in topology['nearest_neighbours'][new_site_idx]:
                    topology['nearest_neighbours'][new_site_idx].append(i)
                if new_site_idx not in topology['nearest_neighbours'][i]:
                    topology['nearest_neighbours'][i].append(new_site_idx)

        nn_lists = list(topology['nearest_neighbours'].values())
        num_nearest_neighbours = np.array([len(a) for a in nn_lists])
        all_sites_greater_than_2_nearest_neighbours = np.all(num_nearest_neighbours >= 2)

def add_new_coordinate_2d_lattice(topology):
    rows = topology['occupation']['rows']
    cols = topology['occupation']['cols']

    row_values = rows.keys()
    col_values = cols.keys() 
    min_span_row = None
    min_span_col = None        

    for row_idx in rows:
        span = max(rows[row_idx]) - min(rows[row_idx])
        if (
            min_span_row is None 
            or
            span < min_span_row
        ):
            min_span_row = span
            min_span_row_idx = row_idx

    for col_idx in cols:
        span = max(cols[col_idx]) - min(cols[col_idx])
        if (
            min_span_col is None 
            or
            span < min_span_col
        ):
            min_span_col = span
            min_span_col_idx = col_idx

    if min_span_col < min_span_row:
        # growing downward in y-axis
        new_row = max(cols[min_span_col_idx]) + 1
        new_col = min_span_col_idx
    else:
        # growing rightward in x-axis
        new_col = max(rows[min_span_row_idx]) + 1
        new_row = min_span_row_idx

    new_coordinate = [new_row, new_col]
    print("new coordinate:", new_coordinate)

    try:
        topology['occupation']['rows'][new_row].append(new_col)
    except:
        topology['occupation']['rows'][new_row] = [new_col]

    try:
        topology['occupation']['cols'][new_col].append(new_row)
    except:
        topology['occupation']['cols'][new_col] = [new_row]


    max_site_idx = max(list(topology['coordinates'].keys()))
    new_site_idx = max_site_idx + 1
    topology['coordinates'][new_site_idx] = new_coordinate
    return new_site_idx

   

def generate_hopping_term(
    deconstructed, 
    include_interaction_energy=False
):
    sites_list = deconstructed['sites']
    dim = deconstructed['dim']
    try:
        interaction_energy = deconstructed['interaction_energy']
    except:
        interaction_energy = include_interaction_energy
    
    if type(sites_list[0]) != list:
        sites_list = [sites_list]
    p_str = ''
    for i in range(dim):
        p_str += 'P'
    overall_term = ''
    first_term = True
    
    hopping_string = 'h'

    for sites in sites_list:
        hopping_string += str(
            '_' + 
            str(sites[0]) + 
            'h' + 
            str(sites[1])
        )
    
    hopping_string += str( '_d' + str(dim)) 
    overall_term += hopping_string

    if interaction_energy == True:
        interaction_term = str(
            'h_e_d' + str(dim) 
        )
        overall_term += str(
            p_str +
            interaction_term
        )
#         overall_term += str(
#             p_str 
#             + 
#             interaction_energy_pauli_term(dim)
#         )
    return overall_term
