##### ---------- -------------------- #####  
"""
This file allows users to inject their own functionality into QMD.
To specify a new set of rules, QMD must be provided some flags
- growth_generation_rule: 
	which tree growth mechanism to use
- measurement_type: 
	which measurement ought to be applied to the quantum system, 
	corresponding to experimental measurements/other 
	constraints of your system. 

These each require new functionality to be used
(or else to use existing functionality). In either case, 
you must inform QMD what functions to use for your case. 
Below are wrapper functions which are called within QMD. 
The functions these pass into must be altered for new QMD applications.
- Tell each of the dictionaries below the 
	corresponding functions to your growth and measurement rules. 
- These must be written to take the parameters QMD passes, 
	some inputs are essential and some optional. 
	Essential inputs for each newly written function
	are explained inline below. 
"""
##### ---------- -------------------- #####  



import ModelGeneration 
import ModelNames 
import ExpectationValues
import ProbeGeneration

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time

def log_print(to_print_list, log_file):
    identifier = str(str(time_seconds()) +" [USER FUNCTIONS]")
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file, flush=True)


##### ---------- -------------------- #####  
# Measurement/Expectation Values
##### ---------- -------------------- #####  

expec_val_function_dict = {
	# measurement_type : expectation_value_function_for_that_meas
    'full_access' : 
    	ExpectationValues.expectation_value, 
    'hahn' : 
    	ExpectationValues.hahn_evolution,
	'n_qubit_hahn' : 
    	ExpectationValues.n_qubit_hahn_evolution,
    'trace_all_but_first' : 
    	ExpectationValues.traced_expectation_value_project_one_qubit_plus
}


##### ---------- -------------------- #####  
# Model Generation
# Here you must also define how many growth steps to allow 
# the decision tree to undergo, in max_spawn_depth_info. 
##### ---------- -------------------- #####  

default_true_operators_by_generator = {
    'test_changes_to_qmd' : 'x',
    'PT_Effective_Hamiltonian' : 'zTxPPyTxPPxTiPPxTz' ,
    # 'NV_centre_spin_large_bath' : 'nv_spin_x_d10PPPPPPPPPPnv_interaction_x_d10PPPPPPPPPPnv_spin_y_d10PPPPPPPPPPnv_interaction_y_d10PPPPPPPPPPnv_spin_z_d10PPPPPPPPPPnv_interaction_z_d10',
    # 'NV_centre_spin_large_bath' : 'nv_spin_x_d9PPPPPPPPPnv_interaction_x_d9PPPPPPPPPnv_spin_y_d9PPPPPPPPPnv_interaction_y_d9PPPPPPPPPnv_spin_z_d9PPPPPPPPPnv_interaction_z_d9',
    # 'NV_centre_spin_large_bath' : 'nv_spin_x_d4PPPPnv_interaction_x_d4PPPPnv_spin_y_d4PPPPnv_interaction_y_d4PPPPnv_spin_z_d4PPPPnv_interaction_z_d4',
    'NV_centre_spin_large_bath' : 'nv_spin_x_d3PPPnv_interaction_x_d3PPPnv_spin_y_d3PPPnv_interaction_y_d3PPPnv_spin_z_d3PPPnv_interaction_z_d3',
    # 'NV_centre_spin_large_bath' : 'nv_spin_z_d3PPPnv_interaction_z_d3',
    # 'reduced_nv_experiment' : 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz' ,
    'reduced_nv_experiment' : 'zTi' ,
    'NV_centre_experiment_debug' : 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz' ,
    'two_qubit_ising_rotation' : 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
    'two_qubit_ising_rotation_hyperfine' : 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz', 
    'two_qubit_ising_rotation_hyperfine_transverse' : 'xTiPPyTiPPzTiPPxTxPPyTyPPzTz',
    # 'two_qubit_ising_rotation_hyperfine_transverse' : 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz', # TODO reinstate -- testing whether alphetisation of true model affects December/new code
    # 'two_qubit_ising_rotation_hyperfine_transverse' : 'xTiP',
    'test_multidimensional' : 'xTiPPiTy', 
    'test_return_champs' : 'xTiPPiTy', 
    'non_interacting_ising' : 'xTxTTx',
    'non_interacting_ising_single_axis' : 'xTxTTx',
    'deterministic_noninteracting_ising_single_axis' : 'xTxTTx',
    'interacting_nearest_neighbour_ising' : 'xTxTTiPPPiTxTTx',
    'interacing_nn_ising_fixed_axis' : 'xTxTTiPPPiTxTTx',
    'deterministic_interacting_nn_ising_single_axis' : 'xTxTTiPPPiTxTTx',
    'deterministic_transverse_ising_nn_fixed_axis' : 'zTiPPiTzPPxTx',
    'ising_1d_chain' : '1Dising_ix_d2PP1Dising_tz_d2',

    'heisenberg_nontransverse' : 'xTxPPzTz',
    'heisenberg_transverse' : 'xTxPPyTyPPiTzPPzTi',
    'heisenberg_xyz' : 'Heis_ix_d3PPPHeis_iy_d3PPPHeis_iz_d3PPPHeis_tz_d3',

    # 'hubbard' : 'h_1_2_d3PPPh_1_3_d3PPPh_2_3_d3'
    # 'hubbard' : 'h_1_2_d2PPiTzPPzTi',
    'hubbard' : 'h_1h2_d2PPiTzPPzTi',
    # 'hubbard_chain_just_hopping' : 'h_1_2_d3PPPh_1_3_d3PPPh_2_3_d3',
    'hubbard_chain_just_hopping' : 'h_1h2_d3PPPh_1h3_d3PPPh_2h3_d3',
    # 'hubbard_chain_just_hopping' : 'h_1_2_d2'
	# 'hubbard_chain' : 'h_1_2_d2PPiTzPPzTi',
	'hubbard_chain' : 'h_1h2_d2PPh_e_d2',
	# 'hubbard_square_lattice_generalised' : 'h_1_2_d6PPPPPPh_1_3_d6PPPPPPh_2_4_d6PPPPPPh_2_5_d6PPPPPPh_3_4_d6PPPPPPh_4_6_d6PPPPPPh_5_6_d6PPPPPPzTiTTiTTTiTTTTiTTTTTiPPPPPPiTzTTiTTTiTTTTiTTTTTiPPPPPPiTiTTzTTTiTTTTiTTTTTiPPPPPPiTiTTiTTTzTTTTiTTTTTiPPPPPPiTiTTiTTTiTTTTzTTTTTiPPPPPPiTiTTiTTTiTTTTiTTTTTz'
	# 'hubbard_square_lattice_generalised' : 'h_1_2_d6PPPPPPh_1_3_d6PPPPPPh_2_4_d6PPPPPPh_2_5_d6PPPPPPh_3_4_d6PPPPPPh_4_6_d6PPPPPPh_5_6_d6PPPPPPzTiTTiTTTiTTTTiTTTTTiPPPPPPiTzTTiTTTiTTTTiTTTTTiPPPPPPiTiTTzTTTiTTTTiTTTTTiPPPPPPiTiTTiTTTzTTTTiTTTTTiPPPPPPiTiTTiTTTiTTTTzTTTTTiPPPPPPiTiTTiTTTiTTTTiTTTTTz'
	'hubbard_square_lattice_generalised' : 
		'h_1h2_1h3_2h4_3h4_d4PPPPh_e_d4',
		# 'h_1h2_1h3_2h4_3h4_2h5_4h6_5h6_d6PPPPPPh_e_d6',
		# 'h_1h2_d6PPPPPPh_1h3_d6PPPPPPh_2h4_d6PPPPPPh_3h4_d6PPPPPPh_2h5_d6PPPPPPh_4h6_d6PPPPPPh_5h6_d6PPPPPPh_e_d6',

	'hopping_topology' : 
        	'h_1h2_d2'
	#	'h_1h2_d4PPPPh_3h4_d4',
}


qhl_models_by_generator = {
	# for growth rules to run multiQHL test on models other than true model
    'PT_Effective_Hamiltonian' : 
    	['zTxPPyTxPPxTiPPxTz'],

    'NV_centre_experiment_debug' : 
    	[
    		'xTiPPxTxPPyTiPPyTyPPzTiPPzTz' 
		],
    'NV_centre_spin_large_bath' : 
    	[
    		'nv_spin_x_d3PPnv_interaction_x_d3',
    		'nv_spin_x_d2PPnv_interaction_x_d2',
    		'nv_spin_x_d4PPnv_interaction_x_d4',
		],
    'reduced_nv_experiment' : 
    	[
    		'xTiPPxTxPPyTiPPyTyPPzTiPPzTz' 
		],
    'two_qubit_ising_rotation_hyperfine' : 
    	[
    		'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
    		# 'xTiPPxTxPPyTiPPyTyPPzTi',
    		'xTiPPyTiPPyTyPPzTi',
#    		'xTiPPyTiPPzTiPPzTz',
    	],
    'two_qubit_ising_rotation_hyperfine_transverse' : 
    	[
    		'xTiPPxTxPPxTyPPxTzPPyTiPPyTyPPyTzPPzTiPPzTz',
    		'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
            'zTi'
    		# 'xTiPPxTxPPyTiPPyTyPPzTi',
    		# 'xTiPPyTiPPzTiPPzTz',
    	],
	'heisenberg_xyz' : [
		'Heis_ix_d3PPPHeis_iy_d3PPPHeis_iz_d3PPPHeis_tz_d3',    
		'Heis_ix_d3PPPHeis_iy_d3PPPHeis_iz_d3',    
	]
}

max_num_parameter_estimate = {
    'test_changes_to_qmd' : 3, 
    'PT_Effective_Hamiltonian' : 4,
    'NV_centre_experiment_debug' : 7,
    'NV_centre_spin_large_bath' : 6,
    'reduced_nv_experiment' : 7,
    'two_qubit_ising_rotation' : 6,
    'two_qubit_ising_rotation_hyperfine' : 6, 
    'two_qubit_ising_rotation_hyperfine_transverse' : 9,
    'non_interacting_ising' : 6,
    'non_interacting_ising_single_axis' : 6,
    'deterministic_noninteracting_ising_single_axis' : 6,
    'interacting_nearest_neighbour_ising' : 6,
    'interacing_nn_ising_fixed_axis' : 6,
    'deterministic_interacting_nn_ising_single_axis' : 6,
    'deterministic_transverse_ising_nn_fixed_axis' : 6,
	'heisenberg_nontransverse' : 8,
	'heisenberg_transverse' : 8,
	'hubbard_chain_just_hopping' : 10,
	'hubbard_chain' : 10,
	'hubbard_square_lattice_generalised' : 2,
	'ising_1d_chain' : 2,
	'heisenberg_xyz' : 2,
	'hopping_topology' : 8,
}

fixed_axis_generators = [
    'non_interacting_ising_single_axis',
    'interacing_nn_ising_fixed_axis',
    'deterministic_noninteracting_ising_single_axis',
    'deterministic_interacting_nn_ising_single_axis'
]

fixed_axes_by_generator = {
	None : 'z',
    'interacing_nn_ising_fixed_axis' : 'x',
    'deterministic_transverse_ising_nn_fixed_axis' : 'x',
    'non_interacting_ising_single_axis' : 'x', 
    'heisenberg_transverse' : 'z',
    'heisenberg_nontransverse' : 'z',
	'ising_1d_chain' : 'x',
}

transverse_axis_by_generator = {
	'ising_1d_chain' : 'z',
	'heisenberg_xyz' : 'z'
}

max_spawn_depth_info = {
    'test_changes_to_qmd' : 1, 
    'PT_Effective_Hamiltonian' : 1,
    'NV_centre_spin_large_bath' : 1,
    'reduced_nv_experiment' : 3,
    'NV_centre_experiment_debug' : 1,
    'two_qubit_ising_rotation' : 2,
    'two_qubit_ising_rotation_hyperfine' : 5, # for dev, should be 5 #TODO put back
    'two_qubit_ising_rotation_hyperfine_transverse' : 8,
    'non_interacting_ising' : 3,
    'non_interacting_ising_single_axis' : 3,
    'deterministic_noninteracting_ising_single_axis' : 1,
    'interacting_nearest_neighbour_ising' : 3,
    'interacing_nn_ising_fixed_axis' : 3,
    'deterministic_interacting_nn_ising_single_axis' : 1,
    'deterministic_transverse_ising_nn_fixed_axis' : 1,
	'heisenberg_nontransverse' : 8,
	'heisenberg_transverse' : 30,
	'hubbard_chain_just_hopping' : 10,
	'hubbard_chain' : 10,
	'hubbard_square_lattice_generalised' : 10,
	'ising_1d_chain' : 5,
	'heisenberg_xyz' : 5,
	'hopping_topology' : 100,
}

max_num_qubits_info = {
    'test_changes_to_qmd' : 2,
    'PT_Effective_Hamiltonian' : 2 ,
    'NV_centre_spin_large_bath' : 10,
    'NV_centre_experiment_debug' : 2,
    'reduced_nv_experiment' : 2,
    'two_qubit_ising_rotation' : 3,
    'two_qubit_ising_rotation_hyperfine' : 3, # for dev, should be 5 #TODO put back
    'two_qubit_ising_rotation_hyperfine_transverse' : 3,
    'non_interacting_ising_single_axis' : 5,
    'non_interacting_ising' : 5, 
    'deterministic_noninteracting_ising_single_axis' : 5,
	'interacting_nearest_neighbour_ising' : 5, 
	'interacing_nn_ising_fixed_axis' : 6, 
    'deterministic_interacting_nn_ising_single_axis' : 5,
    'deterministic_transverse_ising_nn_fixed_axis' : 5,
	'heisenberg_nontransverse' : 3,
	'heisenberg_transverse' : 3,
	'hubbard' : 5, 
	'hubbard_chain_just_hopping' : 5,
	'hubbard_chain' : 5,
	'hubbard_square_lattice_generalised' : 4,
	'ising_1d_chain' : 5,
    'heisenberg_xyz' : 5,
    None : 5,
    'hopping_topology' : 4,
}

model_generation_functions = {
	# growth_generation_rule : growth_function
    'test_changes_to_qmd' :
    	ModelGeneration.test_changes_to_qmd,
    'PT_Effective_Hamiltonian' : 
    	ModelGeneration.NV_centre_experiment_debug, # ie NO spawning, just run first model
    'NV_centre_spin_large_bath' : 
    	ModelGeneration.NV_centre_spin_large_bath,
    'reduced_nv_experiment' : 
    	ModelGeneration.reduced_nv_experimental_method,
    'NV_centre_experiment_debug' : 
    	ModelGeneration.NV_centre_experiment_debug, 
    'simple_ising' : 
    	ModelGeneration.simple_ising,
    'ising_non_transverse' : 
    	ModelGeneration.ising_non_transverse,
    'ising_transverse' : 
    	ModelGeneration.ising_transverse,
    'two_qubit_ising_rotation_hyperfine' : 
    	ModelGeneration.hyperfine_like,
    'two_qubit_ising_rotation_hyperfine_transverse' : 
    	ModelGeneration.hyperfine_like,
    'hyperfine_like' : 
    	ModelGeneration.hyperfine_like,
    'test_multidimensional' : 
    	ModelGeneration.test_multidimensional,
	'test_return_champs' : 
		ModelGeneration.existing_branch_champs_test,
	'non_interacting_ising': 
		ModelGeneration.non_interacting_ising,
	'non_interacting_ising_single_axis':
		ModelGeneration.non_interacting_ising_single_axis,
	'interacting_nearest_neighbour_ising': 
		ModelGeneration.interacting_nearest_neighbour_ising,
	'interacing_nn_ising_fixed_axis':
		ModelGeneration.interacting_nearest_neighbour_ising,
	'deterministic_noninteracting_ising_single_axis' : 
		ModelGeneration.deterministic_noninteracting_ising_single_axis,
	'deterministic_interacting_nn_ising_single_axis' :
		ModelGeneration.deterministic_interacting_nn_ising_single_axis,
	'deterministic_transverse_ising_nn_fixed_axis' : 
		ModelGeneration.deterministic_transverse_ising_nn_fixed_axis,
	'ising_1d_chain' : 
		ModelGeneration.generate_models_ising_1d_chain,
	'heisenberg_nontransverse' : 
		ModelGeneration.heisenberg_nontransverse,
	'heisenberg_transverse':
		ModelGeneration.heisenberg_transverse,
	'heisenberg_xyz' :
		ModelGeneration.generate_models_heisenberg_xyz,
	'hubbard_chain_just_hopping' :
		ModelGeneration.hubbard_chain_just_hopping,
	'hubbard_chain' : 
		ModelGeneration.hubbard_chain,
	'hubbard_square_lattice_generalised' : 
		ModelGeneration.hubbard_square_lattice_generalised,
	'hopping_topology' : 
		ModelGeneration.generate_models_hopping_topology, 
}

tree_completed_initial = {
    'PT_Effective_Hamiltonian' : True,
	'NV_centre_experiment_debug' : True, 
    'NV_centre_spin_large_bath' : True,

	None: False
}

experimental_dataset = {
	# strings of paths relative to QMD/ExperimentalSimulations/Data/
    'PT_Effective_Hamiltonian' : 'test_PT_data.p',
	'NV_centre_experiment_debug' : 'NVB_rescale_dataset.p', 
    'reduced_nv_experiment' : 'NVB_rescale_dataset.p',
    'NV_centre_spin_large_bath' : 'NVB_rescale_dataset.p',
	# 'NV_centre_experiment_debug' : 'NV05_dataset.p', 
	'two_qubit_ising_rotation_hyperfine' : 'NVB_rescale_dataset.p', 
	'two_qubit_ising_rotation_hyperfine_transverse' : 'NVB_rescale_dataset.p', 
	None: 'NVB_rescale_dataset.p'
}


measurement_type = {
    'PT_Effective_Hamiltonian' : 'full_access',
    'NV_centre_spin_large_bath' : 'n_qubit_hahn',    
    'reduced_nv_experiment' : 'hahn' ,
	'NV_centre_experiment_debug' : 'hahn', 
	'two_qubit_ising_rotation_hyperfine' : 'hahn', 
	'two_qubit_ising_rotation_hyperfine_transverse' : 'hahn', 
	None: 'full_access'
}


##### ---------- -------------------- #####  
# Functions to check whether the tree being learned is completed. 
# E.g. by checking if it is at maximum depth (spawn_depth).
##### ---------- -------------------- #####  

tree_finished_functions = {
	None : 
		ModelGeneration.spawn_depth_check,
	'heisenberg_nontransverse' : 
		ModelGeneration.max_num_qubits_reached_check,
	'hubbard_chain_just_hopping' :
		ModelGeneration.max_num_qubits_reached_check,
	'hubbard_chain' :
		ModelGeneration.max_num_qubits_reached_check,
	'hubbard_square_lattice_generalised' :
		ModelGeneration.max_num_qubits_reached_check,
	'ising_1d_chain' : 
		ModelGeneration.max_num_qubits_reached_check,
	'heisenberg_xyz' :
		ModelGeneration.max_num_qubits_reached_check

	# 'heisenberg_transverse':
	# 	ModelGeneration.max_num_qubits_reached_check,
}


##### ---------- -------------------- #####  
# Mapping model name strings to branch numbers
##### ---------- -------------------- #####  

name_branch_map_functions = {
	# growth_generation_rule : latex_mapping_function
	None : 
		ModelNames.branch_is_num_dims,
    'PT_Effective_Hamiltonian' : 
    	ModelNames.branch_is_num_dims,
    'NV_centre_spin_large_bath' : 
    	ModelNames.branch_is_num_params,
    'NV_centre_experiment_debug' : 
    	ModelNames.branch_is_num_params,
    'reduced_nv_experiment' : 
    	ModelNames.branch_is_num_params,
	'two_qubit_ising_rotation_hyperfine' : 
		ModelNames.branch_is_num_params, 
	'two_qubit_ising_rotation_hyperfine_transverse' : 
		ModelNames.branch_is_num_params, 
	'test_return_champs' :
		ModelNames.branch_is_num_params,
	'non_interacting_ising' :
		ModelNames.branch_is_num_dims,
	'interacting_nearest_neighbour_ising' : 
		ModelNames.branch_is_num_dims,
	'non_interacting_ising_single_axis':
		ModelNames.branch_is_num_dims,
	'interacing_nn_ising_fixed_axis':
		ModelNames.branch_is_num_dims,
	'deterministic_noninteracting_ising_single_axis' :
		ModelNames.branch_is_num_dims,
	'deterministic_interacting_nn_ising_single_axis' :
	 	ModelNames.branch_is_num_dims,
 	'deterministic_transverse_ising_nn_fixed_axis' :
 		ModelNames.branch_is_num_dims,
	'hubbard_chain_just_hopping' :
		ModelNames.branch_is_num_dims,
	'hubbard_chain' : 
		ModelNames.branch_is_num_dims,
	'hubbard_square_lattice_generalised' : 
		ModelNames.branch_is_num_dims,
	'ising_1d_chain' : 
		ModelNames.branch_is_num_dims,
	'hopping_topology' : 
		ModelNames.branch_is_num_dims
}


##### ---------- -------------------- #####  
# Mapping model name strings to latex representation
##### ---------- -------------------- #####  

latex_naming_functions = {
	# growth_generation_rule : latex_mapping_function

	None : 
		ModelNames.interaction_latex_name,
    'PT_Effective_Hamiltonian' : 
    	ModelNames.latex_name_ising,
    'NV_centre_spin_large_bath' : 
    	ModelNames.large_spin_bath_nv_system_name,
    'NV_centre_experiment_debug' : 
    	ModelNames.latex_name_ising,
    'reduced_nv_experiment' : 
    	ModelNames.latex_name_ising,
	'two_qubit_ising_rotation_hyperfine' : 
		ModelNames.latex_name_ising, 
	'two_qubit_ising_rotation_hyperfine_transverse' : 
		ModelNames.latex_name_ising, 
	'ising_1d_chain' : 
		ModelNames.latex_name_1d_ising_chain,
	'hubbard': 
		ModelNames.hubbard_latex, 
	'hubbard_chain_just_hopping' :	
		ModelNames.hubbard_latex,
	'hubbard_chain' : 
		ModelNames.hubbard_latex,
	'hubbard_square_lattice_generalised' : 
		ModelNames.hubbard_latex,
	'heisenberg_xyz' : 
		ModelNames.latex_name_heisenberg_xyz,
	'hopping_topology' : 
		ModelNames.hubbard_latex
	# None : 
	# 	ModelNames.default_latex_wrapping,
	# 'test_return_champs' :
	# 	ModelNames.default_latex_wrapping,
	# 'interacting_nearest_neighbour_ising' : 
	# 	ModelNames.nearest_neighbour_ising_latex_name
}


##### ---------- -------------------- #####  
# All possible models according to this growth rule
# primarily just for tree plotting
##### ---------- -------------------- #####  

# all_models_functions = {
# 	None : 
# 		ModelNames.ising_terms_rotation_hyperfine,
# 	'two_qubit_ising_rotation_hyperfine' : 
# 		ModelNames.ising_terms_rotation_hyperfine, 
# 	'two_qubit_ising_rotation_hyperfine_transverse' : 
# 		ModelNames.ising_terms_full_list, 
# 	'test_return_champs' :
# 		ModelNames.test_return_champs_ALL_MODELS,
# 	'non_interacting_ising' :
# 		ModelNames.non_interacting_ising_all_names,
# 	'non_interacting_ising_single_axis':
# 		ModelNames.non_interacting_ising_all_names,
# 	'interacting_nearest_neighbour_ising':
# 		ModelNames.interacting_ising_nearest_neighbours_all_names,
# 	'interacing_nn_ising_fixed_axis':
# 		ModelNames.interacting_ising_nearest_neighbours_all_names  
# }


initial_models = {
	None :
		['x', 'y', 'z'],
	'test_changes_to_qmd' :
		# ['a', 's'],
		['x', 'y', 'z'],
    'PT_Effective_Hamiltonian' : 
    	[
    		'zTxPPyTxPPxTiPPxTz'
    	],
    'NV_centre_spin_large_bath' : 
    	[
    		'nv_interaction_x_d3PPPnv_spin_x_d3'
		],
    'reduced_nv_experiment' : 
    	[
    		'xTiPPxTxPPyTiPPyTyPPzTi', 
    		'xTiPPxTxPPyTiPPzTiPPzTz',
    		'xTiPPyTiPPyTyPPzTiPPzTz' 
		],
    'NV_centre_experiment_debug' : 
    	[
    		'xTiPPxTxPPyTiPPyTyPPzTiPPzTz',
    		'xTiPPyTiPPyTyPPzTiPPzTz',
    		'xTiPPxTxPPyTiPPyTyPPyTzPPzTiPPzTz',
		],
	'two_qubit_ising_rotation_hyperfine' :
		['xTi', 'yTi', 'zTi'],
		# ['yTi', 'zTi'],
	'two_qubit_ising_rotation_hyperfine_transverse' : 
		['xTi', 'yTi', 'zTi'],
	'non_interacting_ising' : 
		['x', 'y', 'z'],
	'interacting_nearest_neighbour_ising' :
		['xTx', 'yTy', 'zTz'],
	'hyperfine_like' : 
		['xTi', 'yTi', 'zTi'],
	'interacing_nn_ising_fixed_axis' :
		['xTx', 'yTy', 'zTz'],
	'deterministic_interacting_nn_ising_single_axis' : 
		['xTx', 'yTy', 'zTz'],
	'ising_1d_chain' : 
		[
			'1Dising_ix_d2',
			'1Dising_ix_d2PP1Dising_tz_d2', 
		],

	'heisenberg_nontransverse' :
		['xTx', 'yTy', 'zTz'],
	'heisenberg_transverse' :
		['xTx', 'yTy', 'zTz'],
	'heisenberg_xyz' : 
		[
			'Heis_ix_d2PPHeis_iy_d2PPHeis_iy_d2',
			'Heis_ix_d2PPHeis_iy_d2PPHeis_iy_d2PPHeis_tz_d2'
		],

	'hubbard' :
		['a', 's', 'z'],
	'hubbard' :
		# ['h_1_2_d2', 'h_1_2_d2PPzTiPPiTz'],
		['h_1h2_d2', 'h_1h2_d2PPh_e_d2'],
	'hubbard_chain_just_hopping' :
		['h_1h2_d2'],
	'hubbard_chain' : 
		# ['h_1_2_d2'],
		['h_1h2_d2'],
	'hubbard_square_lattice_generalised' :
		['h_1h2_1h3_2h4_3h4_d4PPPPh_e_d4'],
		# ['h_1h2_d4PPPPh_1h3_d4PPPPh_2h4_d4PPPPh_3h4_d4PPPPh_e_d4']
		# ['h_1_2_d4PPPPh_1_3_d4PPPPh_2_4_d4PPPPh_3_4_d4PPPPzTiTTiTTTiPPPPiTzTTiTTTiPPPPiTiTTzTTTiPPPPiTiTTiTTTz'],

	'hopping_topology' :
		['h_1h2_d2']
}

##### ---------- -------------------- #####  
# Probe dict generation
##### ---------- -------------------- #####  


experimental_probe_dict_generator = {
	None: 
		ProbeGeneration.NV_centre_ising_probes_plus,
		# ProbeGeneration.separable_probe_dict,
    'PT_Effective_Hamiltonian' : 
    	ProbeGeneration.PT_Effective_Hamiltonian_probe_dict,
    'NV_centre_spin_large_bath' : 
    	ProbeGeneration.NV_centre_ising_probes_plus,
    'reduced_nv_experiment' : 
    	ProbeGeneration.NV_centre_ising_probes_plus,
    'NV_centre_experiment_debug' : 
		ProbeGeneration.NV_centre_ising_probes_plus,
	'two_qubit_ising_rotation' : 
		ProbeGeneration.NV_centre_ising_probes_plus,
	'two_qubit_ising_rotation_transverse' : 
		ProbeGeneration.NV_centre_ising_probes_plus,
	'two_qubit_ising_rotation_hyperfine' : 
		ProbeGeneration.NV_centre_ising_probes_plus,
	'two_qubit_ising_rotation_hyperfine_transverse':
		ProbeGeneration.NV_centre_ising_probes_plus,	
}

simulated_probe_dict_generator = {
	None: 
		ProbeGeneration.separable_probe_dict, 
}

special_probe_functions = {
	None: 
		ProbeGeneration.separable_probe_dict,
	'dec_13_exp' : 
		ProbeGeneration.restore_dec_13_probe_generation, 
	'random': 
		ProbeGeneration.separable_probe_dict, 
	'plus':
		ProbeGeneration.plus_probes_dict, 
	'plus_random' :
		ProbeGeneration.NV_centre_ising_probes_plus,
	'ideal' : 
		ProbeGeneration.ideal_probe_dict,
}

##### ---------- -------------------- #####  
# Wrapper functions 
##### ---------- -------------------- #####  


def new_model_generator(generator, **kwargs):
    model_func = model_generation_functions[generator]
    # print("[User funcs] Using model generation function:", model_func)
    kwargs['generator'] = generator
    return model_func(**kwargs)

def expectation_value_wrapper(method, **kwargs):       
    # method means measurement_type flag, provided to QMD
    # print("method:", method)
    expectation_value_function = expec_val_function_dict[method]
    return expectation_value_function(**kwargs)


def max_spawn_depth(generator, log_file):
    if generator not in max_spawn_depth_info:
        log_print(
        	[
        	"Generator not recognised; \
        	does not have maximum spawn depth or \
        	generation function"
        	], 
        	log_file
    	)
    else:
        return max_spawn_depth_info[generator]



def tree_finished(generator, **kwargs):
	try:
		tree_finished_check = tree_finished_functions[generator]
	except:
		tree_finished_check = tree_finished_functions[None]

	tree_completed = tree_finished_check(
		generator = generator, 
		max_spawn_depth_info = max_spawn_depth_info,
		max_num_qubits_info = max_num_qubits_info,
		**kwargs
	)
	# print("[tree finished] func:", tree_finished_check)
	# print("tree completed:", tree_completed)
	return tree_completed 		



def get_latex_name(
    name, 
    growth_generator=None,
    **kwargs
):
	try:
		# if mapping doesn't work, default to just wrap in $__$. 
		latex_mapping = latex_naming_functions[growth_generator]
		latex_representation = latex_mapping(name, **kwargs)
	except:
		
		latex_mapping = latex_naming_functions[None]
		# print("In except of get_latex_name, mapping:", latex_mapping)
		latex_representation = latex_mapping(name, **kwargs)
		
	# print("Latex Mapping used", latex_mapping)

	return latex_representation

def get_qhl_models(generator):
	try:
		return qhl_models_by_generator[generator]
	except:
		return [
			default_true_operators_by_generator[
				generator
			]
		]


def get_all_model_names(
	growth_generator = None,
	**kwargs
):
	all_models = all_models_functions[growth_generator]
	# print("all model function:", all_models)
	return all_models(**kwargs)



def get_name_branch_map(
    latex_mapping_file, 
    growth_generator=None,
    **kwargs
):
	try:
		# if mapping doesn't work, default to just wrap in $__$. 
		branch_mapping = name_branch_map_functions[growth_generator]
		name_branch_map = branch_mapping(
			latex_mapping_file=latex_mapping_file, 
			**kwargs
		)
	except:
		branch_mapping = name_branch_map_functions[None]
		name_branch_map = branch_mapping(
			latex_mapping_file=latex_mapping_file, 
			**kwargs
		)
	# print("Latex Mapping used", branch_mapping)

	return name_branch_map


def get_initial_op_list(
	growth_generator=None, 
	log_file='qmd.log',
	**kwargs
):
	try:
		initial_ops = initial_models[growth_generator]
	except:
		initial_ops = initial_models[None]
		log_print(
			[
			"Initial operators not known for generator", 
			growth_generator, 
			". Defaulting to:", initial_ops
			],
			log_file
		)
	return initial_ops

def get_tree_completed_initial_value(
	growth_generator = None, 
):
	try:
		tree_completed = tree_completed_initial[growth_generator]
	except:
		tree_completed = tree_completed_initial[None]
	return tree_completed


def get_probe_dict(
	experimental_data=False, 
	growth_generator=None, 
	special_probe=None, 
	**kwargs
):
	# print("[getProbeDict] kwargs:", kwargs)
	# print("[getProbeDict] growth_generator:", growth_generator)
	# print("[getProbeDict] exp data:", experimental_data)

	# elif ideal_probes == True:
	# 	probe_dict_function = ProbeGeneration.ideal_probe_dict
	# elif (
	# 	experimental_data == False
	# 	and
	# 	plus_probes == True
	# ):

	# 	probe_dict_function = ProbeGeneration.plus_probes_dict
    # max_num_qubits = 

	if special_probe == 'None':
		special_probe = None
	print("[getProbeDict] Special probe:", special_probe)

	if special_probe is not None:
		try:
			probe_dict_function = special_probe_functions[special_probe]
		except:
			probe_dict_function = special_probe_functions[None]

	elif experimental_data == True:
		try:
			probe_dict_function = experimental_probe_dict_generator[
				growth_generator
			]
		except:
			probe_dict_function = experimental_probe_dict_generator[
				None
			]
	else:
		try:
			probe_dict_function = simulated_probe_dict_generator[
				growth_generator
			]
		except:
			probe_dict_function = simulated_probe_dict_generator[
				None
			]


	try:
		max_num_qubits = max_num_qubits_info[growth_generator]
	except:
		max_num_qubits = max_num_qubits_info[None]

	max_num_qubits = max(
		list(max_num_qubits_info.values())
	) # TODO this isn't strictly necessary. Take the highest of any of the growth rules in use. 

	probe_dict = probe_dict_function(
		max_num_qubits = max_num_qubits, 
		**kwargs
	)
	print("[getProbeDict] Probe func used:", probe_dict_function)
	return probe_dict

def get_experimental_dataset(growth_generator):
	try:
		ds = experimental_dataset[growth_generator]
	except:
		ds = experimental_dataset[None]
	return ds

def get_measurement_type(growth_generator):
	try:
		meas = measurement_type[growth_generator]
	except:
		meas = measurement_type[None]
	return meas


