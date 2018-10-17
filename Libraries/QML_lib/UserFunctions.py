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

##### ---------- -------------------- #####  
# Measurement/Expectation Values
##### ---------- -------------------- #####  

expec_val_function_dict = {
	# measurement_type : expectation_value_function_for_that_meas
    'full_access' : 
    	ExpectationValues.expectation_value, 
    'hahn' : 
    	ExpectationValues.hahn_evolution,
    'trace_all_but_first' : 
    	ExpectationValues.traced_expectation_value_project_one_qubit_plus
}

    
def expectation_value_wrapper(method, **kwargs):       
    # method means measurement_type flag, provided to QMD
    # print("method:", method)
    expectation_value_function = expec_val_function_dict[method]
    return expectation_value_function(**kwargs)

    


##### ---------- -------------------- #####  
# Model Generation
##### ---------- -------------------- #####  

model_generation_functions = {
	# growth_generation_rule : growth_function
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
}

def new_model_generator(generator, **kwargs):
    model_func = model_generation_functions[generator]
    print("[User funcs] Using model generation function:", model_func)
    return model_func(**kwargs)


##### ---------- -------------------- #####  
# Mapping model name strings to latex representation
##### ---------- -------------------- #####  


latex_naming_functions = {
	# growth_generation_rule : latex_mapping_function

	None : 
		ModelNames.default_latex_wrapping,
	'two_qubit_ising_rotation_hyperfine' : 
		ModelNames.latex_name_ising, 
	'two_qubit_ising_rotation_hyperfine_transverse' : 
		ModelNames.latex_name_ising, 
}

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
		latex_representation = latex_mapping(name, **kwargs)
	# print("Latex Mapping used", latex_mapping)

	return latex_representation

##### ---------- -------------------- #####  
# All possible models according to this growth rule
# primarily just for tree plotting
##### ---------- -------------------- #####  


all_models_functions = {
	None : 
		ModelNames.ising_terms_rotation_hyperfine,
	'two_qubit_ising_rotation_hyperfine' : 
		ModelNames.ising_terms_rotation_hyperfine, 
	'two_qubit_ising_rotation_hyperfine_transverse' : 
		ModelNames.ising_terms_full_list, 
}
def get_all_model_names(
	growth_generator = None,
	**kwargs
):
	all_models = all_models_functions[growth_generator]
	# print("all model function:", all_models)
	return all_models(**kwargs)


	
	



