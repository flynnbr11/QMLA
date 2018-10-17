from ExpectationValues import *
import ModelGeneration 
import ModelNames 


##### ---------- -------------------- #####  
"""
Wrapper function for expectation value, relying on above defined functions
"""
expec_val_function_dict = {
    'full_access' : ExpectationValues.expectation_value, 
    'hahn' : ExpectationValues.hahn_evolution,
    'trace_all_but_first' : ExpectationValues.traced_expectation_value_project_one_qubit_plus
}

    
def expectation_value_wrapper(method, **kwargs):       
    # print("method:", method)
    expectation_value_function = expec_val_function_dict[method]
    return expectation_value_function(**kwargs)

    
