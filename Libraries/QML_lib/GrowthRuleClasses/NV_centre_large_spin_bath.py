import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues


from NV_centre_full_access import NVCentreSpinFullAccess

class NVCentreLargeSpinBath(
    NVCentreSpinFullAccess # inherit from this
):
    # Uses some of the same functionality as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out
    # and different model generation

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )

        self.true_operator = 'nv_spin_x_d3PPPnv_spin_y_d3PPPnv_spin_z_d3PPPnv_interaction_x_d3PPPnv_interaction_y_d3PPPnv_interaction_z_d3'
        self.initial_models = [
        	'nv_spin_x_d2PPnv_spin_y_d2PPnv_spin_z_d2PPnv_interaction_x_d2PPnv_interaction_y_d2PPnv_interaction_z_d2'
    	] 
        self.qhl_models =    	[
        	'nv_spin_x_d3PPPnv_spin_y_d3PPPnv_spin_z_d3PPPnv_interaction_x_d3PPPnv_interaction_y_d3PPPnv_interaction_z_d3', 
        	'nv_spin_x_d3PPPnv_spin_y_d3PPPnv_spin_z_d3PPPnv_interaction_x_d3PPPnv_interaction_y_d3PPPnv_interaction_z_d3'
        ]
        self.max_num_parameter_estimate = 6
        self.max_spawn_depth = 4
        self.max_num_qubits = 10
        self.tree_completed_initially = False
        self.experimental_dataset = 'NVB_rescale_dataset.p'
        self.measurement_type = 'n_qubit_hahn'
        self.fixed_axis_generator = False
        self.fixed_axis = 'z' # e.g. transverse axis

        self.max_num_models_by_shape = {
            1 : 0,
            'other' : 1
        }

    def expectation_value(
        self, 
        ham,
        t,
        state,
        **kwargs
    ):      
        # print("[Growth Rules - NV] Expectation Values")
        exp_val = ExpectationValues.n_qubit_hahn_evolution(
            ham = ham, 
            t = t, 
            state = state, 
            **kwargs
        )
        return exp_val

    def generate_models(
        self, 
    	model_list, 
    	**kwargs
	):

	    # model_list = kwargs['model_list']
	    spawn_step = kwargs['spawn_step']
	    spawn_stage = kwargs['spawn_stage']    

	    print(
	        "[ModelGeneration.NV_centre_spin_large_bath]",
	        "Spawn stage:", spawn_stage
	    )

	    max_num_qubits = max(
	        [DataBase.get_num_qubits(mod) for mod in model_list]
	    )
	    new_gali_model = gali_model_nv_centre_spin(max_num_qubits + 1)
	    return [new_gali_model]

    def latex_name(
        self, 
        name,
        **kwargs
    ):
        import ModelNames
        latex_name = ModelNames.large_spin_bath_nv_system_name(
        	term = name
        )
        return latex_name

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




## Supporting functions

def gali_model_nv_centre_spin(num_qubits):
    axes = ['x', 'y', 'z']
    p_str = 'P' * num_qubits
    model_terms = []
    for axis in axes:
        for contribution in ['interaction', 'spin']:
            model_terms.append(
                'nv_{}_{}_d{}'.format(contribution, axis, num_qubits)
            )
            
    model = p_str.join(model_terms)
    return model





