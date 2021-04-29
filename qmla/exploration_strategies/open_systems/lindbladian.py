import numpy as np
import itertools
import sys
import os
from scipy import linalg
import random
import math

from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.probe_set_generation
from qmla import model_building_utilities

def tensor_expansion(
        binary_matrix,
        operator
        ):
    I = np.identity(2)
   
    # The binary matrix directs the construction of a matrix:
    #   a 1 corresponds to the matrix passed as 'operator'
    #   a 0 corresponds to an identity matrix
    #   the x axis are tensored in order then summed along the y
    for p in range(0,np.shape(binary_matrix)[0]):
        placeholder = np.array([1])
        # This try is incase a line was sent, not a matrix ie. [0,0,1,0]
        # the try deals with the matix case and the except hands lines
        try:
            for q in range(0,np.shape(binary_matrix)[1]):
                # if statement unpacks line in to matrices then tensor prodicts them
                if binary_matrix[p,q] == 0:
                    placeholder = np.kron(placeholder,I)
                else:
                    placeholder = np.kron(placeholder,operator)
            # each line is then summed into fullmatrix
            if p == 0:
                fullmatrix = placeholder
            else:
                fullmatrix = fullmatrix + placeholder
        except:
            for q in binary_matrix:
                if q == 0:
                    placeholder = np.kron(placeholder,I)
                else:
                    placeholder = np.kron(placeholder,operator)
            # no need to sum here so break    
            fullmatrix = placeholder
            break
    return fullmatrix

def liouvillian_evolve_expectation(
    ham,
    t,
    state,
    log_file='QMDLog.log',
    log_identifier='Expecation Value',
    **kwargs
):
    experimental_t_tolerance = 10
    try:
        unitary = linalg.expm(ham * t)
        u_psi = np.dot(unitary, state)
    except:
        print(
            #[
                "Failed to build unitary for ham:\n {}".format(ham)
            #],
            #log_file=log_file, log_identifier=log_identifier
        )
        raise

    N = int(np.sqrt(len(state)))
   
    rho1 = np.zeros((N,N),complex)
    rho2 = np.zeros((N,N),complex)
   
    for c,val in enumerate(state):
        rho1[math.floor(c/N),c%N] = val
       
    for c,val in enumerate(u_psi):
        rho2[math.floor(c/N),c%N] = val  
       
    ex_val_tol = 1e-9
    
    if (np.trace(rho1) > 1 + ex_val_tol
        or
        np.trace(rho1) < 0 - ex_val_tol
    ):
        print('rho1 has a trace: ', np.trace(rho1))
        
    if ((np.trace(rho2) > 1 + ex_val_tol
        or
        np.trace(rho2) < 0 - ex_val_tol)
    ):
        print('rho2 is bad rho2,rho2,ham,t: ', rho2, rho1, ham, t)
           
    #Population -- Operator
    #op = tensor_expansion(np.identity(int(np.log2(len(rho2)))), np.array([[1,0],[0,-1]]))
    #expectation_value_op = 0.5*(1+np.trace(np.dot(rho2,op)))
    #Fidelity -- Between p(0) and p(t)
    expectation_value_fid = np.square(np.trace(linalg.sqrtm(np.dot(linalg.sqrtm(rho2),np.dot(rho1,linalg.sqrtm(rho2))))))
    #Trace Distance -- Between p(0) and p(t)
    #expectation_value_dist = 0.5*(np.trace(linalg.sqrtm(np.square(rho1-rho2))))
    #Expectation Value Choice
    expectation_value = expectation_value_fid
    return expectation_value

def plot_probe(
    max_num_qubits,
    **kwargs
):
        probe_dict = {}
        num_probes = kwargs['num_probes']
        for i in range(num_probes):
            for l in range(1, 1 + max_num_qubits):
                vector_size = np.zeros(l**2)
                vector_size[-1] = 1

                probe_dict[(i,l)] = vector_size
        
        return probe_dict

def random_qubit():
    #Created a vector of values that when squared and summed = 1
    #Vector is 2 long thus can be seen as a normalised qubit
   
        #Initially sets first element of vector
    random_num_1 = random.random()
       
        #Calculates second element from first
    random_num_2 = np.sqrt(1-random_num_1**2)
   
    #Tests that qubit is valid
       
    ex_val_tol = 1e-9
    if random_num_1**2+random_num_2**2 < 1 - ex_val_tol:     #Checks qubit is normal within allowed machine inaccuracies
        print('norm of qubit is not maintained', random_num_1**2+random_num_2**2)
           
            #Re-runs if invalid qubit        
        random_qubit()
    else:
           
            #If valid then builds vector and returns.
        mtrx = np.zeros((1,2))
        mtrx[(0,0)] =random_num_1
        mtrx[(0,1)] = random_num_2
        return mtrx
   
def liouv_separable_probe_dict(     #This may be wrong depending on what scheme of vectorisation we are to use.
    max_num_qubits,
    num_probes,
    **kwargs
):
        #Set paramaters for probes creation
    mixed_state = True
    ex_val_tol = 1e-9
    N = int(np.log2(max_num_qubits))
   
        #Initialises Dictionary
    separable_probes = {}
   
    for qq in range(num_probes):                #Iterates to create correct number of probes
           
            #Calls random_qubit() to get a vector of random values that squared and summed = 1
        state = random_qubit()
       
        for pp in range(1,N+1):                 #Iterates to add additional qubits to increase probe system size
            if pp == 1:

                    #Passing single qubit
                state = state
            else:
                    #Increasing the Hilbert space of state with another random qubit.
                state = np.kron(state,random_qubit())
               
            if mixed_state:                      #Applying alteration if probe is to be mixed.
                mtx = np.diag(np.diag(np.dot(state.T,state)))
               
                #Building Density Matrix from state (MAY NEED CHANGED)
            mtx = np.dot(state.T,state)
           
                #Checks that trace is valid
            if (np.trace(mtx) > 1 + ex_val_tol
                or
                np.trace(mtx) < 0 - ex_val_tol
            ):
                print('The following Matrix does not have a valid trace:', mtx)
               
                #Flattens Densiy Matrix into Superket and returns
            separable_probes[qq,pp*2] = mtx.flatten()  
    return separable_probes

class Lindbladian(
    exploration_strategy.ExplorationStrategy
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        # print("[Exploration Strategies] init nv_spin_experiment_full_tree")
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        #self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.FixedNineEighthsToPowerK
        self.model_constructor = qmla.shared_functionality.model_constructors.LiouvillianModel
        self.true_model = 'HamLiouvillian_lx_1_d1+DissLiouvillian_1A_lz_1_~_1B_ls_1_d1+DissLiouvillian_2A_lx_1_d1'
        self.initial_models = ['HamLiouvillian_lx_1_d1+DissLiouvillian_1A_lz_1_~_1B_ls_1_d1+DissLiouvillian_2A_lx_1_d1']
        self.true_model_terms_params = {
            'HamLiouvillian_lx_1_d1' : 0.66672309902311377,
            'DissLiouvillian_1A_lz_1': 0.12332223344316784,
            'DissLiouvillian_1B_ls_1': 0.03458255527192342,
            'DissLiouvillian_2A_lx_1': 0.25234093640910046     
        }
        #self.gaussian_prior_means_and_widths = {
        #    'HamLiouvillian_lx_1_d1' : (0.3,1),
        #    'DissLiouvillian_1A_lz_1': (0.3,1),      
        #    'DissLiouvillian_1B_ls_1': (0.7,1),
        #    'DissLiouvillian_2A_lx_1': (0.7,1)
        #}
        self.get_expectation_value = liouvillian_evolve_expectation
        self.plot_probes_generation_subroutine = plot_probe
        self.system_probes_generation_subroutine = liouv_separable_probe_dict   

        def generate_true_parameters_alt(self):


            # Dissect true model into separate terms.
            true_model = self.true_model
            true_model_constructor = self.model_constructor(
                # Not a useful object since parameters not set yet
                # -> just use it to get attributes
                name = true_model
            )
            terms = true_model_constructor.terms_names
            latex_terms = [
                true_model_constructor.latex_name_method(name=term) 
                for term in terms
            ]
            true_model_latex = true_model_constructor.latex_name_method(
                name=true_model,
            )
            num_terms = len(terms)

            true_model_terms_params = []
            true_params_dict = {}
            true_params_dict_latex_names = {}

            # Generate true parameters.
            true_prior = self.get_prior(
                model_name = self.true_model,
                log_file = self.log_file, 
                log_identifier = "[ES true param setup]"
            )
            widen_prior_factor = self.true_param_cov_mtx_widen_factor
            old_cov_mtx = true_prior.cov
            new_cov_mtx = old_cov_mtx**(1 / widen_prior_factor)
            true_prior.__setattr__('cov', new_cov_mtx)
            sampled_list = true_prior.sample()

            # Either use randomly sampled parameter, or parameter set in true_model_terms_params
            for i in range(num_terms):
                term = terms[i]
                
                try:
                    # if this term is set in exploration strategy true_model_terms_params,
                    # use that value
                    if 'DissLiouvillian' in term:
                        diss_terms = term.split('_~_')
                        diss_terms[0] = '_'.join(diss_terms[0].split('_')[1:])
                        diss_terms[-1] = '_'.join(diss_terms[-1].split('_')[:-1])
                        for j in diss_terms:
                            term = 'DissLiouvillian_' + j
                            true_param = self.true_model_terms_params[term]
                            true_model_terms_params.append(true_param)
                            true_params_dict[term] = true_param
                            #TODO sort out latex for this type of parameter
                            #true_params_dict_latex_names[latex_terms[i]] = true_param
                    else:
                        true_param = self.true_model_terms_params[term]
                        true_model_terms_params.append(true_param)
                        true_params_dict[terms[i]] = true_param
                        true_params_dict_latex_names[latex_terms[i]] = true_param
                except BaseException:
                    # otherwise, use value sampled from true prior
                    
                    true_param = sampled_list[0][i]
                    true_model_terms_params.append(true_param)
                    true_params_dict[terms[i]] = true_param
                    true_params_dict_latex_names[latex_terms[i]] = true_param
            
            true_param_info = {
                'true_model' : true_model,
                'params_list' : true_model_terms_params, 
                'params_dict' : true_params_dict
            }

            self.log_print([
                "Generating true params; true_param_info:", true_param_info
            ])
            return true_param_info
        #self.generate_true_parameters = generate_true_parameters_alt  #Couldn't get this to work so i edited the original.

        
'''
class GRTest(
        growth_rule.GrowthRule  # inherit from this
    ):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out
   

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
       
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.qinfer_resampler_threshold = 0.5
        self.max_time_to_consider = 100
        # edit time of experimentation
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        # self.qinfer_model_class = qmla.shared_functionality.qinfer_model_interface.QInferLiouvilleExperiment
       
       
        self.exclude_evaluation = True
        self.log_print(["initialising new GR"])
        self.latex_model_naming_function = liouville_latex
        self.expectation_value_function = liouvillian_evolve_expectation
       
        qmla.construct_models.core_operator_dict['a'] = np.array([[0,0],[0,1]])
        qmla.construct_models.core_operator_dict['b'] = np.array([[1,0.3741966805226849],[0.41421356237309503,-1]],complex)
        qmla.construct_models.core_operator_dict['c'] = np.array([[1,0.3741966805226849],[0.41421356237309503,-1]],complex).T
        self.initial_models = ['LiouvillianHam_lx_1_d1+LiouvillianHam_lb_1_d1+LiouvillianDiss_lb_1_d1+LiouvillianDiss_lc_1_d1']
        self.tree_completed_initially = False
        #self.qinfer_resampler_a = 0.98
        #self.qinfer_resampler_threshold = 0.5
        self.true_model = 'LiouvillianHam_lx_1_d1+LiouvillianHam_la_1_d1+LiouvillianDiss_lb_1_d1+LiouvillianDiss_lc_1_d1'  
        #print(self.true_model)
        self.plot_probe_generation_function = plot_probe
        self.true_model = construct_models.alph(self.true_model)
        self.true_model_terms_params = {
            'LiouvillianHam_lx_1_d1' : 0.66322,
            'LiouvillianHam_la_1_d1' : 0.73424,
            'LiouvillianDiss_lb_1_d1': 0.01523,      
            'LiouvillianDiss_lc_1_d1': 0.13678,
        }
       
       
        #qmla.construct_models.compute('LiouvillianHam_lx_1J2_d2+LiouvillianDiss_ls_1_2_d2')
        self.probe_generation_function = liouv_separable_probe_dict   
'''