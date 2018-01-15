ham_exp_installed = True
try: 
    import hamiltonian_exponentiation as h
except: 
    ham_exp_installed = False
import os as os
import time as time
import qinfer as qi
import numpy as np
import importlib as importlib
import pickle as pkl
import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
from functools import partial
import importlib as imp
import future
import numpy as np
import scipy as sp
from scipy import linalg 
import time
import datetime
import sys, os
import time as time
import matplotlib.mlab as mlab
from decimal import Decimal
dire = os.getcwd()


# initialise an array of random probe states
# TODO: generalise this to vectors of length 2^n... here n=1

global use_linalg_global 
global print_prob_diff
global global_max_diff
global exp_scalar_cutoff
exp_scalar_cutoff = 25
if ham_exp_installed:
    use_linalg_global = False
else: 
    use_linalg_global = True

print_prob_diff = 0
global_max_diff = 0

global num_probes
global probes
global probe_counter
num_probes = 40
probes = np.empty([num_probes,2], dtype=complex)
probe_counter = 0

random_vectors = np.random.rand(num_probes,2)
imaginary = np.random.rand(num_probes,2)

# Want: alpha |0> + beta |1>, with alpha & beta complex. 
# Generate random alpha = alpha_real + j.alpha_imag
# normalise those 
# place in probes array

complex_vectors = np.empty([num_probes, 2])
complex_vectors = random_vectors +1.j*imaginary
alpha = complex_vectors[:,0]
beta = complex_vectors[:,1]
norm = np.empty([num_probes])
for l in range(0, num_probes):
    norm[l] = np.sqrt(np.abs(alpha[l])**2 + np.abs(beta[l])**2)
    normalised_alpha = alpha[l]/norm[l] 
    normalised_beta  = beta[l]/norm[l]
    probes[l,0] = normalised_alpha
    probes[l,1] = normalised_beta
    

def eval_loss(
        model, est_mean, true_mps=None,
        true_model=None, true_prior=None
    ):
    
    if true_model is None:
        true_model = model

    if true_mps is None:
        true_mps = true_model.update_timestep(
            promote_dims_left(true_mps, 2), expparams
        )[:, :, 0]

    if model.n_modelparams != true_model.n_modelparams:
        raise RuntimeError("The number of Parameters in True and Simulated model are different.")
                           
    n_pars = model.n_modelparams

    delta = np.subtract(*qi.perf_testing.shorten_right(est_mean, true_mps))
    loss = np.dot(delta**2, model.Q[-n_pars:])

    return loss

def get_prob(t, op_list, param_list, use_linalg=1):
    #use_linalg=1
    compare_custom_linalg = False
    prec = 1e-25
    
    probe_preiodicity = 50
    num_terms = len(param_list)
    hamiltonian = 0
    for i in range(num_terms):
        hamiltonian += param_list[i]*op_list[i,:,:]
        # would have to take exponentials here
        # write new fnc: hamiltonian_build_and_exp
        # - take op_list and param_list, compute e_{-iH1 dt}^n * ... * e_{-iHm dt}^n

    normalised_probe = np.array([probes[probe_counter%num_probes]])
    probe_bra = normalised_probe
    probe_ket = np.transpose(normalised_probe)

    if compare_custom_linalg: 
      linalg = sp.linalg.expm(-1.j*hamiltonian*t)
      custom = h.exp_ham_sparse(hamiltonian, t, plus_or_minus=-1.0, precision=prec, scalar_cutoff=exp_scalar_cutoff, print_method=False)
      print("Matrix diff: %.1E mtx" %np.max(np.abs(linalg-custom)))

      lin_u_probe = np.dot(linalg, probe_ket) # perform unitary matrix of hamiltonian on ket form of probe state
      probe_u_probe = np.dot(probe_bra, lin_u_probe) # multiply be bra form of probe
      lin_probability = abs(probe_u_probe)**2

      cust_u_probe = np.dot(custom, probe_ket) # perform unitary matrix of hamiltonian on ket form of probe state
      probe_u_probe = np.dot(probe_bra, cust_u_probe) # multiply be bra form of probe
      cust_probability = abs(probe_u_probe)**2
      
      if(np.abs(lin_probability - cust_probability) > 1e-5):
        print ("----------- High diff -----------")
      print("Diff %.1E pr" %np.abs(lin_probability - cust_probability))
      if use_linalg == 1:    
        probability = lin_probability
      else: 
        probability = cust_probability

    else:       
      if use_linalg == 1:    
          mtx = (-1.j)*hamiltonian*t
          unitary_mtx = sp.linalg.expm(mtx)
      else: 
#          unitary_mtx = h.exp_ham(hamiltonian, t, plus_or_minus=-1.0, precision=prec, scalar_cutoff=exp_scalar_cutoff, print_method=False)
          unitary_mtx = h.exp_ham_sparse(hamiltonian, t, plus_or_minus=-1.0, precision=prec, scalar_cutoff=exp_scalar_cutoff, print_method=False)

      
#      normalised_probe = np.array([probes[probe_counter%num_probes]])
#      probe_bra = normalised_probe
#      probe_ket = np.transpose(normalised_probe)
      u_probe = np.dot(unitary_mtx, probe_ket) # perform unitary matrix of hamiltonian on ket form of probe state
      probe_u_probe = np.dot(probe_bra, u_probe) # multiply be bra form of probe
      probability = abs(probe_u_probe)**2

    return float(probability)


class tHeurist(qi.Heuristic):
    
    def identity(arg): return arg
    
    def __init__(self, updater, t_field='ts',
                 t_func=identity,
                 maxiters=20
                 ):
        super(tHeurist, self).__init__(updater)
        self._t = t_field
        self._t_func = t_func
        self._maxiters = maxiters
        
    def __call__(self):
        idx_iter = 0
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :] 
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError("PGH did not find distinct particles in {} iterations.".format(self._maxiters))
            
#        eps = np.empty((1,), dtype=self._updater.model.expparams_dtype)
        eps = np.zeros((1,), dtype=self._updater.model.expparams_dtype)
        eps[self._t]  = self._t_func(1 / self._updater.model.distance(x, xp))
        # this does not set non-time fields to anything in eps.... do we ever use those fields?
        return eps

class simulated_QLE(qi.FiniteOutcomeModel):
    
    def __init__(self, op_list, param_list): 
        self._param_list = param_list # Multiplicative parameters for operators
        self._op_list = op_list # Operators to form Hamiltonian 
        # Define Hamiltonian from operators and parameters list
        self._num_terms = len(self._param_list[0])
        super(simulated_QLE, self).__init__(self._op_list, self._param_list)
    
    ## Properties
    @property    
    def n_modelparams(self):
        return self._num_terms
    
    # Modelparams is the list of parameters in the System Hamiltonian - the ones we want to know
    # Possibly add a second axis to modelparams.    
    @property
    def modelparam_names(self):
        modnames = ['w0']
        for modpar in range(self.n_modelparams-1):
            modnames.append('w' + str(modpar+1))
        return modnames

    # expparams are the {t, w1, w2, ...} guessed parameters, i.e. each element 
    # is a particle with a specific sampled value of the corresponding parameter
  
    @property
    def expparams_dtype(self):
        expnames = [('ts', 'float')]
        for exppar in range(self.n_modelparams):
            expnames.append(('w' + str(exppar+1), 'float'))
        return expnames
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True

    ## Functions which are required to inherit from  FiniteOutcomeModel
    ## Currently filler to avoid errors. Will fill in as needed. 
    def are_models_valid(self, modelparams): 
        ## TODO: robust method here
#         w0, w1 = modelparams.T
#         this_thing= np.all(
#             [np.logical_and(w0 > 0, w0 <= 1),
#             np.logical_and(w1 >= -1, w1 <= 1)],
#             axis=0)
#         return this_thing
       
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)        

    def n_outcomes(self, exp_params):
        return 2
        
    def likelihood(self, outcome, modelparams, expparams):
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
        max_pr_diff=0
        t = expparams['ts']
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0])) 
        for idx_row in range(0, modelparams.shape[0]):
            for idx_col in range(0, expparams.shape[0]):
              if print_prob_diff == 1:
                pr_custom = get_prob(t=t, op_list= self._op_list, param_list = modelparams[idx_row], use_linalg=0)
                pr_linalg = get_prob(t=t, op_list= self._op_list, param_list = modelparams[idx_row], use_linalg=1)
                #print("Diff in prob: ", pr_custom - pr_linalg)
                if(np.abs(pr_custom - pr_linalg) > max_pr_diff):
                  max_pr_diff = pr_custom-pr_linalg
              pr = get_prob(t=t, op_list= self._op_list, param_list = modelparams[idx_row], use_linalg=use_linalg_global)
              pr0[idx_row, idx_col] = pr
                #print("diff bw custom and linalg probs: ", np.abs(pr_custom-pr_linalg))
#                pr0[idx_row, idx_col] = get_prob(t=t, op_list= self._op_list, param_list = modelparams[idx_row])
        if print_prob_diff == 1:
          print("Time = ",t)
          print("Max pr diff = ", max_pr_diff)
        likelihood = qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcome, pr0)
        return likelihood    


def run_qle(param_list, op_list, n_particles, n_experiments, resample_thresh, resample_a): 
    # set mean and covariance based on true parameter list
    mean = np.zeros((len(param_list[0])))
    cov  = np.zeros((len(param_list[0]), len(param_list[0])))
    for a in range(len(param_list[0])):
        mean[a] = param_list[0,a] + 0.08
        cov[a,a] = 0.2
#         mean[a] = param_list[0,a] + 0.15
#         cov[a,a] = param_list[0,a] + 0.11
    prior = qi.MultivariateNormalDistribution(mean, cov)
    model = simulated_QLE(op_list=op_list, param_list=param_list)
    param_names = model.modelparam_names
    updater = qi.SMCUpdater(model, n_particles, prior, resample_thresh=resample_thresh, resample_a=resample_a)
    heuristic = tHeurist(updater, t_field='ts')
    track_parameters = np.zeros((len(param_list[0]), n_experiments))
    track_locations = np.zeros([n_particles, len(param_list[0]), n_experiments])
    track_weights = np.empty([n_particles, n_experiments])
    track_loss = np.empty(n_experiments)
    track_time = np.empty(n_experiments)
    resample_points=list()
    resample_points.append(0)
    
    probe_counter=0
    for idx in range(n_experiments):
        probe_counter+=1
        experiment=heuristic()
        datum = model.simulate_experiment(param_list, experiment)
        updater.update(datum, experiment)
        new_eval = updater.est_mean()
        for param in range(len(param_list[0])):
            track_parameters[param, idx] = new_eval[param]
        new_loss = eval_loss(model, new_eval, param_list)
        track_loss[idx] = new_loss[0]
        track_locations[:, :, idx] = updater.particle_locations
        track_weights[:, idx] = updater.particle_weights
        track_time[idx] = experiment[0][0] # time given by heuristic
        if updater.just_resampled is True:
            resample_points.append(idx)
    return track_parameters, track_loss, track_locations, track_weights, track_time, param_names, resample_points

# to run QLE
# set n_particles, n_experiments, op_list, param_list

#n_experiments=200

identity= np.array([[1+0.j, 0+0.j], [0+0.j, 1+0.j]])
 
sigmaz = np.array([[1+0.j, 0+0.j], [0+0.j, -1+0.j]])
 
sigmax = np.array([[0+0.j, 1+0.j], [1+0.j, 0+0.j]])
 
sigmay = np.array([[0+0.j, 0-1.j], [0+1.j, 0+0.j]])


op_list=np.array([sigmax])
param_list = np.array([[0.13]])
#op_list=np.array([evo.sigmax(), evo.sigmay()])
#param_list = np.array([[0.13, 0.66]])
#op_list=np.array([evo.sigmax(), evo.sigmay(), evo.sigmaz()])
#param_list = np.array([[-0.41779883, 0.6153639, 1.27090946]])

print("param list: ", param_list)


#op_list=np.array([evo.sigmax(), evo.sigmaz(), evo.sigmay()])
#param_list = np.array([[0.44, 0.86, 0.5]])


n_par = np.shape(param_list)[1]
n_particles=60
n_experiments=35
num_tests=5
parameter_values=np.empty([n_par, num_tests])

directory = 'qle_plots/test_new_install'
if not os.path.exists(directory):
    os.makedirs(directory)

for a in range(num_tests):
  print("a=", a)
  # For loop to calculate quadratic loss for ranges of resampling threshold & a. 
  qle_results, loss, locations, weights, times, names, resample_points = \
      run_qle(param_list=param_list, op_list=op_list, n_particles=n_particles, \
              n_experiments=n_experiments, resample_thresh=0.5, resample_a=None)
      
  num_params = np.shape(qle_results)[0]
  num_exps = np.shape(qle_results)[1]
  colours = np.array(['r','g','b','k','y'])


  if use_linalg_global == 1:
	    plot_name = directory+'/plot_linalg_' + str(n_particles)+ '_part_' + str(n_experiments) + '_exp_' + str(n_par) +'_params_'+str(a)+'.png'
	    csv_filename = directory+'/parameters_linalg_'+str(n_par)+'_params_'+str(num_tests)+'_tests_' + str(n_particles)+ '_part_' + str(n_experiments) + '_exps.csv'
  else: 
	    plot_name = directory+'/plot_custom_' + str(n_particles)+ '_part_' + str(n_experiments) + '_exp_' + str(n_par) +'_params_'+str(a)+'.png'
	    csv_filename = directory+'/parameters_custom_'+str(n_par)+'_params_'+str(num_tests)+'_tests_' + str(n_particles)+ '_part_' + str(n_experiments) + '_exps.csv'

  plt.clf()
  for i in range(num_params):
      print('i=', i)
      print('True value for ', names[i], ' = ', param_list[0,i])
      print('Estimated value for ', names[i], ' = ', qle_results[i, num_exps -1], '\n')    
      plt.axhline(y=param_list[0,i], xmin=0, xmax=n_experiments, hold=None, color=colours[i%len(colours)])
      plt.plot(qle_results[i], label=names[i],color=colours[i])
      plt.legend(loc='center right')
      plt.title('Hamiltonian Expn fnc')
      plt.xlabel('Experiment Number')
      plt.ylabel(names[i])
      plt.savefig(plot_name)
      parameter_values[i][a] = (qle_results[i, num_exps -1])
          
    
#plt.savefig('improvement_'+str(low)+'_to_'+str(high)+'_qubits.png')
np.savetxt(csv_filename, parameter_values, delimiter=",")
  
    

