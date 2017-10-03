import qinfer as qi
import numpy as np
import scipy as sp
import warnings
import sys
import os
import importlib as imp
import time as time
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", "..", "Libraries","QML_lib"))
from Evo import *
from ProbeStates import *
from Norms import *
from IOfuncts import *
from EvalLoss import *
from Distrib import *
import ProbeStates as pros
import multiPGH as mpgh
# import GenSimQMD_IQLE as gsi
import Evo as evo

imp.reload(pros)
imp.reload(mpgh)
#imp.reload(gsi)
imp.reload(evo)


global global_use_exp_ham
global_use_exp_ham = True
## Simulated IQLE Class

class GenSimQMD_IQLE(qi.FiniteOutcomeModel):
    r"""
    Describes the free evolution of a single qubit prepared in the
    :param np.array : :math:`\left|+\Psi\rangle` state under 
    a Hamiltonian :math:`H = \omega \sum_i \gamma_i / 2`
    of a set of (Pauli) operators given by 
    :param np.array oplist:  
    using the interactive QLE model proposed by [WGFC13a]_.
    
    :param np.array oplist: Set of operators whose sum defines the evolution Hamiltonian

    :param float min_freq: Minimum value for :math:`\omega` to accept as valid.
        This is used for testing techniques that mitigate the effects of
        degenerate models; there is no "good" reason to ever set this other
        than zero, other than to test with an explicitly broken model.
        
    :param str solver: Which solver to use for the Hamiltonian simulation.
        'scipy' invokes matrix exponentiation (i.e. time-independent evolution)
        -> fast, accurate when applicable
        'qutip' invokes ODE solver (i.e. time-dependent evolution can be also managed approx.)
        -> not invoked by deafult
    """
    
    ## INITIALIZER ##

    def __init__(self, oplist, modelparams, probecounter, true_oplist = None, trueparams = None, probelist = None, min_freq=0, solver='scipy', trotter=False):
        self._solver = solver #This is the solver used for time evolution scipy is faster, QuTip can handle implicit time dependent likelihoods
        self._oplist = oplist
        self._probecounter = probecounter
        self._probelist = probelist
        self._trotter = trotter
        
        self._min_freq = min_freq
        if true_oplist is not None and trueparams is None:
            raise(ValueError('\nA system Hamiltonian with unknown parameters was requested'))
        if true_oplist is None:
            warnings.warn("\nI am assuming the Model and System Hamiltonians to be the same", UserWarning)
            self._trueHam = None
        else:
            self._trueHam = getH(trueparams, true_oplist)
        super(GenSimQMD_IQLE, self).__init__(self._oplist)
        #probestate = choose_randomprobe(self._probelist)
        probestate = def_randomprobe(oplist,modelpars=None)
        self.ProbeState=None
        
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return len(self._oplist)

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
        expnames = [('t', 'float')]
        for exppar in range(self.n_modelparams):
            expnames.append(('w_' + str(exppar+1), 'float'))
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
    
    ## METHODS ##

    def are_models_valid(self, modelparams):
        return np.all(np.abs(modelparams) > self._min_freq, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
        
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        # THIS is for calling likelihood outside of the class
        super(GenSimQMD_IQLE, self).likelihood(
            outcomes, modelparams, expparams
        )
        
        #print('outcomes = ' + repr(outcomes))
        
        #Modelparams is the list of parameters in the System Hamiltonian
        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
            
        cutoff=min(len(modelparams), 5)
        # print('modelparams = ' + repr(modelparams[0:cutoff]))
        #print('expparams = ' + repr(np.array([expparams.item(0)[1:]])))
        
        
        # expparams are the {t, w1, w2,...} guessed parameters, i.e. each element 
        # is a particle with a specific sampled value of the corresponding parameter
        
        t = expparams['t']
        #print('Selected t = ' + repr(t))
        
        
        
        
        ########################
        ########################
        #difference between the parameters true and the parameters sampled
        #dw is used only in qutip...
        dw = modelparams[:,]-expparams.item(0)[1:]
        #print('dw=' + repr(dw[0:cutoff]))
        
        
        # Allocating first, it is useful to make sure that a shape mismatch later
        # will not cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        """ Various probestate options are listed here: """
        
        """maximises evolution changes, but may favour only one parameter
        based upon oplist  alone, non-scalable as it eigensolves the Hamiltonian"""
        if (self._probelist is None):
            probestate = def_randomprobe(self._oplist)
            #probestate = choose_probe(self._oplist, np.array([expparams.item(0)[1:]]))
           # """chooses randomly each time the probe in the space"""
        elif (len(self._probelist)==1):
            probestate = self._probelist[0]
        else:
            """chooses randomly each time the probe among a list"""
            #probestate = choose_randomprobe(self._probelist)
            """chooses a linear combination with random weights out of a list"""           
            #probestate = probes_randomcombo(self._probelist)   
            """ensures that the probestates are chosen sequentially to prevent *jumps* in learning"""
            self._probecounter+=1
            if (self._probecounter >= len(self._probelist)):
                self._probecounter = 0
            probestate = self._probelist[self._probecounter]
            #print('probestate:'+repr(probestate))
        self.ProbeState = probestate
            
            
        
        """ Various evolution solvers are listed here: """
        
        if (self._solver == 'scipy'):
            #only for models made by single or commutative operators
            #pr0[:, :] = pr0fromScipy(t, dw, self._oplist, probestate)
            #for all other models
            pr0[:, :] = pr0fromScipyNC(t, modelparams[:,], np.array([expparams.item(0)[1:]]), self._oplist, probestate, Hp=self._trueHam, trotterize=self._trotter, use_exp_ham=global_use_exp_ham)
        else:
            if (self._solver == 'qutip'):
                pr0[:, :] = pr0fromQutip(t, dw, self._oplist, probestate, Hp=self._trueHam)
            else:
                raise ValueError('No solver called "{}" known'.format(self._solver))

        # print("Pr0 = " + str(pr0[0:cutoff]) )
        # print("likelihoods: " + str(qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)))
        
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
        
## Arrangements for IQLE        
#### Probestates; true parameters and operators


n_particles = 1200
n_experiments = 200
num_tests = 30
n_params = 1

param_1 = 0.65
param_2 = 0.44
param_3 = 0.28

start=time.clock()
for a in range(num_tests):

  if n_params == 1:
    simlist=np.array([evo.sigmay()])
    truelist=np.array([evo.sigmay()])
    mean = [[0.5]]
    sim_sigmas = np.array([[0.2]])
    prior = MultiVariateNormalDistributionNocov(len(simlist), mean=mean, sigmas=sim_sigmas)
    true_params = np.array([[param_1]])

  elif n_params == 2:
    simlist=np.array([evo.sigmay(), evo.sigmax()])
    truelist=np.array([evo.sigmay(), evo.sigmax()])
    mean = [[0.5, 0.5]]
    sim_sigmas = np.array([[0.2, 0.2]])
    prior = MultiVariateNormalDistributionNocov(len(simlist), mean=mean, sigmas=sim_sigmas)
    true_params = np.array([[param_1, param_2]])
  elif n_params == 3:
    simlist=np.array([evo.sigmay(), evo.sigmax(), evo.sigmaz()])
    truelist=np.array([evo.sigmay(), evo.sigmax(), evo.sigmaz()])
    mean = [[0.5, 0.5, 0.5]]
    sim_sigmas = np.array([[0.2, 0.2, 0.2]])
    prior = MultiVariateNormalDistributionNocov(len(simlist), mean=mean, sigmas=sim_sigmas)
    true_params = np.array([[param_1, param_2, param_3]])
  
  """Choosing/Sampling the SIM_parameters"""
  sim_params = true_params
  #sim_params = prior.sample(len(simlist))
  #print('Chosen sim_params: ' + str(sim_params))
  #oplist and modelparams are the operators and parameters (particles values) of the simulator
  #true_oplist and trueparams are the operators and parameters of the real system if blank the function assume them the same.


  """List of possible probelist= ..."""
  probelist = list(map(lambda x: pros.def_randomprobe(truelist), range(15)))
  probestate=pros.def_randomprobe(truelist,true_params)
  """MODIFY HERE to implement your choice"""
  probelist = probelist
  # [probestate]
  # probelist
  # None
  model = GenSimQMD_IQLE(oplist=simlist, modelparams=sim_params, true_oplist = truelist, probelist= probelist, trueparams = true_params, probecounter = 0, solver='scipy', trotter=True) #


  print('Chosen probestate: ' + str(probelist))


  ## Simulation Parameters
  #### Number particles/experiments, etc.

  updater = qi.SMCUpdater(model, n_particles, prior, resample_thresh=0.5, resampler = qi.LiuWestResampler(a=0.95), debug_resampling=True)

  inv_field = [item[0] for item in model.expparams_dtype[1:] ]
  print('Inversion fields are: ' + str(inv_field))
  heuristic = mpgh.multiPGH(updater, simlist, inv_field=inv_field)
  print('Heuristic output:' + repr(heuristic()))

  expparams = np.empty((1, ), dtype=model.expparams_dtype)

  print('Initialization Ready')

  ## Running IQLE 
  # probecounter for the choice of the state
  probecounter = 0

  track_loss = np.empty(n_experiments)
  track_cov = np.empty(n_experiments)
  track_time = np.empty(n_experiments)

  track_particle_locs = np.empty([n_particles, len(true_params[0]), n_experiments])
  track_particle_wght = np.empty([n_ps
  articles, n_experiments])

  if global_use_exp_ham == True: 
    exp_method = 'custom'
  else: 
    exp_method = 'linalg'

  results_directory = 'iqle_outcomes_'+str(n_params)+'_params'
  plot_directory = results_directory+'/'+exp_method+'_plots'

  if not os.path.exists(plot_directory):
      os.makedirs(plot_directory)

  if not os.path.exists(results_directory):
      os.makedirs(results_directory)

  results_file = exp_method+'_'+str(n_particles)+'_particles_'+str(n_experiments)+'_experiments.txt'
  results_filename=results_directory+'/'+results_file
  plot_file = exp_method+'_covariance_and_ql_test_'+str(a)

  if a == 0 :
    with open(results_filename, "a") as myfile:
        myfile.write("True params: "+ str(true_params)+"\n")

  plot_filename = plot_directory+'/'+plot_file

  for idx_experiment in range(n_experiments):
      
      experiment = heuristic()
      #print('Chosen experiment: ' + repr(experiment))
      
      if idx_experiment == 0:
          print('Initial time selected > ' + str(experiment[0][0]))
      
      track_time[idx_experiment] = experiment[0][0]
      
      datum = model.simulate_experiment(sim_params, experiment)
      #print(repr(datum))
      updater.update(datum, experiment)
      heuristic = mpgh.multiPGH(updater, simlist, inv_field=inv_field)
      
      track_cov[idx_experiment] = np.linalg.norm(updater.est_covariance_mtx())
      
      track_particle_locs[:, :, idx_experiment] = updater.particle_locations
      track_particle_wght[:, idx_experiment] = updater.particle_weights


      new_eval = updater.est_mean()
      
      new_loss = eval_loss(model, new_eval, true_params)
      track_loss[idx_experiment] = new_loss[0]
      
      if idx_experiment == n_experiments-1:
          print('Final time selected > ' + str(experiment[0][0]))
      

  with open(results_filename, "a") as myfile:
      myfile.write(str(new_eval)+"\n")


  ## Record losses and plot outcomes 

#  loss = eval_loss(model, new_eval, true_params)
  
  plt.clf()
  plt.semilogy(track_loss, label='Q.L.')
  plt.semilogy(track_cov, label='Cov. Norm.')
  plt.legend()
  plt.xlabel('# of Measurements')
  plt.savefig(plot_filename)
  
  
  for par_idx in range(n_params):
    track = [np.mean(track_particle_locs[:, par_idx, step]) for step in range(n_experiments)]
    plot_to_save = plot_directory + '/param_' + str(par_idx+1) + '_test_'+str(a)
    plt.clf()
    plt.plot(track, marker = 'o')
    if par_idx == 0:
      par_val = param_1
    elif par_idx == 1:
      par_val = param_2
    elif par_idx == 2:
      par_val = param_3
    plt.axhline(y=par_val, xmin=0, xmax=n_experiments, hold=None, color='black')
    plt.savefig(plot_to_save) 

end=time.clock()
time_taken = end-start
time_str = "Time: "+str(time_taken)+"\n"

with open(results_filename, "a") as myfile:
    myfile.write(time_str)

myfile.close()




