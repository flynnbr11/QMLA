from __future__ import print_function # so print doesn't show brackets

import qinfer as qi
import numpy as np
import scipy as sp
import warnings

from Evo import *
from ProbeStates import *
from MemoryTest import print_loc
from psutil import virtual_memory

global_print_loc=True
global debug_print
debug_print = False
global likelihood_dev
likelihood_dev = False

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

    def __init__(self, oplist, modelparams, probecounter, true_oplist = None, num_probes=40, probe_dict=None, trueparams = None, probelist = None, min_freq=0, solver='scipy', trotter=False, qle=True, use_exp_custom=True, enable_sparse=True):
        self._solver = solver #This is the solver used for time evolution scipy is faster, QuTip can handle implicit time dependent likelihoods
        self._oplist = oplist
        self._probecounter = probecounter
        self._a = 0
        self._b = 0 
        self.QLE = qle
        self._trotter = trotter
        self._modelparams = modelparams
        self._true_oplist = true_oplist
        self._trueparams = trueparams
        self.use_exp_custom = use_exp_custom
        self.enable_sparse = enable_sparse
        self._min_freq = min_freq
        if true_oplist is not None and trueparams is None:
            raise(ValueError('\nA system Hamiltonian with unknown parameters was requested'))
        if true_oplist is None:
            warnings.warn("\nI am assuming the Model and System Hamiltonians to be the same", UserWarning)
            self._trueHam = None
        else:
           #self._trueHam = getH(trueparams, true_oplist)
           self._trueHam = np.tensordot(trueparams, true_oplist, axes=1)
           #print("true ham = \n", self._trueHam)
#TODO: changing to try get update working for >1 qubit systems -Brian
        if debug_print: print("Gen sim. True ham has been set as : ")
        if debug_print: print(self._trueHam)
        
        if debug_print:print("True params & true_oplist: [which are passed to getH] ")
        if debug_print:print(trueparams)
        if debug_print:print(true_oplist)
        super(GenSimQMD_IQLE, self).__init__(self._oplist)
        #probestate = choose_randomprobe(self._probelist)
        probestate = def_randomprobe(oplist,modelpars=None)
        self.ProbeState=None
        self.NumProbes = num_probes
        if probe_dict is None: 
            self._probelist = seperable_probe_dict(max_num_qubits=12, num_probes = self.NumProbes) # TODO -- make same as number of qubits in model.
        else:
            self._probelist = probe_dict    
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
        
    def old_likelihood(self, outcomes, modelparams, expparams): ##TODO REPLACING THIS WITH LIKELIHOOD FNC BELOW TO INTRODUCE PARTIAL TRACE
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
        print ("outcomes : ", outcomes)
            
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
        
        
        
        #print('Pr0 from QuTip()' + str(pr0fromQutip(t, dw)))
        #print('Pr0 from SciPy()' + str(pr0fromScipy(t, dw)))

        #print('Pr0 from Likelihood' + str(np.cos(t * dw / 2) ** 2))
        
        
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
        print("Probe state = ", self.ProbeState)    
            
        
        """ Various evolution solvers are listed here: """
        
        if (self._solver == 'scipy'):
            if debug_print: print("In GenSimQMD. Solver = ", self._solver)
            if debug_print: print("trueHam : ")
            if debug_print: print(self._trueHam)
            #only for models made by single or commutative operators
            #pr0[:, :] = pr0fromScipy(t, dw, self._oplist, probestate)
            #for all other models
            if debug_print: print("in Pr0fromScipy, oplist = ", self._oplist)
            pr0[:, :] = pr0fromScipyNC(t, modelparams[:,], np.array([expparams.item(0)[1:]]), self._oplist, probestate, Hp=self._trueHam, trotterize=self._trotter, use_exp_custom=self.use_exp_custom)
        else:
            if (self._solver == 'qutip'):
                pr0[:, :] = pr0fromQutip(t, dw, self._oplist, probestate, Hp=self._trueHam)
            else:
                raise ValueError('No solver called "{}" known'.format(self._solver))

        print("Pr0 = " + str(pr0[0:cutoff]) )
        #print("likelihoods: " + str((qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0))[0:cutoff]  ))
        
        #if likelihood_dev: print("About to enter qi.FiniteOutcomeModel. \npr0 has shape ", np.shape(pr0))
        #if likelihood_dev: print("outcomes has shape ", np.shape(outcomes))
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
        
    def likelihood(self, outcomes, modelparams, expparams):
        super(GenSimQMD_IQLE, self).likelihood(
            outcomes, modelparams, expparams
        )
        print_loc(global_print_loc)
        cutoff=min(len(modelparams), 5)
        self._a += 1
        if self._a % 2 == 1:
            self._b += 1
        print_loc(global_print_loc)
            
        # choose a probe from self._probelist 
        # two indices: 
        # probe index
        # - max qubits in this system
        
        num_particles = modelparams.shape[0]
        num_parameters = modelparams.shape[1]
        print_loc(global_print_loc)
        if  num_particles == 1:
            #print("true evolution with params: ", modelparams[0:cutoff], "\t true params: ", self._trueparams)
            print_loc(global_print_loc)            
            ham_list = [self._trueHam]
            print("Memory used by ham_list : ", sys.getsizeof(ham_list))
            print_loc(global_print_loc)            
            sample = np.array([expparams.item(0)[1:]])[0:num_parameters]
            print_loc(global_print_loc)            
            true_evo = True
        else:
            print_loc(global_print_loc)            
            print("Total memory before assining ham_list: ", virtual_memory().used)
            ham_list = [np.tensordot(params, self._oplist, axes=1) for params in modelparams]
            print("Total memory after assining ham_list: ", virtual_memory().used)
            print("Length of: \nmodelparams ",  len(modelparams), "\noplist : ", len(self._oplist))
            print("Length ham_list : ", len(ham_list), "\t elements has shape : ", ham_list[0].shape)
            print("Single element has size : ", sys.getsizeof(ham_list[0]))
            print("Memory used by ham_list : ", sys.getsizeof(ham_list))
            print("Memory used by ham_list as %: ", 100*(sys.getsizeof(ham_list)/virtual_memory().total))
            print_loc(global_print_loc)            
            sample = np.array([expparams.item(0)[1:]])
            print_loc(global_print_loc)            
            true_evo = False
        print_loc(global_print_loc)
    
        ham_num_qubits = np.log2(ham_list[0].shape[0])
        probe = self._probelist[(self._b % self.NumProbes), ham_num_qubits]
        ham_minus = np.tensordot(sample, self._oplist, axes=1)[0]
        print_loc(global_print_loc)

        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
            
        print_loc(global_print_loc)
            
        times = expparams['t']
        print_loc(global_print_loc)

        if self.QLE is True:
            #print("using QLE function")
            print_loc(global_print_loc)
            pr0 = get_pr0_array_qle(t_list=times, ham_list=ham_list, probe=probe, use_exp_custom=self.use_exp_custom, enable_sparse = self.enable_sparse)    
            print_loc(global_print_loc)
        else: 
            #print("using IQLE function")
            print_loc(global_print_loc)
            pr0 = get_pr0_array_iqle(t_list=times, ham_list=ham_list, ham_minus=ham_minus, probe=probe, use_exp_custom=self.use_exp_custom, enable_sparse = self.enable_sparse)    
            print_loc(global_print_loc)

        print_loc(global_print_loc)
        del ham_list # TODO: can i do this??
#        print("Memory used by ham_list after del: ", sys.getsizeof(ham_list))
        print_loc(global_print_loc)
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)



def seperable_probe_dict(max_num_qubits, num_probes):
    seperable_probes = {}
    for i in range(num_probes):
        seperable_probes[i,0] = random_probe(1)
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = np.tensordot(seperable_probes[i,j-1], random_probe(1), axes=0).flatten(order='c')
            if np.linalg.norm(seperable_probes[i,j]) < 0.999999999 or np.linalg.norm(seperable_probes[i,j]) > 1.0000000000001:
                print("non-unit norm: ", np.linalg.norm(seperable_probes[i,j]))
    return seperable_probes


def random_probe(num_qubits):
    dim = 2**num_qubits
    real = np.random.rand(1,dim)
    imaginary = np.random.rand(1,dim)
    complex_vectors = np.empty([1, dim])
    complex_vectors = real +1.j*imaginary
    norm_factor = np.linalg.norm(complex_vectors)
    probe = complex_vectors/norm_factor
    return probe[0][:]





        
