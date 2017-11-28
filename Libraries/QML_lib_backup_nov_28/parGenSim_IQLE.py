import qinfer as qi
import numpy as np
import scipy as sp


from joblib import Parallel, delayed
import multiprocessing



from Evo import *
from ProbeStates import *

class parGenSim_IQLE(qi.FiniteOutcomeModel):
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

    def __init__(self, oplist, modelparams, probecounter, probelist = None, min_freq=0, solver='scipy', trotter=False):
        self._solver = solver #This is the solver used for time evolution scipy is faster, QuTip can be more precise
        self._oplist = oplist
        self._trotter = trotter
        self._probelist = probelist
        self._probecounter = probecounter
        self._min_freq = min_freq #how min_freq is chosen?
        super(parGenSim_IQLE, self).__init__(self._oplist)
        
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return len(self._oplist)

    # Modelparams is the list of parameters in the System Hamiltonian the ones we want to know
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
        super(parGenSim_IQLE, self).likelihood(
            outcomes, modelparams, expparams
        )
        
        #print('outcomes = ' + repr(outcomes))
        
        #Modelparams is the list of parameters in the System Hamiltonian
        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
            
        cutoff=min(len(modelparams), 5)
        #print('modelparams = ' + repr(modelparams[0:cutoff]))
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
            probestate = choose_probe(self._oplist, np.array([expparams.item(0)[1:]]))
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
            # print('probestate:'+repr(probestate))
            
            
        
        """ Various evolution solvers are listed here: """
        
        if (self._solver == 'scipy'):
            #only for models made by single or commutative operators
            #pr0[:, :] = pr0fromScipy(t, dw, self._oplist, probestate)
            #for all other models
            pr0[:, :] = Parallel(n_jobs=2)
            (delayed(pr0fromScipyNC)(t, modelparams[:,], np.array([expparams.item(0)[1:]]), self._oplist, probestate,trotterize=self._trotter)
            (i ** 2) for i in range(10))
        else:
            if (self._solver == 'qutip'):
                pr0[:, :] = pr0fromQutip(t, dw, self._oplist, probestate)
            else:
                raise ValueError('No solver called "{}" known'.format(self._solver))

        #print("Pr0 = " + str(pr0[0:cutoff]) )
        #print("likelihoods: " + str(qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)))
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)