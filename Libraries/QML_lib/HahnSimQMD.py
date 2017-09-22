import qinfer as qi
import numpy as np
import scipy as sp
import warnings

from Evo import *
from ProbeStates import *

     
        
class HahnSimQMD(qi.FiniteOutcomeModel):
    r"""
    Attempts simulation of Hahn echo experiments
    """
    
    ## INITIALIZER ##

    def __init__(self, oplist, modelparams, probecounter = None, true_oplist = None, trueparams = None, probelist = None, min_freq=-1000, trotter=False, IQLE=True, datasource = 'sim'):
        
        self._oplist = oplist
        self._probelist = probelist
        self._trotter = trotter
        self._IQLE = IQLE
        self._datasource = datasource
        
        self._min_freq = min_freq
        if true_oplist is not None and trueparams is None:
            raise(ValueError('\nA system Hamiltonian with unknown parameters was requested'))
        if true_oplist is None:
            warnings.warn("\nI am assuming the Model and System Hamiltonians to be the same", UserWarning)
            self._trueHam = None
        else:
            self._trueHam = getH(trueparams, true_oplist)
        super(HahnSimQMD, self).__init__(self._oplist)
        

        
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
        super(HahnSimQMD, self).likelihood(
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
            
            
        
        """ Computing the likelihoods here: """
        
        if self._datasource == 'sim':
            pr0[:, :] = pr0fromHahnPeak(t, modelparams[:,], np.array([expparams.item(0)[1:]]), self._oplist, probestate, Hp=self._trueHam, trotterize=self._trotter, IQLE=self._IQLE)
        
        elif self._datasource == 'offline':
            pr0[:, :] = EXPOFFpr0fromHahnPeak(t, modelparams[:,], np.array([expparams.item(0)[1:]]), self._oplist, probestate, Hp=self._trueHam, trotterize=self._trotter, IQLE=self._IQLE)
        
        

        # print("Pr0 = " + str(pr0[0:cutoff]) )
        # print("likelihoods: " + str(qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)))
        
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
        
        
        
class HahnSignalAnaSimQMD(qi.FiniteOutcomeModel):
    r"""
    Attempts simulation of Hahn echo experiments
    """
    
    ## INITIALIZER ##

    def __init__(self, modelparams, Nqubit_interact = 25, bound_params = None, IQLE=True, datasource = 'sim'):
        
        self._modelparams = modelparams
        self.Nqubit_interact = Nqubit_interact
        self._bound_params = bound_params
        
        self._IQLE = IQLE
        self._datasource = datasource

        super(HahnSignalAnaSimQMD, self).__init__(self._modelparams)
        

        
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return len(self._modelparams[0])
        

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
        """
        Checks against self._bound_params if the modelparams chosen are acceptable
        """
        return np.logical_and(np.all(modelparams > self._bound_params[:,0], axis=1) , np.all(modelparams < self._bound_params[:,1], axis=1))
    
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
        super(HahnSignalAnaSimQMD, self).likelihood(
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

        
        """ Computing the likelihoods here: """
        
        if self._datasource == 'sim':
            pr0[:, :] = pr0fromHahnAnaSignal(t, modelparams[:,], np.array([expparams.item(0)[1:]]), self.Nqubit_interact, IQLE=self._IQLE)
        
        elif self._datasource == 'offline':
            pr0[:, :] = EXPOFFpr0fromHahnSignal(t, modelparams[:,], np.array([expparams.item(0)[1:]]), self.Nqubit_interact, IQLE=self._IQLE)
        

        # print("Pr0 = " + str(pr0[0:cutoff]) )
        # print("likelihoods: " + str(qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)))
        
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
        
    def simulate_majo_experiment(self, modelparams, expparams, repeat=1):
        # Call the superclass simulate_experiment, not recording the result.
        # This is used to count simulation calls.
        super(HahnSignalAnaSimQMD, self).simulate_experiment(modelparams, expparams, repeat)
        
        if self.is_n_outcomes_constant:
            # In this case, all expparams have the same domain
            all_outcomes = self.domain(None).values
            probabilities = self.likelihood(all_outcomes, modelparams, expparams)
            cdf = np.cumsum(probabilities, axis=0)
            majonum = np.repeat(0.5, repeat*modelparams.shape[0]*expparams.shape[0]).reshape(repeat,1,modelparams.shape[0],expparams.shape[0])
            outcome_idxs = all_outcomes[np.argmax(cdf > majonum, axis=1)]
            outcomes = all_outcomes[outcome_idxs]
        else:
            # Loop over each experiment, sadly.
            # Assume all domains have the same dtype
            assert(self.are_expparam_dtypes_consistent(expparams))
            dtype = self.domain(expparams[0, np.newaxis])[0].dtype
            outcomes = np.empty((repeat, modelparams.shape[0], expparams.shape[0]), dtype=dtype)
            for idx_experiment, single_expparams in enumerate(expparams[:, np.newaxis]):
                all_outcomes = self.domain(single_expparams).values
                probabilities = self.likelihood(all_outcomes, modelparams, single_expparams)
                cdf = np.cumsum(probabilities, axis=0)[..., 0]
                randnum = np.random.random((repeat, 1, modelparams.shape[0]))
                majonum = np.repeat(0.5, repeat*modelparams.shape[0]).reshape(repeat,1,modelparams.shape[0])
                outcomes[:, :, idx_experiment] = all_outcomes[np.argmax(cdf > majonum, axis=1)]
                
        return outcomes[0, 0, 0] if repeat == 1 and expparams.shape[0] == 1 and modelparams.shape[0] == 1 else outcomes