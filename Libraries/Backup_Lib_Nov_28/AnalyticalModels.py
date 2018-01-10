import qinfer as qi
import numpy as np

class ChirpInvModel(qi.FiniteOutcomeModel, qi.DifferentiableModel):
    r"""
    Properly includes a Chirping in the Likelihood for the model
    """
    
    ## INITIALIZER ##

    def __init__(self, min_freq=-1):
        super(ChirpInvModel, self).__init__()
        self._min_freq = min_freq

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 2
    
    @property
    def modelparam_names(self):
        return ['w', 'a']
    
    @property
    def modelparam_dtype(self):
        return [('w', 'float'), ('a', 'float')]
        
    @property
    def expparams_dtype(self):
        return [('ts', 'float'), ('w_', 'float'), ('a_', 'float')]
    
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
        w, a = modelparams.T
        return np.all(
            [np.logical_and(w > 0, w <= 1),
            np.logical_and(a >= -1, a <= 1)],
            axis=0)
    
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
        # Customized here.
        super(ChirpInvModel, self).likelihood(
            outcomes, modelparams, expparams
        )
        
        #print('outcomes = ' + repr(outcomes))

        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
            
        #print('modelparams = ' + repr(modelparams))
        #print('expparams = ' + repr(expparams))
        #print('m = ' + str(modelparams[0:1]))
        #print('w_ = ' + str(expparams[0:1]))
            
        t = expparams['ts']
        #dw = modelparams['w'] - expparams['w_']
        dw = modelparams[:,]-expparams.item(0)[1:]
        
        #print('dW = ' + repr(dw[0:1]))
        #print('dw = ' + repr(dw[0:1, 0, np.newaxis]))
        #print('da = ' + repr(dw[0:1, 1, np.newaxis]))
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        #print(np.shape(pr0))
        pr0[:, :] = (np.cos(t / 2 * 
                          ( dw[:, 0, np.newaxis] + (t * dw[:, 1, np.newaxis] / 2) )
                          )) ** 2
        
        #print("Pr0 = " + str(pr0) )
        
        # Now we concatenate over outcomes.
        #print("likelihoods: " + str(qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)))
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

    def score(self, outcomes, modelparams, expparams, return_L=False):
        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]
        
            
        t = expparams['ts']
        #dw = modelparams['w'] - expparams['w_']
        dw = modelparams[:,]-expparams.item(0)[1:]

        outcomes = outcomes.reshape((outcomes.shape[0], 1, 1))

        arg = t / 2 * ( dw[:, 0, np.newaxis] + (t * dw[:, 1, np.newaxis] / 2) )
        #arg = t / 2 * ( dw[:, 0, np.newaxis] )
        
        q = (
            np.power( t / np.tan(arg), outcomes) *
            np.power(-t * np.tan(arg), 1 - outcomes)
        )[np.newaxis, ...]

        assert q.ndim == 4
        
        
        if return_L:
            return q, self.likelihood(outcomes, modelparams, expparams)
        else:
            return q
