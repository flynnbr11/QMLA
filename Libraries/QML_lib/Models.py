import qinfer as qi
import numpy as np

class ChirpModel(qi.FiniteOutcomeModel):
    
    @property
    def n_modelparams(self):
        return 2
    
    @property
    def is_n_outcomes_constant(self):
        return True
    def n_outcomes(self, expparams):
        return 2
    
    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)
    
    @property
    def expparams_dtype(self):
        return [('ts', 'float', 1)]
    
    def likelihood(self, outcomes, modelparams, expparams):
        # We first call the superclass method, which basically
        # just makes sure that call count diagnostics are properly
        # logged.
        super(ChirpModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Next, since we have a two-outcome model, everything is defined by
        # Pr(0 | modelparams; expparams), so we find the probability of 0
        # for each model and each experiment.

        pr0 = (np.cos(
                (modelparams[:, np.newaxis, 0] + (modelparams[:, np.newaxis, 1] * expparams['ts'] / 2)) *  
                expparams['ts']
            )) ** 2 # square each element
        
        # Now we use pr0_to_likelihood_array to turn this two index array
        # above into the form expected by SMCUpdater and other consumers
        # of likelihood().
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)