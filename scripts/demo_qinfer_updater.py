import numpy as np
from copy import copy

from qinfer import (
    SMCUpdater,
    FiniteOutcomeModel,
    MultivariateNormalDistribution,
    UniformDistribution,
)


def compute_prob_measuring_zero(particle, time):
    r"""
    Function to compute probability of measuring 0 in the gap experiment
    """
    return (2 / 3) * (1 + 0.5 * np.cos(particle * time))


class DemoModel(FiniteOutcomeModel):
    def __init__(self, **kwargs):
        super(DemoModel).__init__(**kwargs)

    def are_models_valid(self):
        return True

    @property
    def expparams_dtype(self):
        return [("t", "float")]

    @property
    def n_modelparams(self):
        return 1

    @property
    def n_outcomes(self):
        return 2

    def likelihood(self, outcomes, modelparams, expparams):

        # modelparams give particles
        # outcomes should be a length 1 list with a single outcome for the single experiment performed

        time = expparams["t"]
        print(
            f"Likelihood fnc given \nOutcomes:\n {outcomes} \nParticles:\n {modelparams} \nExperiment:\n {expparams}"
        )

        prob_measuring_zero = compute_prob_measuring_zero(
            modelparams, time=expparams["t"]  # particles
        )

        likelihood = FiniteOutcomeModel.pr0_to_likelihood_array(
            outcomes=outcomes,
            pr0=prob_measuring_zero,
        )

        return likelihood


# Make instance of our model and generate an SMCUpdater from it
demo_model = DemoModel()
uniform_distribution = UniformDistribution(ranges=[0, 1])

updater = SMCUpdater(model=demo_model, n_particles=3, prior=uniform_distribution)

# Track particles to inspect before/after update
track_particles = [copy(updater.particle_locations)]
track_weights = [copy(updater.particle_weights)]

# Design experiment
experiment = np.empty((1,), dtype=demo_model.expparams_dtype)
experiment["t"] = 7.69

# Update distribution
updater.update(outcome=0, expparams=experiment)
track_particles.append(updater.particle_locations)
track_weights.append(updater.particle_weights)


# Within update, it calls hypothetical_update - we can inspect its outputs here
hypothetical_weights, likelihood_array = updater.hypothetical_update(
    outcomes=np.array([0]),
    expparams=experiment,
    return_likelihood=True,
)
likelihood_array = likelihood_array.reshape(
    3,
)


# Inspect particle calculations

normalisation_factor = np.sum(likelihood_array * track_weights[0])

# check the calculation of norm factor is correct
if not np.isclose(normalisation_factor, updater.normalization_record[0][0]):
    print("Normalisation record is not the same as expected")

for particle_idx in range(3):

    likelihood = likelihood_array[particle_idx]
    weight_before = track_weights[0][particle_idx]
    weight_after = track_weights[1][particle_idx]

    check_correct = np.isclose(
        weight_after,
        (weight_before * likelihood) / normalisation_factor,
        atol=1e-8,  # due to printing accuracy used in likelihood
    )

    if not check_correct:
        print(f"Calculation not the same at particle idx {particle_idx}")
