from __future__ import absolute_import

import qmla.growth_rules as GR

__all__ = [
    'growth_classes',
    'get_growth_generator_class'
]

growth_classes = {
    # Experimental paper growth rules
    'NVExperimentalData':
        GR.NVCentreExperimentalData,
    'ExperimentNVCentre':
        GR.ExperimentNVCentre,
    'ExperimentNVCentreNoTransvereTerms':
        GR.ExperimentNVCentreNoTransvereTerms,
    'ExpAlternativeNV':
        GR.ExpAlternativeNV,
    'NV_alternative_model_2':
        GR.ExpAlternativeNV_second,
    'ExperimentNVCentreVaryTrueModel':
        GR.ExperimentNVCentreVaryTrueModel,
    'ExperimentFullAccessNV':
        GR.ExperimentFullAccessNV,
    'NVLargeSpinBath':
        GR.NVLargeSpinBath,
    'ExperimentReducedNV':
        GR.ExperimentReducedNV,
    'SimulatedNVCentre':
        GR.SimulatedNVCentre,
    'TestSimulatedNVCentre':
        GR.TestSimulatedNVCentre,
    'NVCentreSimulatedShortDynamicsGenticAlgorithm' : 
        GR.NVCentreSimulatedShortDynamicsGenticAlgorithm, 
    'NVCentreExperimentalShortDynamicsGenticAlgorithm':
        GR.NVCentreExperimentalShortDynamicsGenticAlgorithm,
    'NVCentreSimulatedLongDynamicsGenticAlgorithm' :
        GR.NVCentreSimulatedLongDynamicsGenticAlgorithm,
    'NVCentreGenticAlgorithmPrelearnedParameters' : 
        GR.NVCentreGenticAlgorithmPrelearnedParameters, 
    'ExperimentNVCentreNQubits':
        GR.ExperimentNVCentreNQubits,
    'NVCentreRevivals' : 
        GR.NVCentreRevivals,
    'NVCentreRevivalsSimulated' : 
        GR.NVCentreRevivalsSimulated,
    'NVCentreNQubitBath' : 
        GR.NVCentreNQubitBath, 
    'NVCentreRevivalSimulation' : 
        GR.NVCentreRevivalSimulation,

    # Theoretical paper growth rules
    'IsingPredetermined':
        GR.IsingPredetermined,
    'IsingProbabilistic':
        GR.IsingProbabilistic,
    'IsingSharedField':
        GR.IsingSharedField,
    'HeisenbergXYZPredetermined':
        GR.HeisenbergXYZPredetermined,
    'HeisenbergXYZProbabilistic':
        GR.HeisenbergXYZProbabilistic,
    'HeisenbergSharedField':
        GR.HeisenbergSharedField,
    'FermiHubbardPredetermined':
        GR.FermiHubbardPredetermined,
    'FermiHubbardProbabilistic':
        GR.FermiHubbardProbabilistic,
    'LatticeSet':
        GR.LatticeSet,
    'IsingLatticeSet':
        GR.IsingLatticeSet,
    'HeisenbergLatticeSet':
        GR.HeisenbergLatticeSet,
    'FermiHubbardLatticeSet':
        GR.FermiHubbardLatticeSet,

    # Others
    'Demonstration':
        GR.Demonstration,
    'basic_lindbladian':
        GR.Lindbladian,
    'GrowthRule':
        GR.GrowthRule,
    'Presentation':
        GR.PresentationPlotGeneration,
    'Genetic':
        GR.Genetic,
    'GeneticTest':
        GR.GeneticTest,
    'HeisenbergGenetic':
        GR.HeisenbergGenetic,
    'IsingGenetic':
        GR.IsingGenetic,
    'IsingGeneticTest':
        GR.IsingGeneticTest,
    'IsingGeneticSingleLayer':
        GR.IsingGeneticSingleLayer,
    'TestReducedParticlesBayesFactors':
        GR.TestReducedParticlesBayesFactors,
    'TestAllParticlesBayesFactors':
        GR.TestAllParticlesBayesFactors,
}


def get_growth_generator_class(
    growth_generation_rule,
    **kwargs
):
    r"""
    Get an instance of the class specified by the user which implements a Growth Rule.

    Instance of a :class:`~qmla.GrowthRule` (or subclass).
    This is used to specify how QMLA proceeds, in particular by designing the next batch
    of models to test.
    Growth rule is specified by the name passed to implement_qmla in the launch script,
    through the command line flag `growth_rule`. This string is searched for in the
    growth_classes dictionary. New growth rules must be added here so that QMLA can find them.


    :param str growth_generation_rule: string corresponding to a growth rule
    :params **kwargs: arguments required by the growth rule, passed directly
        to the desired growth rule's constructor.
    :return GrowthRule gr: growth rule class instance
    """

    try:
        gr = growth_classes[growth_generation_rule](
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
    except BaseException:
        print(
            "Cannot find growth rule in available rules:",
            growth_generation_rule)
        raise

    return gr
