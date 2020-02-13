from __future__ import absolute_import

import qmla.growth_rules as GR

__all__ = [
    'get_growth_generator_class'
]

growth_classes = {
    # Experimental paper growth rules
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
    'ExpNVRevivals':
        GR.ExpNVRevivals,
    'ExperimentFullAccessNV':
        GR.ExperimentFullAccessNV,
    'NVLargeSpinBath':
        GR.NVLargeSpinBath,
    'ExperimentReducedNV':
        GR.ExperimentReducedNV,
    # Theoretical paper growth rules
    'IsingPredetermined':
        GR.IsingPredetermined,
    'IsingProbabilistic':
        GR.IsingProbabilistic,
    'HeisenbergXYZPredetermined':
        GR.HeisenbergXYZPredetermined,
    'HeisenbergXYZProbabilistic':
        GR.HeisenbergXYZProbabilistic,
    'FermiHubbardPredetermined':
        GR.FermiHubbardPredetermined,
    'FermiHubbardProbabilistic':
        GR.FermiHubbardProbabilistic,
    # Others
    'basic_lindbladian':
        GR.Lindbladian,
    'example':
        GR.GrowthRuleTemplate,
    'Presentation':
        GR.PresentationPlotGeneration,
    'Genetic':
        GR.Genetic
}


def get_growth_generator_class(
    growth_generation_rule,
    **kwargs
):
    # TODO: note that in most places, this is called with use_experimental_data.
    # in some plotting functions this is not known, but it should not matter unless
    # called to get probes etc.

    try:
        log_file = kwargs['log_file']
    except:
        log_file = '.default_qmd.log'

    try:
        gr = growth_classes[growth_generation_rule](
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
    except BaseException:
        print("Cannot find growth rule in available rules:", growth_generation_rule)
        raise

    return gr
