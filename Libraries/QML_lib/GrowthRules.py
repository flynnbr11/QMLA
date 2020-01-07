import sys
import os

this_dir = str(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append((os.path.join(this_dir, "GrowthRuleClasses")))
sys.path.append(os.path.join(".."))
sys.path.append(os.path.join(this_dir, "..", "GrowthRuleClasses"))

import SuperClassGrowthRule
import NVCentreExperimentGrowthRules
import NVExperimentVaryTrueModel
import NVCentreFullAccess
import NVCentreLargeSpinBath
import NVCentreExperimentWithoutTransverseTerms
import NVCentreRevivals
import NVCentreAlternativeTrueModel
import NVGrowByFitness
import NVProbabilistic
import ReducedNVExperiment
import PresentationPlotGeneration
import ConnectedLattice
import Ising
import HeisenbergXYZ
import FermiHubbard
import BasicLindbladian
import ExampleGrowthRule

growth_classes = {
    # Experimental paper growth rules
    'two_qubit_ising_rotation_hyperfine_transverse' : 
        NVCentreExperimentGrowthRules.nv_centre_spin_experimental_method,
    'two_qubit_ising_rotation_hyperfine' : 
        NVCentreExperimentWithoutTransverseTerms.nv_centre_spin_experimental_method_without_transvere_terms,
    'NV_alternative_model' : 
        NVCentreAlternativeTrueModel.nv_centre_spin_experimental_method_alternative_true_model,
    'NV_alternative_model_2' : 
        NVCentreAlternativeTrueModel.nv_centre_spin_experimental_method_alternative_true_model_second,
    'nv_experiment_vary_model' : 
        NVExperimentVaryTrueModel.nv_centre_spin_experimental_method_varying_true_model,
    'NV_fitness_growth' : 
        NVGrowByFitness.nv_fitness_growth, 
    'NV_centre_revivals' : 
        NVCentreRevivals.nv_centre_revival_data,
    'NV_spin_full_access' : 
        NVCentreFullAccess.nv_centre_spin_full_access,
    'NV_centre_spin_large_bath' : 
        NVCentreLargeSpinBath.nv_centre_large_spin_bath,
    'reduced_nv_experiment' : 
        ReducedNVExperiment.reduced_nv_experiment,
    # Theoretical paper growth rules
    'ising_predetermined' : 
        Ising.ising_chain_predetermined,
    'ising_probabilistic' : 
        Ising.ising_chain_probabilistic,
    'heisenberg_xyz_predetermined' : 
        HeisenbergXYZ.heisenberg_xyz_predetermined,
    'heisenberg_xyz_probabilistic' :
        HeisenbergXYZ.heisenberg_xyz_probabilistic,
    'fermi_hubbard_predetermined' : 
        FermiHubbard.fermi_hubbard_predetermined,
    'fermi_hubbard_probabilistic' : 
        FermiHubbard.fermi_hubbard_probabilistic,
    # Others
    'basic_lindbladian' :
        BasicLindbladian.basic_lindbladian,
    'example' : 
        ExampleGrowthRule.example_growth,
    'presentation' : 
        PresentationPlotGeneration.presentation_plot_generation
}


def get_growth_generator_class(
    growth_generation_rule,
    **kwargs
):
    # TODO: note that in most places, this is called with use_experimental_data.
    # in some plotting functions this is not known, but it should not matter unless
    # called to get probes etc. 

    # print("Trying to find growth class for ", growth_generation_rule)
    # print("kwargs:", kwargs)
    try:
        gr = growth_classes[growth_generation_rule](
            growth_generation_rule = growth_generation_rule, 
            **kwargs
        )
    except:
        raise
        # print("{} growth class not found.".format(growth_generation_rule))
        gr = None

    return gr

    