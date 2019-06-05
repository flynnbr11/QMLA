import sys
import os

this_dir = str(os.path.dirname(os.path.realpath(__file__)))
sys.path.append((os.path.join(this_dir, "GrowthRuleClasses")))

import SuperClassGrowthRule
import NV_centre_experiment_growth_rules
import NV_centre_full_access
import NV_centre_large_spin_bath
import NV_centre_experiment_without_transverse_terms
import Reduced_NV_experiment
import IsingChain
import Hubbard


growth_classes = {
    # 'test_growth_class' : 
    #     SuperClassGrowthRule.default_growth, 
    # 'two_qubit_ising_rotation_hyperfine_transverse' : 
    #     SuperClassGrowthRule.default_growth, 
    'two_qubit_ising_rotation_hyperfine_transverse' : 
        NV_centre_experiment_growth_rules.NVCentreSpinExperimentalMethod,
    'two_qubit_ising_rotation_hyperfine' : 
        NV_centre_experiment_without_transverse_terms.NVCentreSpinExperimentalMethodWithoutTransvereTerms,
    'NV_spin_full_access' : 
        NV_centre_full_access.NVCentreSpinFullAccess,
    'NV_centre_spin_large_bath' : 
        NV_centre_large_spin_bath.NVCentreLargeSpinBath,
    'reduced_nv_experiment' : 
        Reduced_NV_experiment.reducedNVExperiment,
    'ising_1d_chain' : 
        IsingChain.isingChain,
    'hubbard_square_lattice_generalised' : 
        Hubbard.hubbardSquare
}


def get_growth_generator_class(
    growth_generation_rule,
    **kwargs
):
    # print("Trying to find growth class for ", growth_generation_rule)
    try:
        gr = growth_classes[growth_generation_rule](
            growth_generation_rule = growth_generation_rule, 
            **kwargs
        )
    except:
        # print("{} growth class not found.".format(growth_generation_rule))
        # raise
        gr = None

    return gr

    