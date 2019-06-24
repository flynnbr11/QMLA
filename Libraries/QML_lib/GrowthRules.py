import sys
import os

this_dir = str(os.path.dirname(os.path.realpath(__file__)))
sys.path.append((os.path.join(this_dir, "GrowthRuleClasses")))

import SuperClassGrowthRule
import NV_centre_experiment_growth_rules
import NV_centre_full_access
import NV_centre_large_spin_bath
import NV_centre_experiment_without_transverse_terms
import NV_centre_revivals
import Reduced_NV_experiment
import IsingChain
import Hubbard
import Heisenberg
import Hopping
import IsingMultiAxis


growth_classes = {
    # 'test_growth_class' : 
    #     SuperClassGrowthRule.default_growth, 
    # 'two_qubit_ising_rotation_hyperfine_transverse' : 
    #     SuperClassGrowthRule.default_growth, 
    'two_qubit_ising_rotation_hyperfine_transverse' : 
        NV_centre_experiment_growth_rules.NVCentreSpinExperimentalMethod,
    'two_qubit_ising_rotation_hyperfine' : 
        NV_centre_experiment_without_transverse_terms.NVCentreSpinExperimentalMethodWithoutTransvereTerms,
    'NV_centre_revivals' : 
        NV_centre_revivals.NVCentreRevivalData,
    'NV_spin_full_access' : 
        NV_centre_full_access.NVCentreSpinFullAccess,
    'NV_centre_spin_large_bath' : 
        NV_centre_large_spin_bath.NVCentreLargeSpinBath,
    'reduced_nv_experiment' : 
        Reduced_NV_experiment.reducedNVExperiment,
    'ising_1d_chain' : 
        IsingChain.isingChain,
    'ising_multi_axis' : 
        IsingMultiAxis.isingChainMultiAxis, 
    'hubbard_square_lattice_generalised' : 
        Hubbard.hubbardSquare,
    'heisenberg_xyz' : 
        Heisenberg.heisenbergXYZ, 
    'hopping_topology' : 
        Hopping.hopping
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

    