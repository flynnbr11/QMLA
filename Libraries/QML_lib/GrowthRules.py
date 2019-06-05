import sys
import os

this_dir = str(os.path.dirname(os.path.realpath(__file__)))
sys.path.append((os.path.join(this_dir, "GrowthRuleClasses")))

import SuperClassGrowthRule
import NV_centre_experiment_growth_rules




growth_classes = {
    # 'test_growth_class' : 
    #     SuperClassGrowthRule.default_growth, 
    # 'two_qubit_ising_rotation_hyperfine_transverse' : 
    #     SuperClassGrowthRule.default_growth, 
    'two_qubit_ising_rotation_hyperfine_transverse' : 
        NV_centre_experiment_growth_rules.nv_spin_experiment_full_tree,
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
        print("{} growth class not found.".format(growth_generation_rule))
        # raise
        gr = None
    return gr

    