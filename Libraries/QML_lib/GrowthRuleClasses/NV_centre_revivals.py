import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ExpectationValues
import ProbeGeneration

from NV_centre_experiment_growth_rules import NVCentreSpinExperimentalMethod

class NVCentreRevivalData(
    NVCentreSpinExperimentalMethod # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        import Heuristics
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        self.experimental_dataset = 'NV_revivals.p'

 