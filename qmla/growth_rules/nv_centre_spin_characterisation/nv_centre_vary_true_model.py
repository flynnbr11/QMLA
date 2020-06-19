import random
import sys
import os

from qmla.growth_rules.nv_centre_spin_characterisation import nv_centre_experiment
import qmla.shared_functionality.probe_set_generation
from qmla import construct_models


class ExperimentNVCentreVaryTrueModel(
    nv_centre_experiment.ExperimentNVCentre  # inherit from this
):
    # Uses all the same functionality, growth etc as
    # default NV centre spin experiments/simulations
    # but uses an expectation value which traces out

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.max_time_to_consider
        self.true_model = 'xTiPPyTiPPzTiPPzTz'
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.NV_centre_ising_probes_plus
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict
        self.shared_probes = True
        self.true_model_terms_params = {}

        self.max_time_to_consider = 20
        self.min_param = 0
        self.max_param = 10


class ExperimentNVCentreVaryTrueModel_3_params(
    ExperimentNVCentreVaryTrueModel  # inherit from this
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.true_model = 'xTiPPyTiPPzTi'


class ExperimentNVCentreVaryTrueModel_5_params(
    ExperimentNVCentreVaryTrueModel  # inherit from this
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.true_model = 'xTiPPxTxPPyTiPPyTyPPzTi'


class ExperimentNVCentreVaryTrueModel_6_params(
    ExperimentNVCentreVaryTrueModel  # inherit from this
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.true_model = 'xTiPPxTxPPyTiPPyTyPPzTiPPzTz'


class ExperimentNVCentreVaryTrueModel_7_params(
    ExperimentNVCentreVaryTrueModel  # inherit from this
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.true_model = 'xTiPPxTxPPxTzPPyTiPPyTyPPzTiPPzTz'
