import NV_centre_experiment_growth_rules


class NVCentreSpinExperimentalMethodWithoutTransvereTerms(
	NV_centre_experiment_growth_rules.NVCentreSpinExperimentalMethod
):
    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):

        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        self.max_num_parameter_estimate = 6
        self.max_spawn_depth = 5

