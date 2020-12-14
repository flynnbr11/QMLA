from qmla.exploration_strategies.nv_centre_spin_characterisation import nv_centre_experiment

class ExperimentNVCentreNoTransvereTerms(
        nv_centre_experiment.NVCentreSimulatedExperiment
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):

        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.max_num_parameter_estimate = 6
        self.max_spawn_depth = 5
        self.max_num_models_by_shape = {
            1: 0,
            2: 12,
            'other': 0
        }
