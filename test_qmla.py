import pytest
import qmla



def test_qmla_attributes():
    growth_rule = qmla.growth_rules.HeisenbergXYZProbabilistic(
        growth_generation_rule = 'HeisenbergXYZProbabilistic'
    )

    instance = qmla.QuantumModelLearningAgent()
    controls = instance.qmla_controls
    assert controls.true_model == instance.true_model_name, 'True model not set'
    assert \
        controls.qhl_mode + controls.further_qhl + controls.qhl_mode_multiple_models <= 1,  \
        'Multiple QMLA modes enabled'

    