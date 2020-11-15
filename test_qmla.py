import pytest
import qmla



def test_qmla_attributes():
    exploration_strategy = qmla.exploration_strategies.HeisenbergXYZProbabilistic(
        exploration_rules = 'HeisenbergXYZProbabilistic'
    )

    instance = qmla.QuantumModelLearningAgent()
    controls = instance.qmla_controls
    assert controls.true_model == instance.true_model_name, 'True model not set'
    assert \
        controls.qhl_mode + controls.further_qhl + controls.qhl_mode_multiple_models <= 1,  \
        'Multiple QMLA modes enabled'

    