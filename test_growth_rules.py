import pytest
import qmla



def test_heisenberg_xyz():
    exploration_strategy = qmla.exploration_strategies.HeisenbergXYZProbabilistic(
        exploration_rules = 'HeisenbergXYZProbabilistic'
    )
    assert \
        exploration_strategy.lattice_dimension \
        == 1, \
        "Testing GR lattice dimension" 
