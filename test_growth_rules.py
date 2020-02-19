import pytest
import qmla



def test_heisenberg_xyz():
    growth_rule = qmla.growth_rules.HeisenbergXYZProbabilistic(
        growth_generation_rule = 'HeisenbergXYZProbabilistic'
    )
    assert \
        growth_rule.lattice_dimension \
        == 1, \
        "Testing GR lattice dimension" 
