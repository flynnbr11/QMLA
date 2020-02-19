import pytest
import qmla



def test_qmla_import():
    growth_rule = qmla.growth_rules.HeisenbergXYZProbabilistic(
        growth_generation_rule = 'HeisenbergXYZProbabilistic'
    )
    assert \
        growth_rule.lattice_dimension \
        == 1, \
        "Testing GR lattice dimension" 

    assert 1 == 1


# def test_qmla_growth_rule():
#     growth = qmla.growth_rules.HeisenbergXYZProbabilistic(
#         growth_generation_rule = 'HeisenbergXYZProbabilistic'
#     )
#     assert \
#         growth.lattice_dimension \
#         == 2, \
#         "Testing GR lattice dimension" 
