import pytest
# import qmla


def test_basic():
    a = 2
    assert a == 2, 'a=2'

# def test_qmla_growth_rule():
#     growth = qmla.growth_rules.HeisenbergXYZProbabilistic(
#         growth_generation_rule = 'HeisenbergXYZProbabilistic'
#     )
#     assert \
#         growth.lattice_dimension \
#         == 2, \
#         "Testing GR lattice dimension" 
