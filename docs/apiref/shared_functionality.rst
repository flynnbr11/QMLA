..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _shared_functionality:

.. currentmodule:: qmla

Shared functionality
====================
Some stuff is shared across exploration strategies.

Experiment Design Hueristics
----------------------------
.. currentmodule:: qmla.shared_functionality.experiment_design_heuristics
.. autoclass:: ExperimentDesignHueristic
    :members: 

Expectation Values
-------------------------------
.. currentmodule:: qmla.shared_functionality.expectation_value_functions
.. autofunction:: default_expectation_value

Prior probability distributions
--------------------------------
.. currentmodule:: qmla.shared_functionality.prior_distributions
.. autofunction:: gaussian_prior

QInfer Interface
----------------
.. currentmodule:: qmla.shared_functionality.qinfer_model_interface
.. autoclass:: QInferModelQMLA
    :members:

Latex name mapping 
~~~~~~~~~~~~~~~~~~
Some examples of working latex name maps are provided. 

.. currentmodule:: qmla.shared_functionality.latex_model_names
.. autofunction:: pauli_set_latex_name
.. autofunction:: grouped_pauli_terms
.. autofunction:: fermi_hubbard_latex
