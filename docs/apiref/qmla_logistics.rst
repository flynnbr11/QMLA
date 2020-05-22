..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _qmla_logistics:
.. currentmodule:: qmla


QMLA logistics
===============


User controls 
-------------

Classes
~~~~~~~~~~~~~~~
:class:`ControlsQMLA` Controls (user and otherwise) to specify QMLA instance.

.. autoclass:: ControlsQMLA
    :members:



Database framework
------------------

Classes 
~~~~~~~~~~~~~~~~
:class:`Operator` Object for mathematical properties of a single model.

.. autoclass:: Operator
    :members:

Functions
~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_num_qubits
.. autofunction:: get_constituent_names_from_name
.. autofunction:: alph
.. autofunction:: consider_new_model


Model Generation
----------------

Functions
~~~~~~~~~
.. autofunction:: process_basic_operator


Initialising growth rule
------------------------

Functions 
~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_growth_rule


Trees and branches
------------------
Classes
~~~~~~~~~~~~~~~~
.. autoclass:: TreeQMLA
    :members:
.. autoclass:: BranchQMLA
    :members:


Parameter definition
--------------------
Functions
~~~~~~~~~~~~~~~~~~~
.. autofunction:: set_shared_parameters


Redis
-----
Functions
~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_redis_databases_by_qmla_id
.. autofunction:: get_seed


Logging
-------
Functions
~~~~~~~~~~~~~~~~~~~
.. autofunction:: print_to_log

