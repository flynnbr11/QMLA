..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _models:
.. currentmodule:: qmla

Models 
======

Model for learning
------------------

Classes
~~~~~~~~~~~~~~~
Model to perform parameter learning upon, usually quantum Hamiltonian learning. 

.. currentmodule:: qmla.model_for_learning
.. autoclass:: ModelInstanceForLearning
    :members:
    :private-members: _initialise_model_for_learning


Model for comparisons
---------------------
Classes
~~~~~~~~~~~~~~~~
Model to use during Bayes factor comparisons

.. autoclass:: ModelInstanceForComparison
    :members:

Model for storage
-----------------

Classes
~~~~~~~~~~~~~~~~
Reduced model to store in :class:`~qmla.QuantumModelLearningAgent`

.. autoclass:: ModelInstanceForStorage
    :members: