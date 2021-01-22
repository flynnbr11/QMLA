
.. _section_applications:


Applications 
=============

NV centre characterisation
--------------------------

.. TODO : change name of section and reference to indicate publication

The model searches presented in [GFK20]_ have exploration strategies as presented here. 

Greedy search 
~~~~~~~~~~~~~
.. currentmodule:: qmla.exploration_strategies.nv_centre_spin_characterisation
.. autoclass:: NVCentreExperimentalData
    :members: 
    :private-members: 

.. autoclass:: FullAccessNVCentre
    :members: 
    :private-members: 

.. autoclass:: SimulatedExperimentNVCentre
    :members: 
    :private-members: 

.. autoclass:: TieredGreedySearchNVCentre
    :members: 
    :private-members: 

Genetic algorithm for spin bath
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: qmla.exploration_strategies.nv_centre_spin_characterisation.nature_physics_2021
.. autoclass:: NVCentreGenticAlgorithmPrelearnedParameters
    :members: 
    :private-members: 


Genetic Algorithms
------------------
Genetic algorithms can be used within the :term:`Exploration Strategy` of :term:`QMLA`; 
here we provide a genetic algorithm framework which can be plugged in. 

.. currentmodule:: qmla.shared_functionality.genetic_algorithm
.. autoclass:: GeneticAlgorithmQMLA
    :members: 
    :private-members: 


Genetic Exploration Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A base class for genetic algorithm incorporated into the :term:`Exploration Strategy`. 

.. currentmodule:: qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy
.. autoclass:: Genetic
    :members: 
    :private-members: 


