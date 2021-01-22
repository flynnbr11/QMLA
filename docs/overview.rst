..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _section_overview:

============
Overview
============


:term:`Quantum Model Learning Agent (QMLA) <QMLA>` is a machine learning protocol for the 
characterisation of quantum mechanical systems and devices. 
It aims to determine the model which best explains observed data
from the system under study. 
It does this by considering a series of candidate models, 
performing a learning procedure to optimise the performance of 
those candidates, and then selecting the best candidate. 
New candidate models can be constructed iteratively, using 
information gained so far in the procedure, to improve the 
model approximation.

:term:`QMLA` can operate on simulated or experimental data, or be incorporated with an online experiment. 
This document provides details on QMLA's mechanics, and provides examples of usage. 
Particular attention is paid to the concept and design of :term:`Exploration Strategies (ES) <ES>`, 
the primary mechanism by which users ought to interact with the software. 
Custom :term:`ES` can be developed and plugged into the :term:`QMLA` framework, in order to 
target systems, run experiments and generate models according to the user's requirements. 
Several aspects of the framework are modular, allowing users to select the combination
which suits their requirements, or easily add functionality as needed. 

This chapter briefly introduces the core concepts at a high level; 
thorough detail can be found in :ref:`guide`.

Models
======
Models encapsulate the physics of a system.
We generically refer to `models` 
because :term:`QMLA` is indifferent to the formalism employed to describe the system. 
Usually we mean `Hamiltonian` models, although :term:`QMLA` may also be used to 
learn `Lindbladian` models. 

Models are simply the mathematical objects which can be used to predict the behaviour of a
system, uniquely represented by a parameterisation. 
Each term in a model is really a matrix corresponding 
to some physical interaction; each such term is assigned a scalar parmeter.
The total model is a matrix, which is computed by  
the sum of the terms multiplied by their parameters. 
For example, 1-qubit models can be constructed using the Pauli operators
:math:`\hat{\sigma}_x, \hat{\sigma}_y, \hat{\sigma}_z`, e.g.
:math:`\hat{H}_{xy} = \alpha_x \hat{\sigma}_x + \alpha_y \hat{\sigma}_y`. 
Then, :math:`\hat{H}_{xy}` is completely described by the vector 
:math:`\vec{\alpha} =(\alpha_x, \alpha_y)`, when we know the corresponding terms
:math:`\vec{T} = ( \hat{\sigma}_x, \hat{\sigma_y} )`. 
In general then, models are given by 
:math:`\hat{H}(\vec{\alpha}) = \vec{\alpha} \cdot \vec{T}`. 

In the Hamiltonian (closed) formalism, terms included in the model correspond 
to interactions between particles in the system. 
For example, the Ising model Hamiltonian on :math:`N` sites (spins), 
:math:`\hat{H}^{\otimes N} = J \sum\limits_{i=1}^{N-1} \hat{\sigma}_i^z \hat{\sigma}_{i+1}^z`,
includes terms 
:math:`\hat{\sigma}_i^z \hat{\sigma}_{i+1}^z`
which are the interactions between nearest neighbour sites :math:`(i, i+1)`. 


:term:`QMLA` reduces assumptions about which interactions are present, 
for instance by considering models :math:`\hat{H}^{\otimes 5}` and 
:math:`\hat{H}^{\otimes 8}`, and determining which model (5 or 8 spins)
best describes the observed data. 
Moreover, :term:`QMLA` facilitates consideration of all terms independently, 
e.g. whether the system is better described by a partially connected
Ising lattice :math:`\hat{H}_1` 
or a nearest-neighbour connected Ising chain :math:`\hat{H}_2`:

.. math::

    \hat{H}_1 =  
    \alpha_1 \hat{\sigma}_1^z \hat{\sigma}_{2}^z
    + \alpha_2  \hat{\sigma}_1^z \hat{\sigma}_{3}^z
    + \alpha_3  \hat{\sigma}_1^z \hat{\sigma}_{4}^z

.. math::

    \hat{H}_2 =  
    \beta_1  \hat{\sigma}_1^z \hat{\sigma}_{2}^z
    + \beta_2  \hat{\sigma}_2^z \hat{\sigma}_{3}^z
    + \beta_3  \hat{\sigma}_3^z \hat{\sigma}_{4}^z


Then, models exist in a `model space`, i.e. the space of all valid combinations of the available terms.
Any combination of terms is permissible in a given model. 
:term:`QMLA` can then be thought of as a search through the model space for 
the set of terms which produce data that best matches that of the system. 
Since these terms correspond to the physical interactions affecting the system, 
the outcome can be thought of as a complete characterisation. 


Model Training
==============
Model traning is the process of optimising the parameters :math:`\vec{\alpha}` of a given model against the system's data. 
The model which is being learned does not need to be the `true` model; any model can attempt
to describe any data.
A core hypothesis of :term:`QMLA` is that models which better reflect the true model
will produce data more consistent with the system data, when compared against less-physically-similar models. 

In principle, any parameter-learning algorithm can fulfil the role of training models in the :term:`QMLA` framework,
but in practice, :term:`Quantum Hamiltonian Learning (QHL) <QHL>` is used to
perform  Bayesian inference on the parameterisation, and hence attempt to find the optimal parameterisation for each model
[WGFC13a]_, [WGFC13b]_, [GFWC12]_. 
This is performed using [QInfer]_. 

Model Comparison
================
Two candidate models :math:`\hat{H}_1, \hat{H}_2`, having undergone model training,
can be compared against each other to determine which one better describes the system data. 
:term:`Bayes factor (BF) <Bayes factor>` provide a quantitative measure of the relative strength 
of the models at producing the data. 
We take the :term:`BF` :math:`B(\hat{H}_1, \hat{H}_2)` between two candidate models 
as evidence that one model is preferable. 
Evidence is compiled in a series of pairwise comparisons; models are compared with 
a number of competitors such that the strongest model from a pool can be determined as that which
won the highest number of pairwise comparisons.  

.. _section_structure:

Structure
=========
:term:`QMLA` is structured over several levels:

Models 
    are individual candidates (e.g. Hamiltonians) which attempt to capture the physics of the :term:`system`.

Layers/Branches:
    models are grouped in layers, which are thought of as branches on exploration trees.

Exploration trees
    are the objects on which the model search takes place: we think of models as *leaves*
    on *branches* of a tree. 
    The model search is then the effort to find the single leaf on the tree which best describes the :term:`system`. 
    They grow and are pruned according to rules set out in the exploration strategy. 

Exploration Strategies (:term:`ES`)
    are bespoke sets of rules which decide how :term:`QMLA` ought to proceed at each step. 
    For example, given the result of training/comparing a previous set of models, the :term:`ES` 
    determines the next set of candidate models to be considered.    

Instance:
    a single implementation of the :term:`QMLA` protocol, whether to run the entire model search or another subroutine the framework.
    Within an instance, several exploration trees can grow independently in parallel: 
    we can then think of :term:`QMLA` as a search for the single best leaf among a forest of trees,
    each of which corresponds to a unique exploration strategy.

Run
    many instances which pertain to the same problem. 
    :term:`QMLA` is run independently for a number of instances, allowing for analysis of the algorithm's performance overall, 
    e.g. that :term:`QMLA` finds a particular model in 50% of 100 instances. 


Outputs
=======

:term:`QMLA` automatically performs a series of analyses and produces associated plots. 
These are stored in a unique folder generated for the :term:`run` upon launch:
this folder is specified by the date and time of the launch and is located, relative to the 
:term:`QMLA` main project directory in, e.g.,  ``launch/results/Jan_01/12_34``. 
These are detailed in :ref:`section_analysis`.

User Interface 
==============

In order to tailor :term:`QMLA` to a user's needs, 
they must design a bespoke :ref:`section_exploration_strategies`.
That is, the user must write a class building upon and inheriting from :class:`~qmla.exploration_strategies.ExplorationStrategy`, 
encompassing all of the logic required to achieve their use case, 
for example by incorporating a genetic algorithm within the method called upon for constructing new 
candidates, :meth:`~qmla.exploration_strategies.ExplorationStrategy.generate_models`.
Then, that class must be available to :func:`~qmla.get_exploration_class`, 
by ensuring it is included in one of the ``import`` statements in ``qmla/exploration_strategies/__init__.py``.
Finally, instruct :term:`QMLA` to use that :term:`ES` for a run in the launch script (see :ref:`section_launch`).
These steps are laid out in full in :ref:`section_tutorial`.