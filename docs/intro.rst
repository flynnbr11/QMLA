..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _intro:

============
Overview
============


`Quantum Model Learning Agent` (QMLA) is a machine learning protocol for the 
characterisation of quantum mechanical systems and devices. 
It aims to determine the model which best explains observed data
from the system under study. 
It does this by considering a series of candidate models, 
performing a learning procedure to optimise the performance of 
those candidates, and then selecting the best candidate. 
New candidate models can be constructed iteratively, using 
information gained so far in the procedure, to improve the 
model approximation.

QMLA can operate on simulated or experimental data, or be incorporated with an online experiment. 
This document provides details on QMLA's mechanics, and provides examples of usage. 
Particular attention is given to the concept and design of `Exploration Strategies`, 
the primary mechanism by which users ought to interact with the software. 
Custom Exploration Strategies can be developed and plugged into the QMLA framework, to 
target systems, run experiments and generate models according to the user's requirements. 
Several aspects of the framework are modular, allowing users to select the combination
which suits their requirements, or easily add functionality as needed. 

This chapter briefly introduces the core concepts at a high level; 
thorough detail can be found in :ref:`guide`.

Models
======
Models encapsulate the physics of the system.
We generically refer to `models` 
because QMLA is indifferent to the formulism employed to describe
the system. 
Usually we mean `Hamiltonian` models, although QMLA can also 
learn `Lindbladian` models. 

Models are uniquely represented by a parameterisation. 
Each term in a model is really a matrix corresponding 
to some physical interaction. 
Each term in the model is assigned a parmeter. 
The total model is a matrix, which is computed by  
the sum of the terms multiplied by their parameters. 
For example, 1-qubit models can be constructed using the Pauli operators
:math:`\hat{\sigma}_x, \hat{\sigma}_y, \hat{\sigma}_z`, e.g.
:math:`\hat{H}_{xy} = \alpha_x \hat{\sigma}_x + \alpha_y \hat{\sigma}_y`. 
Then, :math:`\hat{H}_{xy}` is completely described by the vector 
:math:`\vec{\alpha} =(\alpha_x, \alpha_y)`, when we know the corresponding operators
:math:`\vec{\hat{O}} = ( \hat{\sigma}_x, \hat{\sigma_y} )`. 
In general then, models are given by 
:math:`\hat{H}(\vec{\alpha}) = \vec{\alpha} \vec{\hat{O}}`. 

In the Hamiltonian (closed) formalism, terms included in the model correspond 
to interactions between particles in the system. 
For example, the Ising model Hamiltonian on :math:`N` sites (spins), 
:math:`\hat{H}^{\otimes N} = J \sum\limits_{i=1}^{N-1} \hat{\sigma}_i^z \hat{\sigma}_{i+1}^z`,
includes terms 
:math:`\hat{\sigma}_i^z \hat{\sigma}_{i+1}^z`
which are the interactions between nearest neighbour sites :math:`(i, i+1)`. 


QMLA reduces assumptions about which interactions are present, 
for instance by considering models :math:`\hat{H}^{\otimes 5}` and 
:math:`\hat{H}^{\otimes 8}`, and determining which model (5 or 8 spins)
best describes the observed data. 
Moreover, QMLA facilitates consideration of all terms independently, 
e.g. whether the system is better described by a partially connected
Ising lattice :math:`\hat{H}_1` 
or a nearest-neighbour connected Ising chain :math:`\hat{H}_2`:

:math:`\hat{H}_1 =  
\alpha_1 \hat{\sigma}_1^z \hat{\sigma}_{2}^z
+ \alpha_2  \hat{\sigma}_1^z \hat{\sigma}_{3}^z
+ \alpha_3  \hat{\sigma}_1^z \hat{\sigma}_{4}^z`

:math:`\hat{H}_2 =  
\beta_1  \hat{\sigma}_1^z \hat{\sigma}_{2}^z
+ \beta_2  \hat{\sigma}_2^z \hat{\sigma}_{3}^z
+ \beta_3  \hat{\sigma}_3^z \hat{\sigma}_{4}^z`


Then, models exist in a `model space`, i.e. the space of all available terms.
Any combination of terms is permissible in a given model. 
QMLA can then be thought of as a search through the model space for 
the set of terms which produce data that best matches that of the system. 



Model Learning
==============
Model learning is the process of optimising the parameters of a given model against the system's data. 
The model which is being learned does not need to be the `true` model; any model can attempt
to describe any data.
A core hypothesis of QMLA is that models which better reflect the true model
will produce data more consistent with the system data, when compared against less 
similar models. 

In practice, :term:`Quantum Hamiltonian Learning` is applied
to perform Bayesian inference and hence find the best parameterisation
for each model. 
In principle, however, any parameter learning technique can fulfil this part of the protocol, 
such as Hamiltonian tomography. 


Model Comparison
================
Two candidate models :math:`\hat{H}_1, \hat{H}_2`, having undergone parameter learning,
can be compared to determine which one better describes the system data. 
Bayes factors provide a quantitative measure of the relative strength 
of the models at producing the data. 
We take the Bayes factor :math:`B(\hat{H}_1, \hat{H}_2)` between two candidate models 
as evidence that one model is preferable. 
Evidence is compiled in a series of pairwise comparisons: models are compared with 
a number of competitors such that the strongest model from a pool can be determined. 

Outputs
=======
QMLA automatically performs a series of analyses and produces associated plots. 

Analyses are available on three levels: 

Run: 
    Results across a number of instances.

    Example: the number of isntance wins for champion models. 
    
    Example: average dynamics reproduced by champion models. 
Instance: 
    Performance of a single insance. 
    
    Example: branches and models generated
Model: 
    Individual model performance within an instance. 
    
    Example: parameter estimation through QHL. 
    
    Example: pairwise comparison between models.



