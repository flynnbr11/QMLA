API Reference
=============

.. currentmodule:: qmla

.. _section_quantum_model_learning_agent:
   

Quantum Model Learning Agent
----------------------------

Manager class
~~~~~~~~~~~~~~
The overall :term:`QMLA` protocol is managed by this class. 

.. autoclass:: QuantumModelLearningAgent
    :members: 
    :private-members: 


Logistics
---------
Here we list some of the functionality used as the logistics to implement :term:`QMLA`.

User controls 
~~~~~~~~~~~~~
:class:`~qmla.ControlsQMLA` Controls (user and otherwise) to specify QMLA instance.

.. autoclass:: ControlsQMLA
    :members:

Database framework
~~~~~~~~~~~~~~~~~~
:class:`Operator` Object for mathematical properties of a single model.

.. autoclass:: Operator
    :members:

Functions
^^^^^^^^^
.. autofunction:: get_num_qubits
.. autofunction:: get_constituent_names_from_name
.. autofunction:: alph

Model Generation
~~~~~~~~~~~~~~~~
.. autofunction:: process_basic_operator

.. _section_string_processing:

String to matrix processing
~~~~~~~~~~~~~~~~~~~~~~~~~~
These functions map strings to matrices which can be used in the construction of models. 


Initialising Exploration Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_exploration_class


Trees and branches
~~~~~~~~~~~~~~~~~~
.. autoclass:: ExplorationTree
    :members:
.. autoclass:: BranchQMLA
    :members:


Parameter definition
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: set_shared_parameters


Redis
~~~~~
.. autofunction:: get_redis_databases_by_qmla_id
.. autofunction:: get_seed


Logging
~~~~~~~
.. autofunction:: print_to_log
  
.. currentmodule:: qmla

Models 
------
Model for training
~~~~~~~~~~~~~~~~~~
Model to perform parameter learning upon, usually :term:`QHL`. 

This is a disposable class which instatiates indepdendently from :class:`~qmla.QuantumModelLearningAgent`, 
trains the model via :func:`qmla.remote_learn_model_parameters`, performs analysis on the trained model, 
summarises the outcome of the training and sends a concise data packet to the database, before being deleted. 
The model training refers to :term:`QHL`, performed in conjunction with [QInfer]_, 
via :meth:`~qmla.ModelInstanceForLearning.update_model`.

.. autoclass:: ModelInstanceForLearning
    :members:
    :private-members: 


Model for comparisons
~~~~~~~~~~~~~~~~~~~~~
Model to use during Bayes factor comparisons.

This is a disposable class which reads the redis database to retrieve information about the 
trainng of the given model ID. 
It then reconstructs the model, e.g. based on the final estimated mean of the parameter distribution. 
Then, it is interfaced with a competing instance of the class within :func:`~qmla.remote_bayes_factor_calculation`:
the opponent's experiments are used for further updates to the present model, such that the two models under consideration
have identical experiment records (at least partially whereupon the BF is based), allowing for meaningful comparison among the two.  

.. autoclass:: ModelInstanceForComparison
    :members:

Model for storage
~~~~~~~~~~~~~~~~~
This object is much smaller than the other forms of the model, i.e. those used for training 
(:class:`~qmla.ModelInstanceForLearning`) and comparisons (:class:`~qmla.ModelInstanceForComparison`), 
which retains only the useful information for storage/analysis within the bigger picture in 
:class:`~qmla.QuantumModelLearningAgent`. 
It retrieves the succinct summaries of the training/comparisons pertainng to a single model 
which are stored on the redis database, allowing for later anlaysis as required by :term:`QMLA`.

.. autoclass:: ModelInstanceForStorage
    :members:

Implementation 
--------------
Model learning
~~~~~~~~~~~~~~
.. autofunction:: remote_learn_model_parameters

Model comparison
~~~~~~~~~~~~~~~~
.. autofunction:: remote_bayes_factor_calculation
.. autofunction:: qmla.remote_bayes_factor.plot_dynamics_from_models

Exploration Strategies
----------------------
.. currentmodule:: qmla.exploration_strategies
:class:`~qmla.exploration_strategies.ExplorationStrategy` Exploration Strategies do this thing. 
    TODO.

.. autoclass:: ExplorationStrategy
    :members:
    :private-members: _initialise_model_for_learning

.. currentmodule:: qmla
Functions
^^^^^^^^^
In order to initialise an :term:`ES`, :term:`QMLA` calls this function, 
which searches within the namespace of ``qmla/exploration_strategies``. 

.. autofunction:: get_exploration_class

Modular functionality
---------------------
.. currentmodule:: qmla
As outlined in :ref:`section_modular_functionality`, some subroutines are modular; 
here we list some of the availbale implementations. 

Experiment Design Hueristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: qmla.shared_functionality.experiment_design_heuristics
.. autoclass:: ExperimentDesignHueristic
    :members: 

Expectation Values
~~~~~~~~~~~~~~~~~~
.. currentmodule:: qmla.shared_functionality.expectation_value_functions
.. autofunction:: default_expectation_value

Prior probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: qmla.shared_functionality.prior_distributions
.. autofunction:: gaussian_prior

QInfer Interface
~~~~~~~~~~~~~~~~
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
