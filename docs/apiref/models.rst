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

Model for training
------------------

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
---------------------

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
-----------------

This object is much smaller than the other forms of the model, i.e. those used for training 
(:class:`~qmla.ModelInstanceForLearning`) and comparisons (:class:`~qmla.ModelInstanceForComparison`), 
which retains only the useful information for storage/analysis within the bigger picture in 
:class:`~qmla.QuantumModelLearningAgent`. 
It retrieves the succinct summaries of the training/comparisons pertainng to a single model 
which are stored on the redis database, allowing for later anlaysis as required by :term:`QMLA`.

.. autoclass:: ModelInstanceForStorage
    :members: