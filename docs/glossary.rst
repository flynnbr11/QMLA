
Glossary
========

.. glossary::
    :sorted:

    Instance 
        A single implementation of QMLA. 
        
        .. seealso:: Defined in :ref:`section_structure`.
    
    Run
        A collection of :term:`instance` s. Note that for a run, 
        all instances must target the same :term:`system` . 

        .. seealso:: Defined in :ref:`section_structure`.

    QMLA
        .. seealso:: :term:`Quantum Model Learning Agent`

    QHL
        .. seealso:: :term:`Quantum Hamiltonian Learning`

    QLE 
        .. seealso:: :term:`Quantum Likelihood Estimation`

    Quantum Hamiltonian Learning
        Algorithm for learning the parameters of a given model. 

    Quantum Model Learning Agent
        Algorithm/framework for finding model of quantum system.

    Quantum Likelihood Estimation
        Algorithm used to perform Bayesian inference during :term:`QHL`

    Exploration Strategy
        The mechanism by which a tree grows, specifying new
        models to consider, when to stop considering new models, 
        how to remove models, etc. 
        
        .. seealso:: Defined in :ref:`section_exploration_strategies`.
        
    ES
        .. seealso:: :term:`Exploration Strategy`
   
    Exploration Tree
        Unique tree associated with an individual :term:`Exploration Strategy`. 
        
        .. seealso:: Defined in :ref:`section_structure`.

    ET 
        .. seealso:: :term:`Exploration Tree`

    System
        The target system, i.e. underlying model. 
        In simulation, this is used to generate the expectation values
        against which likelihood estimation occurs. 
        In experiments, the form of the system is unknown, but 
        data obtained from experiments are used in the likelihood 
        estimation instead. 

    True Model
        .. seealso:: :term:`system`


    Bayes factor
        Statistical measure of performance between two models at explaining the same dataset

    BF 
        .. seealso:: :term:`Bayes factor`

    global champion
        Single model favoured by :term:`Quantum Model Learning Agent` as the strongest candidate
        to represent the :term:`system`.
    
