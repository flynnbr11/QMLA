
Glossary
========

.. glossary::
    :sorted:

    Instance 
        A single implementation of QMLA. 
    
    Run
        A collection of :term:`Instance`s. Note that for a run, 
        all instances must target the same :term:`System`. 

    QMLA
        Quantum Model Learning Agent. 

    QHL
        Quantum Hamiltonian Learning. Given a parameterisation 
        of a model for a quantum system, 
        and a target system from which to extract data, 
        QHL learns the parameters which best match the data, through
        :term:`QLE`. 

    QLE 
        Quantum Likelihood Estimation. The algorithm which is used to 
        perform Bayesian inference during parameter learning for a given
        candidate model. 

    System
        The target system, i.e. underlying model. 
        In simulation, this is used to generate the expectation values
        against which likelihood estimation occurs. 
        In experiments, the form of the system is unknown, but 
        data obtained from experiments are used in the likelihood 
        estimation instead. 

    True Model
        .. seealso:: :term:`System`

    Quantum Hamiltonian Learning
        .. seealso:: :term:`QHL`

    Quantum Model Learning Agent
        .. seealso:: :term:`QMLA`

    Quantum Likelihood Estimation
        .. seealso:: :term:`QLE`

    Exploration Strategy
        .. seealso:: :term:`ES`
    ES
        Exploration Strategy. 
        The mechanism by which a tree grows, specifying new
        models to consider, when to stop considering new models, 
        how to remove models, etc. 
