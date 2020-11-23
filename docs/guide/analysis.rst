.. _section_analysis:

Output and Analysis
-------------------

When a run is launched (either locally or remotely), a results directory 
is built for that run. 
In that directory, results are stored in several formats from each instance. 

By default, QMLA provides a set of analyses, generating several plots
in the sub-directories of the run's results directory. 


Analyses are available on various levels: 

Run: 
    results across a number of instances.

    Example: the number of instance wins for champion models. 
    
    Example: average dynamics reproduced by champion models. 
Instance: 
    Performance of a single insance. 
    
    Example: models generated and the branches on which they reside
Model: 
    Individual model performance within an instance. 
    
    Example: parameter estimation through QHL. 
    
    Example: pairwise comparison between models.

Comparisons:
    Pairwise comparison of models' performance. 

    Example: dynamics of both candidates (with respect to a single basis).

Within the :ref:`section_launch` scripts, there is a ``plot_level`` variable which informs QMLA of how many plots to produce by default. 
This gives users a level of control over how much analysis is performed. 
For instance, while testing an Exploration Strategy, a higher degree of testing may be required, 
so plots relating to every individual model are desired. 
For large runs, however, where a large number of models are generated/compared, 
plotting each model's training performance is overly cumbersome and is unneccessary. 

The plots generated at each plot level are:

``plot_level=1``

    :meth:`~qmla.QuantumModelLearningAgent._plot_model_terms`

``plot_level=2``

``plot_level=3``

    :meth:`~qmla.QuantumModelLearningAgent._plot_dynamics_all_models_on_branches`

    :meth:`~qmla.QuantumModelLearningAgent._plot_evaluation_normalisation_records`

``plot_level=4``
    
    :meth:`~qmla.ModelInstanceForLearning._plot_learning_summary`

    :meth:`~qmla.ModelInstanceForLearning._plot_dynamics`

    :meth:`~qmla.ModelInstanceForLearning._plot_preliminary_preparation`

    :func:`~qmla.remote_bayes_factor.plot_dynamics_from_models`

``plot_level=5``

    :meth:`~qmla.ModelInstanceForLearning._plot_distributions`
    
    :meth:`~qmla.shared_functionality.experiment_design_heuristics.ExperimentDesignHueristic.plot_heuristic_attributes`
    

``plot_level=6``
