.. _section_analysis:

Output and Analysis
-------------------

When a run is launched (either locally or remotely), a results directory 
is built for that run. 
In that directory, results are stored in several formats from each instance. 

By default, QMLA provides a set of analyses, generating several plots
in the sub-directories of the run's results directory. 

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

    :func:`~qmla.plot_dynamics_from_models`

``plot_level=5``

    :meth:`~qmla.ModelInstanceForLearning._plot_distributions`
    
    :meth:`~qmla.plot_heuristic_attributes`
    

``plot_level=6``
