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

    :meth:`~qmla.ModelInstanceForLearning._plot_learning_summary`

