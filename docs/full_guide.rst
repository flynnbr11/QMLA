..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _guide:

User Guide
============


Quantum Model Learning Agent
----------------------------

The class which controls everything is :class:`~qmla.QuantumModelLearningAgent`. 
An :term:`instance` of this class is used to run one of the available algorithms; many 
independent instances can operate simultaneously and be analysed together (e.g. 
to see the *average* reproduction of dynamics following model learning). 
This is referred to as a :term:`run`
The :term:`QMLA` class provides methods for each of the available algorithms, as well 
as routines required therein, and methods for analysis and plotting. 
In short, the available algorithms are

``Quantum Model Learning Agent`` complete model search 

    :meth:`~qmla.QuantumModelLearningAgent.run_complete_qmla`

``Quantum Hamiltonian Learning`` just run the parameter optimisation subroutine. 
Runs on the model set as ``true_model`` within the :term:`ES`.

    :meth:`~qmla.QuantumModelLearningAgent.run_quantum_hamiltonian_learning`

``Multi-model quantum Hamiltonian learning`` just run the parameter optimisation subroutine. 
Runs on several models independently; the models are set in the list  ``qhl_models`` within the :term:`ES`.

    :meth:`~qmla.QuantumModelLearningAgent.run_quantum_hamiltonian_learning_multiple_models`


The primary function of the :class:`~qmla.QuantumModelLearningAgent` class is to manage the model search. 
Models are assigned a unique ``model_ID`` upon generation. 
QMLA considers a set of models as a `layer` or a :class:`~qmla.BranchQMLA`. 
Models can reside on multiple branches. 
For each :term:`ES` included in the instance, an :term:`Exploration Tree (ET) <ET>` is built. 
On a given tree, the associated :term:`ES` determines how to proceed, 
in particular by deciding which models to consider. 
The first branch of the tree holds the initial models :math:`\mu^1 = \{ M_1^1, \dots M_n^1\}` 
for that :term:`ES`. 
After the initial models have been trained and compared on the first branch, 
the :term:`ES` uses the available information (e.g. the number of pairwise 
wins each model has) to construct a new set of models, 
:math:`\mu^2 = \{ M_1^2, \dots M_n^2\}`. 
Subsequent branches 
:math:`\mu^i`
similarly construct models 
based on the information available to the :term:`ES` so far. 

Each :class:`~qmla.BranchQMLA` is resident on its associated :term:`ES` tree, but the branch is also known
to :term:`QMLA`. Branches are assigned unique IDs by :term:`QMLA`, such that :term:`QMLA` has a 
birds-eye view of all of the mdoels on all branches on all trees 
(in general there can be multiple :term:`ES` entertained in a single :term:`instance`). 
Indeed, a useful way to think of :term:`QMLA` is as a search across a forest consisting of :math:`N` trees, 
where each leaf is a unique model, and there can be multiple leaves per branch and multiple branches per tree, 
with the ultimate goal of identifying the single best leaf for describing the :term:`system`.

When :term:`QMLA` finds that it has completed a :term:`layer`, it is ready for the next batch of work:
it checks whether the :term:`ET` has finished growing, in which case it begins the process of nominating the champion 
from that :term:`ES`. 
Otherwise, :term:`QMLA` calls on the :term:`ES` (via the :term:`ET` ) to request a set of models, 
which it places on its next branch, completely indifferent to how those models are generated, 
or whether they have been learned already. 
This allows for completely self-contained logic in the :term:`ES`: 
QMLA will simply learn and compare
the models it is presented - it is up to the :term:`ES` to decide how to interpret them. 
As such the core :term:`QMLA` algorithm can be thought of as a simple loop: 
while the :term:`ES` continues to return models, place those models on a branch, learn them 
and compare them. 
When all :term:`ES` indicate they are finished, nominate champions from each :term:`ET`;
compare the champions of each tree against each other, and thus determine a :term:`global champion`. 


.. _section_exploration_strategies:

Exploration Strategy
--------------------

Exploration Strategies (ES) are the engine of :term:`QMLA`. 
They specify how :term:`QMLA` should proceed at each stage, 
most importantly by determining the next set of models for :term:`QMLA` to test.
These are the primary mechanism by which most users should interface with the :term:`QMLA` framework: 
by designing an :class:`~qmla.exploration_strategies.ExplorationStrategy` which implements the 
user-specific logic required. 
In particular, each :term:`ES` must provide a :meth:`~qmla.exploration_strategies.ExplorationStrategy.generate_models` 
method to construct models given information about the previous models' training/comparisons.
User :term:`ES` classes can be used to specify parameters required throughout the :term:`QMLA` protocol. 
These are all detailed in the ``setup`` methods of the :class:`~qmla.exploration_strategies.ExplorationStrategy` class;
users should familiarise themselves with these settings before proceeding. 


At minimum, a functional :term:`ES` should look like: 

.. code-block:: python

    class UserExplorationStrategy(qmla.ExplorationStrategy):
        def __init__(
            self,
            exploration_rules,
            true_model=None,
            **kwargs
        ):
            super().__init__(
                exploration_rules=exploration_rules,
                true_model=true_model,
                **kwargs
            )
            self.true_model = 'pauliSet_1_x_d1+pauliSet_1_y_d1'
        
An example of :term:`ES` design including several parameter settings is 

.. code-block:: python

    from qmla.shared_functionality import experiment_design_heuristics as edh

    class UserExplorationStrategy(qmla.ExplorationStrategy):
        def __init__(
            self,
            exploration_rules,
            true_model=None,
            **kwargs
        ):
            super().__init__(
                exploration_rules=exploration_rules,
                true_model=true_model,
                **kwargs
            )
            # Overwrite true model
            self.true_model = 'pauliSet_1_x_d1+pauliSet_1_y_d1'

            # Overwrite modular functionality
            self.model_heuristic_function = edh.VolumeAdaptiveParticleGuessHeuristic

            # Overwrite parameters
            self.max_num_qubits = 2
            self.num_probes = 10
            self.qinfer_resampler_a = 0.95

            # User specific attributes (not available by default in QMLA)
            self.model_base_terms = [
                "pauliSet_1_x_d2", 
                "pauliSet_1_y_d2", 
                "pauliSet_1_z_d2", 
                "pauliSet_2_x_d2", 
                "pauliSet_2_y_d2", 
                "pauliSet_2_z_d2", 
            ]
            self.search_exhausted = False


        def generate_models(
            self, 
            model_list,
            **kwargs
        ):
            if self.spawn_stage[-1] == None: 
                # Use spawn_stage for easy signals between calls to this method
                # e.g. to alter the functionality after some condition is method

                self.spawn_stage.append("one_parameter_models")
                return self.model_base_terms

            previous_champion = model_list[0] 
            champion_terms = previous_champion.split("+")
            nonpresent_terms = list(set(self.model_base_terms) - set(champion_terms))
            new_models = [
                "{}+{}".format(previous_champion, term) for term in nonpresent_terms
            ]                

            if len(new_models) == 1:
                # After this, there will be no more to test, 
                # so signal to QMLA that this ES is finished. 
                self.search_exhausted = True

            return new_models

        def check_tree_completed(
            self,
            spawn_step,
            **kwargs
        ):
            r"""
            QMLA asks the exploration tree whether it has finished growing; 
            the exploration tree queries the exploration strategy through this method.
            """
            return self.search_exhausted


Each :term:`ES` is assigned a unique :term:`Exploration Tree (ET) <ET>`, 
although most users need not alter the infrastructure of the :term:`ET` or :term:`QMLA`. 
We detail two :term:`ES` examples which are used in publications: 
greedy term addition for the study of an electron spin in a nitrogen vacancy centre
and genetic algorithm for generic target systems, with an example Hesienberg-XYZ :term:`system`.


Models
------
Models are specified by a string of terms separated by ``+``,
e.g. ``pauliSet_1_x_d1+pauliSet_1_y_d1``. 
Model names are unique and are assigned a ``model_id`` upon generation within :class:`~qmla.QuantumModelLearningAgent` : 
:term:`QMLA` will recognise if a model string has already been proposed and therefore been
assigned a ``model_id``, rather than retraining models which is computationally expensive.
The uniqueness of models is ensured by the terms being sorted alphabetically internally within the string 
(``pauliSet_1_x_d1+pauliSet_1_y_d1`` instead of ``pauliSet_1_y_d1+pauliSet_1_x_d1``), 
but note :term:`QMLA` ensures this internally so users do not need to enfore it in their 
:meth:`~qmla.exploration_strategies.ExplorationStrategy.generate_models`.

The strings are processed into models as follows. 
By separating models into their terms (``model_name.split('+')``), 
the cardinality (number of terms, :math:`n`) is found. 
An :math:`n-` dimensional Gaussian is constructed to represent the 
parameter distribution for the model; 
individual parameters can be specified in ``gaussian_prior_means_and_widths``
of :meth:`~qmla.exploration_strategies.ExplorationStrategy._setup_model_learning`. 
The terms are then processed into matrices. 
A number of :ref:`section_string_processing` functions are available by default;
new processing functions can be added by the user but must be incorporated in 
:func:`~qmla.process_string_to_matrix` so that :term:`QMLA` will know where to find them.


Quantum Hamiltonian Learning
----------------------------

The algorithm for parameter learning when a model is known or presumed. 



Bayes factors
----------------------------

The quantity which is used to distinguish between models. 

Exploration Strategy tree
----------------------------
The object which manages a single Exploration Strategy. 
Consists of a number of branches. 
Branches are shared with the parent :term:`QMLA` instance: 
the branch is indexed uniquely by :term:`QMLA`. i.e. it is possible for the 
branch list of a :term:`ES` tree to be `[1, 4, 5, 7, 8, 9]` etc, since :term:`QMLA` 
is in charge of this. 

.. _section_modular_functionality: 

Modular functionality
---------------------

A number of functions are modular, so they can be set by a ES. 
The function can either be overwritten in the :term:`ES` method, 
or set using pointers to modular options. 
    .. seealso:: :meth:`~qmla.exploration_strategies.ExplorationStrategy._setup_modular_functions`. 


.. _section_probes:
Probes
------------

The `probe` is the input state used during the learning procedure. 
Different probes permit different biases on the information available to the
algorithm; it is essential to consider which probes are appropriate for learning
different classes of models. 

Experiment design
----------------------------

A crucial aspect of the QHL subroutine is the design of experiments 
which provide informative data which allows meaningful 
updates of the prior distribution. 

.. _section_analysis:

Output and Analysis
-------------------

When a run is launched (either locally or remotely), a results directory 
is built for that run. 
In that directory, results are stored in several formats from each instance. 

By default, :term:`QMLA` provides a set of analyses, generating several plots
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

Within the :ref:`section_launch` scripts, there is a ``plot_level`` variable which informs :term:`QMLA` of how many plots to produce by default. 
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


.. _section_launch:

Launch
------

How to launch :term:`QMLA`. 

>>> # this is a code example
