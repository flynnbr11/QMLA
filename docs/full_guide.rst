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
After the initial models have been trained and compared on :math:`\mu^1`, 
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
the models it is presented - it is the responsibility of the :term:`ES` to interpret them. 
As such, the core :term:`QMLA` algorithm can be thought of as a simple loop: 
while the :term:`ES` continues to return models, place those models on a branch, learn them 
and compare them. 
When all :term:`ES` indicate they are finished, nominate champions from each :term:`ET`;
compare the champions of each tree against each other, and thus determine a :term:`global champion`. 


.. _section_exploration_strategies:

Exploration Strategy
--------------------

:term:`Exploration Strategies (ES) <ES>` are the engine of :term:`QMLA`. 
The :term:`ES` specifies how :term:`QMLA` should proceed at each stage, 
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
        
An example of :term:`ES` design, including a simple greedy-addition model generation method as well as seeting several parameter settings, is:

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
            self.model_heuristic_subroutine = edh.VolumeAdaptiveParticleGuessHeuristic

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

In order to implement a new :term:`ES`, :term:`QMLA` searches in the directory ``qmla/exploration_strategies``, 
so the user's :term:`ES` must be ``import`` ed to the ``qmla/exploration_strategies/__init__.py``.
:term:`QMLA` retrieves the :term:`ES` through calls to the function :func:`~qmla.get_exploration_class`, 
by searching for the :term:`ES` specified in the :ref:`section_launch` script. 
For example, the launch script (e.g. at ``qmla/launch/local_launch.sh``) should be updated to call the user's :term:`ES`, e.g.

.. code-block:: bash

    #!/bin/bash

    ###############
    # QMLA run configuration
    ###############
    num_instances=1
    run_qhl=0 
    experiments=500 
    particles=2000 

    ###############
    # Choose an exploration strategy 
    ###############

    exploration_strategy='UserExplorationStrategy'

A complete step-by-step example of implementing custom :term:`ES` is given in :ref:`section_tutorial`.
Users should ensure they understand the options for launching :term:`QMLA` as outlined in :ref:`section_launch`. 

Each :term:`ES` is assigned a unique :term:`Exploration Tree (ET) <ET>`, 
although most users need not alter the infrastructure of the :term:`ET` or :term:`QMLA`. 


Models
------

Construction
~~~~~~~~~~~~~

Models are specified by a string of terms separated by ``+``,
e.g. ``pauliSet_1_x_d1+pauliSet_1_y_d1``. 
Model names are unique and are assigned a ``model_id`` upon generation within :class:`~qmla.QuantumModelLearningAgent` : 
:term:`QMLA` will recognise if a model string has already been proposed and therefore been
assigned a ``model_id``, rather than retraining models which is computationally expensive.
The uniqueness of models is ensured by the terms being sorted alphabetically internally within the string 
(e.g. ``pauliSet_1_x_d1+pauliSet_1_y_d1`` instead of ``pauliSet_1_y_d1+pauliSet_1_x_d1``), 
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
:func:`~qmla.process_basic_operator` so that :term:`QMLA` will know where to find them.


Classes
~~~~~~~
Models are central to the :term:`QMLA` framework so it sensible to identify their core functionality
so we can design software to facilitate them. 
In particular, there are three forms of classes which each depict models, but fulfil different roles. 
In brief, these classes are 

:class:`~qmla.ModelInstanceForLearning`
    Class used for the training of individual models. 

:class:`~qmla.ModelInstanceForComparison`
    Class used for comparing models which have already been trained

:class:`~qmla.ModelInstanceForStorage`
    Class retained by :class:`~qmla.QuantumModelLearningAgent`, storing the results of the model's 
    training and comparisons.

We next detail each of these roles of the model concept.

.. _section_training: 

Training
~~~~~~~~~
:term:`QMLA` relies on a subroutine for training individual candidate models: 
it is imperative that a given candidate is optimised against the :term:`system`, 
as otherwise it might appear as a relatively weak candidate compared with its potential. 
In principle, any parameter learning subroutine can fulfil this role in :term:`QMLA`, 
such as Hamiltonian tomography or using neural networks for parameter estimation. 
The in-built facility for this subroutine is :term:`quantum Hamiltonian Learning (QHL) <QHL>`. 
We do not descibe the :term:`QHL` protocol here but readers can refer to [WGFC13a]_, [WGFC13b]_ for details. 

:class:`~qmla.ModelInstanceForLearning` is a disposable class which instatiates indepdendently from :class:`~qmla.QuantumModelLearningAgent`.
It trains a given model via :func:`qmla.remote_learn_model_parameters`, performs analysis on the trained model, 
summarises the outcome of the training and sends a concise data packet to the database, before being deleted. 
The model training refers to quantum Hamiltonian learning, performed in conjunction with [QInfer]_, 
via :meth:`~qmla.ModelInstanceForLearning.update_model`.
Importantly, :term:`QMLA` trains models simply by calling :func:`qmla.remote_learn_model_parameters`: 
this function acts \emph{remotely} and is therefore independent, allowing for multiple instance 
of the function and :class:`~qmla.ModelInstanceForLearning` to run simultaneously. 
As such, this class mechanism allows for \emph{parallel processing} within :term:`QMLA`, 
enabling speedup proportional to the number of processes available (for the model training stages). 


Comparisons
~~~~~~~~~~~
Like the training subroutine, in principle :term:`QMLA` can operate with any model comparison subroutine, 
but in practice we use :term:`Bayes factors (BF) <BF>`. 
This is a quantity which is used to distinguish between models. 

:class:`~qmla.ModelInstanceForComparison` is a disposable class which reads the redis database to retrieve information about the 
trainng of the given ``model_id``. 
It then reconstructs the model, e.g. based on the final estimated mean of the parameter distribution. 
Then, to compare models, :func:`~qmla.remote_bayes_factor_calculation` interfaces two instances of
:class:`~qmla.ModelInstanceForComparison` such that each model is exposed to the opponent's experiments for further updates, 
such that the two models under consideration have identical experiment records 
(at least partially whereupon the BF is based), allowing for meaningful comparison among the two.  
This is achieved through :meth:`~qmla.ModelInstanceForComparison.update_log_likelihood`.

Similiar to the training stage, :func:`~qmla.remote_bayes_factor_calculation` can be run in parallel to provide a large speedup to the 
overall :term:`QMLA` protocol. 

Storage
~~~~~~~
Finally, :class:`~qmla.ModelInstanceForStorage` is a much smaller onject than the previous forms of the model, 
which retains only the useful information for storage/analysis within the bigger picture in 
:class:`~qmla.QuantumModelLearningAgent`. 
It retrieves the succinct summaries of the training/comparisons pertainng to a single model 
which are stored on the redis database, allowing for later anlaysis as required by :term:`QMLA`.
The retrieval of trained model data is performed in :meth:`~qmla.ModelInstanceForStorage.model_update_learned_values`. 

.. _section_modular_functionality: 

Modular functionality
---------------------
A large amount of the design of an :term:`ES` involves implementation of subroutines: 
there are a number of methods of :class:`~qmla.exploration_strategies.ExplorationStrategy` which can be overwritten 
in order to achieve functionality specific to the target :term:`system`. 
In this section we describe these subroutines. 
Many of the subroutines have a number of sensible implementations: we make :term:`QMLA` \emph{modular} 
by providing a set of pre-built subroutines, and allow them to be easily swapped
so that a new :term:`ES` can benefit from arbitrary combiniations of subroutines. 
The subroutines are called by wrapper methods in :class:`~qmla.exploration_strategies.ExplorationStrategy`;
to set which function is called, change the attribute in the definition of the custom :term:`ES`. 
Alternatively, directly overwrite the wrapper. 
The pre-built functions are in ``qmla/shared_functionality``. 

Within :class:`~qmla.exploration_strategies.ExplorationStrategy`, these modular functions are set in
:meth:`~qmla.exploration_strategies.ExplorationStrategy._setup_modular_subroutines`. 

An example of setting each of these subroutines is 

.. code-block:: python

    from qmla.shared_functionality import experiment_design_heuristics as edh
    from qmla.shared_functionality import expectation_value_functions as ev
    from qmla.shared_functionality import \
        qmla.shared_functionality.probe_set_generation as probes
    from qmla.shared_functionality import qinfer_model_interface as qii
    from qmla.shared_functionality import prior_distributions
    from qmla.shared_functionality import latex_model_names as lm

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

            # Overwrite expectation value subroutine
            self.expectation_value_subroutine = ev.default_expectation_value

            # Overwrite probe generation subroutines
            self.system_probes_generation_subroutine = probes.plus_probes_dict
            self.plot_probes_generation_subroutine = probes.zero_state_probes

            # Overwrite exeperiment design heuristic
            self.model_heuristic_subroutine = edh.VolumeAdaptiveParticleGuessHeuristic
                    
            # Overwrite QInfer interface
            self.qinfer_model_subroutine = qii.QInferModelQMLA

            # Overwrite prior distribution subroutine
            self.prior_distribution_subroutine = priors.gaussian_prior

            # Overwrite latex mapping subroutine
            self.latex_string_map_subroutine = lm.lattice_set_grouped_pauli

.. _section_probes:

Probes
~~~~~~

The :term:`probe` is the input state used during the learning procedure. 
Different probes permit different biases on the information available to the
algorithm; it is essential to consider which probes are appropriate for learning
different classes of models. 
In general the training procedure loops over the available probes, to minimise the chance of 
favouring some models due to bias inherent in the probe. 
For example, if the probe is (close to) an eigenstate of one candidate model, that model
will never learn effectively since there will be little variation in measurements correspdonding 
to evolving the probe according to that model. 
Intuitively, the most informative probe for a given model is a superposition of its eigenstates, 
since any evolution in this basis will be reflected by the measurement. 

The default set of probes is to use a random set. 
Alternative sets include :math:`|+\rangle^{\otimes N}` or :math:`|0\rangle^{\otimes N}`.
Probes are generated in a dictionary, of which the keys are ``(probe_id, num_qubits)``; 
``probe_id`` runs from 1 to the ``num_probes`` attribute of the :class:`~qmla.exploration_strategies.ExplorationStrategy` controls; 
the ``num_qubits`` runs from 1 to ``max_num_qubits``. 

There are a number of sets of probes required, all similarly set by specifying the subroutine:

    ``system_probes_generation_subroutine``
        Probes used for evolution on the target :term:`system`

    ``simulator_probes_generation_subroutine``
        Probes correspdonding exactly to those used on the system. 
        These should be the same so that the likelihood function is meaningful, but in realistic cases
        there may be slight differences in probe preparation, e.g. due to expected noise in an experimental system. 
        Therefore it is possible to specify a different set. 
        Note to enable this functionality, ``shared_probes`` must also be set to ``False``. 
    
    ``plot_probes_generation_subroutine``
        Probes used for plots throughout the protocol. 
        Plots should be in the same basis for consistency; 
        we generate them once per :term:`run` to save time, since the plot probes are the same everywhere. 
        The standard plotting probes are :math:`|+\rangle^{\otimes N}`.
    
    ``evaluation_probe_generation_subroutine``
        Some :term:`ES` use evaluation datasets within model selection; 
        to specify a different generator than ``system_probes_generation_subroutine``, set this attribute. 
        Defaults to ``None``. 

.. _section_edh: 

Experiment design heuristic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order for the model :ref:`section_training` to perform well, 
it is essential that the parameter learning subroutine is fed useful, meaningful data. 
We use an :term:`experiment design heuristic (EDH) <EDH>` to generate informative experiments. 
The :term:`EDH` can encompass custom logic for particular use cases, 
although the most common (particle guess heuristic [WGFC13a]_) attempts to select an evolution time :math:`t`
which can distinguish between strong and weak parameterisations (particles) based on the current 
distribution. 

Primarily the :term:`EDH` must choose an evolution time :math:`t` and :term:`probe`,
since these two together specify an entire experiment in most use cases. 
:term:`QMLA` can consider more complex experiment designs, in which case the :term:`EDH` must also 
choose informative values for all inputs. 

QInfer interface
~~~~~~~~~~~~~~~~
As mentioned, the workhorse of model :ref:`section_training` is [QInfer]_. 
The default behaviour of :class:`~qmla.shared_functionality.qinfer_model_interface.QInferModelQMLA` is to call 
:meth:`~qmla.shared_functionality.qinfer_model_interface.QInferModelQMLA.likelihood` for both the calculation of the datum from the :term:`system`,
and the likelihoods of all the particles through the simulator. 
This too can be replaced, for example if calls to the :term:`system` need to interface with a real experiment, 
or the particles should be computed through a quantum simulator. 

Prior distribution 
~~~~~~~~~~~~~~~~~~
QInfer works by taking an initial :term:`prior` distribution, which it iteratively narrows based on 
quantum likelihood estimation. 
This process of narrowing the distribution is what we call \emph{learning}: after :math:`N_E` experiments
worth of data, the mean of the remaining distribution is considered as the optimised parameterisation.

The :term:`prior` can be altered to incorporate the user's prior knowledge about the system. 
The default generator for the prior is to construct an :math:`n` dimensional Gaussian through
:func:`~qmla.shared_functionality.prior_distributions.gaussian_prior`. 
Importantly, the range of each :term:`term`'s parameter can be different, 
e.g. near-neighbour couplings having much higher frequency than distant neighbours. 
Terms' prior mean and width can be specified in ``gaussian_prior_means_and_widths``.
Terms which do not have specific means/widths in ``gaussian_prior_means_and_widths`` are assigned based on the 
:class:`~qmla.exploration_strategies.ExplorationStrategy` attributes ``min_param``, ``max_param``:
the defaults are 

    ``mean = (max_param + min_param)/2``;

    ``width= (max_param - min_param)/4``. 

To overwrite this, e.g. to change the default width of each parameter's distribution, 
users can implement a new :term:`prior` generation function to replace ``prior_distribution_subroutine``. 


.. code-block:: python

    self.gaussian_prior_means_and_widths = {
        'pauliSet_1_x_d1' : (5, 1), 
        'pauliSet_1_y_d1' : (150, 25), 
        'pauliSet_1_z_d1' : (1e6, 1e2)
    }
    self.min_param = 0
    self.max_param = 10

    self.prior_distribution_subroutine = alternatve_prior_generation


.. _section_latex_map:

Latex name mapping 
~~~~~~~~~~~~~~~~~~
Each model string format requires a method which can map the string to a Latex string. 
This is because much of the analysis automatically generated by :term:`QMLA` refers to individual 
models or terms, so it is useful that these can be rendered into a readable format, rather than 
the raw string used to generate the matrices used by the algorithm. 
The mapping function should be able to operate either on single terms or entire models strings. 
If using terms like ``pauliSet_i_t_dN``, the default :func:`~qmla.shared_functionality.latex_model_names.pauli_set_latex_name`
should work. 
Further examples, specific to models of bespoke :term:`ES` are 
:func:`~qmla.shared_functionality.latex_model_names.grouped_pauli_terms`,
:func:`~qmla.shared_functionality.latex_model_names.fermi_hubbard_latex`.

>>> from qmla.shared_functionality.latex_model_names import grouped_pauli_terms
>>> self.latex_string_map_subroutine = grouped_pauli_terms

.. _section_analysis:

Output and Analysis
-------------------
.. currentmodule:: qmla

When a run is launched (either locally or remotely), a results directory 
is built for that run. 
In that directory, results are stored in several formats from each instance. 

By default, :term:`QMLA` provides a set of analyses, generating several plots
in the sub-directories of the run's results directory. 


Analyses are available on various levels: 

    :Run: 
        results across a number of instances.

        Example: the number of instance wins for champion models. 
        
        Example: average dynamics reproduced by champion models. 
    :Instance: 
        Performance of a single insance. 
        
        Example: models generated and the branches on which they reside
    :Model: 
        Individual model performance within an instance. 
        
        Example: parameter estimation through QHL. 
        
        Example: pairwise comparison between models.

    :Comparisons:
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
There are two mechanisms for launching :term:`QMLA`: locally and in parallel. 
Both of are available through ``bash`` scripts in ``qmla/launch``. 
When launched in parallel, the model training/comparison subroutines are run 
on remote processes, e.g. in a compute cluster. 
In either case, the user has a set of top-level controls, 
bearing in mind that the majority of user requirements are implemented in the :class:`~qmla.exploration_strategies.ExplorationStrategy`.
Following the setting of these controls, the remainder of the launch script call a number of ``bash`` and ``Python``
scripts for the actual implementation, which most users should not need to alter. 


The available controls to the user are

    :num_instances: number of instance in the run

    :run_qhl: if 1, only implements :term:`QHL` on the ``true_model`` attribute of the :term:`ES`, 
        i.e. :meth:`~qmla.QuantumModelLearningAgent.run_quantum_hamiltonian_learning`. 

    :run_qhl_mulit_model: if 1, only implements :term:`QHL` on the ``qhl_models`` attribute (list) of the :term:`ES`, 
        i.e. :meth:`~qmla.QuantumModelLearningAgent.run_quantum_hamiltonian_learning_multiple_models`. 
        if *both* this and :run_qhl: are 0, then the full :term:`QMLA` protocol is run
        (:meth:`~qmla.QuantumModelLearningAgent.run_complete_qmla`). 
    
    :exp: Number of experiments used during model training
    
    :prt: Number of particles used during model training

    :plot_level:  specifies the granularity of plots generated. See :ref:`section_analysis`
    
    :debug_mode: (bool) whether to run :term:`QMLA` in degug mode. Should not be required by most users; 
        this mode merely logs further data in the instances' log files, which can be found in the :term:`run results directory`. 

    :exploration_strategy: specify the name of the :term:`ES` class to use. 

    :alt_exploration_strategies: list of alternative :term:`ES` for the case where multiple :term:`ET` s are considered.  
        This list should be in brackets with elements separated by spaces (i.e no commas). 
        Note that in ``parallel_launch.sh``, this must be enabled through setting 
        ``multiple_exploration_strategies=1``, while in ``local_launch.sh`` it is sufficient that the list is not empty. 


An example of the top few lines of ``local_launch.sh`` is then given by

.. code-block:: bash

    #!/bin/bash

    ###############
    # QMLA run configuration
    ###############
    num_instances=100
    run_qhl=0 # perform QHL on known (true) model
    run_qhl_mulit_model=0 # perform QHL for defined list of models.
    exp=500 # number of experiments
    prt=2000 # number of particles

    ###############
    # QMLA settings - user
    ###############
    plot_level=4
    debug_mode=0

    ###############
    # QMLA settings - default
    ###############
    do_further_qhl=0 
    q_id=0 
    use_rq=0
    further_qhl_factor=1
    further_qhl_num_runs=$num_instances
    plots=0
    number_best_models_further_qhl=5

    ###############
    # Choose exploration strategy/strategies
    ###############

    exploration_strategy='UserExplorationStrategy'

    alt_exploration_strategies=(
        'IsingLatticeSet'
        'Genetic'
    )

Redis server
~~~~~~~~~~~~
:term:`QMLA` uses a redis server as a database and job broker for the implementation of remote tasks. 
This is launhed automatically when using ``parallel_launch.sh``, but using ``local_launch.sh``, 
must be initiated in terminal as 

.. code-blocks:: bash

    redis-server