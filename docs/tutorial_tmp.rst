
In a text editor, open ; here we will ensure that we are running the
algorithm, with 5 experiments and 20 particles, on the
:term:`Exploration Strategy` named . Ensure the first few lines of read:

::

    #!/bin/bash

    ##### -------------------------------------------------- #####
    # QMLA run configuration
    ##### -------------------------------------------------- #####
    num_instances=2 # number of instances in run
    run_qhl=0 # perform QHL on known (true) model
    run_qhl_multi_model=0 # perform QHL for defined list of models
    experiments=2 # number of experiments
    particles=10 # number of particles
    plot_level=5


    ##### -------------------------------------------------- #####
    # Choose an exploration strategy 
    # This will determine how QMLA proceeds. 
    ##### -------------------------------------------------- #####
    exploration_strategy="TestInstall"

Now we can run Ensure the terminal running redis is kept active, and
open a separate terminal window. We must activate the Python virtual
environment configured for :term:`QMLA`, which we set up in
[listing:qmla\_setup]. Then, we navigate to the :term:`QMLA` directory,
and launch:

::


    # activate the QMLA Python virtual environment 
    source qmla_test/qmla-env/bin/activate

    # move to the QMLA directory 
    cd qmla_test/QMLA
    # Run QMLA
    cd launch   
    ./local_launch.sh

There may be numerous warnings, but they should not affect whether
:term:`QMLA` has succeeded; :term:`QMLA` will any significant error.
Assuming the has completed successfully, :term:`QMLA` stores the run’s
results in a subdirectory named by the date and time it was started. For
example, if the was initialised on January :math:`1^{st}` at 01:23,
navigate to the corresponding directory by

::

    cd results/Jan_01/01_23

For now it is sufficient to notice that the code has successfully: it
should have generated (in ) files like and .

Custom 
=======

Next, we design a basic :term:`Exploration Strategy`, for the purpose of
demonstrating how to the algorithm. are placed in the directory . To
make a new one, navigate to the exploration strategies directory, make a
new subdirectory, and copy the template file.

::


    cd ~/qmla_test/QMLA/exploration_strategies/
    mkdir custom_es

    # Copy template file into example
    cp template.py custom_es/example.py
    cd custom_es

Ensure :term:`QMLA` will know where to find the :term:‘Exploration
Strategy‘ by importing everything from the custom :term:‘Exploration
Strategy‘ directory into to the main module. Then, in the directory,
make a file called which imports the new :term:`Exploration Strategy`
from the file. To add any further inside the directory , include them in
the custom , and they will automatically be available to :term:`QMLA`.

::


    # inside qmla/exploration_strategies/custom_es
    #  __init__.py    
    from qmla.exploration_strategies.custom_es.example import *

    # inside qmla/exploration_strategies, add to the existing
    # __init__.py 
    from qmla.exploration_strategies.custom_es import *

Now, change the structure (and name) of the :term:`Exploration Strategy`
inside . Say we wish to target the

.. math::

   \label{eqn:example_es_true_ham}
       \begin{split}
           \al &= \irow{ \alpha_{1,2} & \alpha_{2,3} & \alpha_{3,4}} \\
           \terms &= \icol{ \hat{\sigma}_{z}^1 \otimes \hat{\sigma}_{z}^2 \\ \hat{\sigma}_{z}^2 \otimes \hat{\sigma}_{z}^3  \\ \hat{\sigma}_{z}^3 \otimes \hat{\sigma}_{z}^4 } \\
           \Longrightarrow \ho &= \hat{\sigma}_{z}^{(1,2)} \hat{\sigma}_{z}^{(2,3)} \hat{\sigma}_{z}^{(3,4)} \\
       \end{split}

:term:`QMLA` interprets models as strings, where terms are separated by
, and parameters are implicit. So the target model in
[eqn:example\_es\_true\_ham] will be given by

.. math:: \ttt{pauliSet\_1J2\_zJz\_d4+pauliSet\_2J3\_zJz\_d4+pauliSet\_3J4\_zJz\_d4}.

Adapting the template :term:`Exploration Strategy` slightly, we can
define a model generation strategy with a small number of hard coded
candidate models introduced at the first branch of the . We will also
set the parameters of the terms which are present in :math:`\ho`, as
well as the range in which to search parameters. Keeping the s at the
top of the , rewrite the :term:`Exploration Strategy` as:

::

    class ExampleBasic(
        exploration_strategy.ExplorationStrategy
    ):

        def __init__(
            self,
            exploration_rules,
            true_model=None,
            **kwargs
        ):
            self.true_model = 'pauliSet_1J2_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_3J4_zJz_d4'
            super().__init__(
                exploration_rules=exploration_rules,
                true_model=self.true_model,
                **kwargs
            )

            self.initial_models = None
            self.true_model_terms_params = {
                'pauliSet_1J2_zJz_d4' : 2.5,
                'pauliSet_2J3_zJz_d4' : 7.5,
                'pauliSet_3J4_zJz_d4' : 3.5,
            }
            self.tree_completed_initially = True
            self.min_param = 0
            self.max_param = 10

        def generate_models(self, **kwargs):

            self.log_print(["Generating models; spawn step {}".format(self.spawn_step)])
            if self.spawn_step == 0:
                # chains up to 4 sites
                new_models = [
                    'pauliSet_1J2_zJz_d4',
                    'pauliSet_1J2_zJz_d4+pauliSet_2J3_zJz_d4',
                    'pauliSet_1J2_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_3J4_zJz_d4',
                ]
                self.spawn_stage.append('Complete')

            return new_models

To run the example :term:`Exploration Strategy` for a meaningful tests,
return to the of [listing:local\_launch], but change some of the
settings:

::

    prt=2000
    exp=500
    run_qhl=1
    exploration_strategy=ExampleBasic

Run locally again as in [listing:launch\_example]; then move to the as
in [listing:results\_directory].

Analysis
========

:term:`QMLA` stores results and generates plots over the entire range of
the algorithm, i.e. the , and models. The depth of analysis performed
automatically is set by the user control in ; for , only the most
crucial figures are generated, while generates plots for every
individual model considered. For model searches across large model
spaces and/or considering many candidates, excessive plotting can cause
considerable slow-down, so users should be careful to generate plots
only to the degree they will be useful. Next we show some examples of
the available plots.

Model analysis
--------------

We have just run for the model in [eqn:example\_es\_true\_ham] for a
single instance, using a reasonable number of particles and experiments,
so we expect to have trained the model well. -level results are stored
(e.g. for the instance with ) in . Individual models’ insights can be
found in , e.g. the model’s [fig:qmla\_learning\_summary], and in
[fig:qmla\_model\_dynamics].

 analysis
---------

Now we can run the full :term:`QMLA` algorithm, i.e. train several
models and determine the most suitable. :term:`QMLA` will call the
method of the :term:`Exploration Strategy`, set in [listing:basic\_es],
which tells :term:`QMLA` to construct three models on the first branch,
then terminate the search. Here we need to train and compare all models
so it takes considerably longer to run: the purpose of testing, we
reduce the resources so the entire algorithm runs in about 15 minutes.
Some applications will require significantly more resources to learn
effectively. In realistic cases, these processes are run in parallel, as
we will cover in [apdx:paralllel\_processing].

Reconfigure a subset of the settings in the script
([listing:local\_launch]) and run it again:

::

    exp=250
    prt=1000
    run_qhl=0
    exploration_strategy=ExampleBasic

In the corresponding , navigate to , where instance level analysis are
available.

::

    cd results/Jan_01/01_23/instances/qmla_1

Figures of interest here show the composition of the models
([fig:qmla\_model\_composition]), as well as the between candidates
([fig:qmla\_bayes\_factors]). Individual model comparisons – i.e. – are
shown in [fig:qmla\_bayes\_factor\_comparison], with the dynamics of all
candidates shown in [fig:qmla\_branch\_dynamics]. The probes used during
the training of all candidates are also plotted
([fig:qmla\_training\_probes]).

 analysis
---------

Considering a number of together is a **. In general, this is the level
of analysis of most interest: an individual instance is liable to errors
due to the probabilistic nature of the model training and generation
subroutines. On average, however, we expect those elements to perform
well, so across a significant number of instances, we expect the average
outcomes to be meaningful.

Each has an script to generate plots at the level.

::

    cd results/Jan_01/01_23
    ./analyse.sh

Run level analysis are held in the main and several sub-directories
created by the script. Here, we recommend running a number of with very
few resources so that the test finishes quickly [1]_. The results will
therefore be meaningless, but allow fo elucidation of the resultant
plots. First, reconfigure some settings of [listing:local\_launch] and
launch again.

::

    num_instances=10
    exp=20
    prt=100
    run_qhl=0
    exploration_strategy=ExampleBasic

Some of the generated analysis are shown in . The number of for each
model, i.e. their ** are given in [fig:qmla\_win\_rates]. The *top
models*, i.e. those with highest , analysed further: the average
parameter estimation progression for :math:`\ho` – including only the
where :math:`\ho` was deemed champion – are shown in
[fig:champ\_param\_progression]. Irrespecitve of the , the rate with
which each term is found in the (:math:`\hat{t} \in \hp`) indicates the
that the term is really present; these rates – along with the parameter
values learned – are shown in [fig:qmla\_branch\_dynamics]. The from
each can attempt to reproduce system dynamics: we group together these
reproductions for each model in [fig:run\_dynamics].

.. figure:: qmla_run_data/Jan_17/22_27/performance/dynamics.pdf
   :alt:  Run plot : median dynamics of the . The models which won most
   are shown together in the top panel, and individually in the lower
   panels. The median dynamics from the models’ learnings in its winning
   are shown, with the shaded region indicating the 66% confidence
   region.

    Run plot : median dynamics of the . The models which won most are
   shown together in the top panel, and individually in the lower
   panels. The median dynamics from the models’ learnings in its winning
   are shown, with the shaded region indicating the 66% confidence
   region. 

Parallel implementation
=======================

We provide utility to run :term:`QMLA` on parallel processes. Individual
models’ training can run in parallel, as well as the calculation of
between models. The provided script is designed for job scheduler
running on a compute cluster. It will require a few adjustments to match
the system being used. Overall, though, it has mostly a similar
structure as the script used above.

:term:`QMLA` must be downloaded on the compute cluster as in
[listing:qmla\_setup]; this can be a new fork of the repository, though
it is sensible to test installation locally as described in this chapter
so far, then *push* that version, including the new :term:‘Exploration
Strategy‘, to Github, and cloning the latest version. It is again
advisable to create a Python virtual environment in order to isolate
:term:`QMLA` and its dependencies [2]_. Open the parallel launch script,
, and prepare the first few lines as

::

    #!/bin/bash

    ##### -------------------------------------------------- #####
    # QMLA run configuration
    ##### -------------------------------------------------- #####
    num_instances=10 # number of \glspl{instance} in run
    run_qhl=0 # perform QHL on known (true) model
    run_qhl_multi_model=0 # perform QHL for defined list of models
    experiments=250
    particles=1000
    plot_level=5


    ##### -------------------------------------------------- #####
    # Choose an exploration strategy 
    # This will determine how QMLA proceeds. 
    ##### -------------------------------------------------- #####
    exploration_strategy="ExampleBasic"

When submitting jobs to schedulers like , we must specify the time
required, so that it can determine a fair distribution of resources
among users. We must therefore *estimate* the time it will take for an
to complete: clearly this is strongly dependent on the numbers of
experiments (:math:`\Ne`) and particles (:math:`\Np`), and the number of
models which must be trained. :term:`QMLA` attempts to determine a
reasonable time to request based on the attribute of the
:term:`Exploration Strategy`, by calling . In practice, this can be
difficult to set perfectly, so the attribute of the :term:‘Exploration
Strategy‘ can be used to correct for heavily over- or under-estimated
time requests. Instances are run in parallel, and each trains/compares
models in parallel. The number of processes to request, :math:`N_c` for
each is set as in the :term:`Exploration Strategy`. Then, if there are
:math:`N_r` in the run, we will be requesting the job scheduler to admit
:math:`N_r` distinct jobs, each requiring :math:`N_c` processes, for the
time specified.

The script works together with , though note a number of steps in the
latter are configured to the cluster and may need to be adapted. In
particular, the first command is used to load the redis utility, and
later lines are used to initialise a redis server. These commands will
probably not work with most machines, so must be configured to achieve
those steps.

::


    module load tools/redis-4.0.8

    ... 

    SERVER_HOST=$(head -1 "$PBS_NODEFILE")
    let REDIS_PORT="6300 + $QMLA_ID"

    cd $LIBRARY_DIR
    redis-server RedisDatabaseConfig.conf --protected-mode no --port $REDIS_PORT & 
    redis-cli -p $REDIS_PORT flushall

When the modifications are finished, :term:`QMLA` can be launched in
parallel similarly to the local version:

::

    source qmla_test/qmla-env/bin/activate

    cd qmla_test/QMLA/launch
    ./parallel_launch.sh

Jobs are likely to queue for some time, depending on the demands on the
job scheduler. When all jobs have finished, results are stored as in the
local case, in , where can be used to generate a series of automatic
analyses.

Customising 
============

User interaction with the :term:`QMLA` codebase should be achieveable
primarily through the framework. Throughout the algorithm(s) available,
:term:`QMLA` calls upon the :term:`Exploration Strategy` before
determining how to proceed. The usual mechanism through which the
actions of :term:`QMLA` are directed, is to set attributes of the
:term:`Exploration Strategy` class: the complete set of influential
attributes are available at :raw-latex:`\cite{qmla_docs}`.

:term:`QMLA` directly uses several methods of the :term:‘Exploration
Strategy‘ class, all of which can be overwritten in the course of
customising an :term:`Exploration Strategy`. Most such methods need not
be replaced, however, with the exception of , which is the most
important aspect of any :term:`Exploration Strategy`: it determines
which models are built and tested by :term:`QMLA`. This method allows
the user to impose any logic desired in constructing models; it is
called after the completion of every branch of the on the
:term:`Exploration Strategy`.

Greedy search
-------------

A first non-trivial :term:`Exploration Strategy` is to build models
greedily from a set of *primitive* terms,
:math:`\termset = \{ \hat{t} \} `. New models are constructed by
combining the previous branch champion with each of the remaining,
unused terms. The process is repeated until no terms remain.

.. figure:: appendix/figures/greedy_exploration_strategy.pdf
   :alt:  Greedy search mechanism. **Left**, a set of primitive terms,
   :math:`\termset`, are defined in advance. **Right**, models are
   constructed from :math:`\termset`. On the first branch, the primitve
   terms alone constitute models. Thereafter, the strongest model
   (marked in green) from the previous branch is combined with all the
   unused terms.

    Greedy search mechanism. **Left**, a set of primitive terms,
   :math:`\termset`, are defined in advance. **Right**, models are
   constructed from :math:`\termset`. On the first branch, the primitve
   terms alone constitute models. Thereafter, the strongest model
   (marked in green) from the previous branch is combined with all the
   unused terms. 

We can compose an :term:`Exploration Strategy` using these rules, say
for

.. math:: \termset = \left\{ \hat{\sigma}_{x}^1, \ \hat{\sigma}_{y}^1, \ \hat{\sigma}_{x}^1 \otimes \hat{\sigma}_{x}^2, \ \hat{\sigma}_{y}^1 \otimes \hat{\sigma}_{y}^2 \right\}

as follows. Note the termination criteria must work in conjunction with
the model generation routine. Users can overwrite the method for custom
logic, although a straightforward mechanism is to use the attribute of
the :term:`Exploration Strategy` class: when the final element of this
list is , :term:`QMLA` will terminate the search by default. Also note
that the default termination test checks whether the number of branches
() exceeds the limit , which must be set artifically high to avoid
ceasing the search too early, if relying solely on . Here we demonstrate
how to impose custom logic to terminate the seach also.

::

    class ExampleGreedySearch(
        exploration_strategy.ExplorationStrategy
    ):
        r"""
        From a fixed set of terms, construct models iteratively, 
        greedily adding all unused terms to separate models at each call to the generate_models. 

        """

        def __init__(
            self,
            exploration_rules,
            **kwargs
        ):
            
            super().__init__(
                exploration_rules=exploration_rules,
                **kwargs
            )
            self.true_model = 'pauliSet_1_x_d3+pauliSet_1J2_yJy_d3+pauliSet_1J2J3_zJzJz_d3'
            self.initial_models = None
            self.available_terms = [
                'pauliSet_1_x_d3', 'pauliSet_1_y_d3', 
                'pauliSet_1J2_xJx_d3', 'pauliSet_1J2_yJy_d3'
            ]
            self.branch_champions = []
            self.prune_completed_initially = True
            self.check_champion_reducibility = False

        def generate_models(
            self,
            model_list,
            **kwargs
        ):
            self.log_print([
                "Generating models in tiered greedy search at spawn step {}.".format(
                    self.spawn_step, 
                )
            ])
            try:
                previous_branch_champ = model_list[0]
                self.branch_champions.append(previous_branch_champ)
            except:
                previous_branch_champ = ""

            if self.spawn_step == 0 :
                new_models = self.available_terms
            else:
                new_models = greedy_add(
                    current_model = previous_branch_champ, 
                    terms = self.available_terms
                )

            if len(new_models) == 0:
                # Greedy search has exhausted the available models;
                # send back the list of branch champions and terminate search.
                new_models = self.branch_champions
                self.spawn_stage.append('Complete')

            return new_models

    def greedy_add(
        current_model, 
        terms,
    ):
        r""" 
        Combines given model with all terms from a set.
        
        Determines which terms are not yet present in the model, 
        and adds them each separately to the current model. 

        :param str current_model: base model
        :param list terms: list of strings of terms which are to be added greedily. 
        """

        try:
            present_terms = current_model.split('+')
        except:
            present_terms = []
        nonpresent_terms = list(set(terms) - set(present_terms))
        
        term_sets = [
            present_terms+[t] for t in nonpresent_terms
        ]

        new_models = ["+".join(term_set) for term_set in term_sets]
        
        return new_models

This can be implemented locally or in parallel as described above, and
analysed as in [listing:analysing\_run], generating figures in
accordance with the set by the user in the launch script. Outputs can
again be found in the subdirectory, including a map of the models
generated, as well as the branches they reside on, and the between
candidates, [fig:example\_es\_greedy].

Tiered greedy search
--------------------

We provide one final example of a non-trivial :term:‘Exploration
Strategy‘: tiered greedy search. Similar to the idea of
[sec:greedy\_search], except terms are introduced hierarchically: sets
of terms :math:`\termset_1, \termset_2, \dots \termset_n` are each
examined greedily, where the overall strongest model of one tier forms
the seed model for the subsequent tier. This is depicted in the main
text in [fig:greedy\_search]. A corresponding :term:‘Exploration
Strategy‘ is given as follows.

::


    class ExampleGreedySearchTiered(
        exploration_strategy.ExplorationStrategy
    ):
        r"""
        Greedy search in tiers.

        Terms are batched together in tiers; 
        tiers are searched greedily; 
        a single tier champion is elevated to the subsequent tier. 

        """

        def __init__(
            self,
            exploration_rules,
            **kwargs
        ):
            super().__init__(
                exploration_rules=exploration_rules,
                **kwargs
            )
            self.true_model = 'pauliSet_1_x_d3+pauliSet_1J2_yJy_d3+pauliSet_1J2J3_zJzJz_d3'
            self.initial_models = None
            self.term_tiers = {
                1 : ['pauliSet_1_x_d3', 'pauliSet_1_y_d3', 'pauliSet_1_z_d3' ],
                2 : ['pauliSet_1J2_xJx_d3', 'pauliSet_1J2_yJy_d3', 'pauliSet_1J2_zJz_d3'],
                3 : ['pauliSet_1J2J3_xJxJx_d3', 'pauliSet_1J2J3_yJyJy_d3', 'pauliSet_1J2J3_zJzJz_d3'],
            }
            self.tier = 1
            self.max_tier = max(self.term_tiers)
            self.tier_branch_champs = {k : [] for k in self.term_tiers} 
            self.tier_champs = {}
            self.prune_completed_initially = True
            self.check_champion_reducibility = True

        def generate_models(
            self,
            model_list,
            **kwargs
        ):
            self.log_print([
                "Generating models in tiered greedy search at spawn step {}.".format(
                    self.spawn_step, 
                )
            ])

            if self.spawn_stage[-1] is None:
                try:
                    previous_branch_champ = model_list[0]
                    self.tier_branch_champs[self.tier].append(previous_branch_champ)
                except:
                    previous_branch_champ = None

            elif "getting_tier_champ" in self.spawn_stage[-1]:
                previous_branch_champ = model_list[0]
                self.log_print([
                    "Tier champ for {} is {}".format(self.tier, model_list[0])
                ])
                self.tier_champs[self.tier] = model_list[0]
                self.tier += 1
                self.log_print(["Tier now = ", self.tier])
                self.spawn_stage.append(None) # normal processing

                if self.tier > self.max_tier:
                    self.log_print(["Completed tree for ES"])
                    self.spawn_stage.append('Complete')
                    return list(self.tier_champs.values())
            else:
                self.log_print([
                    "Spawn stage:", self.spawn_stage
                ])

            new_models = greedy_add(
                current_model = previous_branch_champ, 
                terms = self.term_tiers[self.tier]
            )
            self.log_print([
                "tiered search new_models=", new_models
            ])

            if len(new_models) == 0:
                # no models left to find - get champions of branches from this tier
                new_models = self.tier_branch_champs[self.tier]
                self.log_print([
                    "tier champions: {}".format(new_models)
                ])
                self.spawn_stage.append("getting_tier_champ_{}".format(self.tier))
            return new_models

        def check_tree_completed(
            self,
            spawn_step,
            **kwargs
        ):
            r"""
            QMLA asks the exploration tree whether it has finished growing; 
            the exploration tree queries the exploration strategy through this method
            """
            if self.tree_completed_initially:
                return True
            elif self.spawn_stage[-1] == "Complete":
                return True
            else:
                return False
        

    def greedy_add(
        current_model, 
        terms,
    ):
        r""" 
        Combines given model with all terms from a set.
        
        Determines which terms are not yet present in the model, 
        and adds them each separately to the current model. 

        :param str current_model: base model
        :param list terms: list of strings of terms which are to be added greedily. 
        """

        try:
            present_terms = current_model.split('+')
        except:
            present_terms = []
        nonpresent_terms = list(set(terms) - set(present_terms))
        
        term_sets = [
            present_terms+[t] for t in nonpresent_terms
        ]

        new_models = ["+".join(term_set) for term_set in term_sets]
        
        return new_models

with corresponding results in [fig:example\_es\_tiered\_greedy].

.. [1]
   This will take about ten minutes

.. [2]
   Indeed it is sensible to do this for any Python development project.
