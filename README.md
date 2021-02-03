# Quantum Model Learning Agent

Quantum Model Learning Agent (QMLA) is a machine learning protocol for
the characterisation of quantum mechanical systems and devices. It aims
to determine the model which best explains observed data from the system
under study. It does this by considering a series of candidate models,
performing a learning procedure to optimise the performance of those
candidates, and then selecting the best candidate. New candidate models
can be constructed iteratively, using information gained so far in the
procedure, to improve the model approximation.

QMLA can operate on simulated or experimental data, or be incorporated
with an online experiment. This document provides details on QMLA’s
mechanics, and provides examples of usage. Particular attention is paid
to the concept and design of Exploration Strategies (ES), the primary
mechanism by which users ought to interact with the software. Custom ES
can be developed and plugged into the QMLA framework, in order to target
systems, run experiments and generate models according to the user’s
requirements. Several aspects of the framework are modular, allowing
users to select the combination which suits their requirements, or
easily add functionality as needed.

This `README` briefly introduces the core concepts at a high level;
[full documentation is available][] on `readthedocs`.

# Models

Models encapsulate the physics of a system. We generically refer to
models because QMLA is indifferent to the formalism employed to describe
the system. Usually we mean *Hamiltonian* models, although QMLA may also
be used to learn *Lindbladian* models.

Models are simply the mathematical objects which can be used to predict
the behaviour of a system, uniquely represented by a parameterisation.
Each term in a model is really a matrix corresponding to some physical
interaction; each such term is assigned a scalar parmeter. The total
model is a matrix, which is computed by the sum of the terms multiplied
by their parameters. For example, 1-qubit models can be constructed
using the Pauli operators
*σ̂*<sub>*x*</sub>, *σ̂*<sub>*y*</sub>, *σ̂*<sub>*z*</sub>,
e.g. *Ĥ*<sub>*x**y*</sub> = *α*<sub>*x*</sub>*σ̂*<sub>*x*</sub> + *α*<sub>*y*</sub>*σ̂*<sub>*y*</sub>
Then, *Ĥ*<sub>*x**y*</sub> is completely described by the vector 
*α* = (*α*<sub>*x*</sub>, *α*<sub>*y*</sub>) when we know the corresponding terms
*T⃗* = ( *σ̂*<sub>*x*</sub>, *σ̂*<sub>*y*</sub> ). 
In general then, models are given by *Ĥ*(*α⃗*) = *α⃗* ⋅ *T⃗*.

In the Hamiltonian (closed) formalism, terms included in the model
correspond to interactions between particles in the system. For example,
the Ising model Hamiltonian on *N* sites (spins),
*Ĥ*<sup> ⊗ *N*</sup> = *J*∑<sub>*i*</sub> = 1<sup>*N* − 1</sup>*σ̂*<sub>*i*</sub><sup>*z*</sup>*σ̂*<sub>*i+1*</sub><sup>*z*</sup>,
includes terms *σ̂*<sub>*i*</sub><sup>*z*</sup>*σ̂*<sub>*i* + 1</sub><sup>*z*</sup> 
which are the interactions between nearest neighbour sites (*i*, *i+1*).

QMLA reduces assumptions about which interactions are present, for instance by considering models
*Ĥ*<sup> ⊗ 5</sup> and *Ĥ*<sup> ⊗ 8</sup>, and determining which model
(5 or 8 spins) best describes the observed data. Moreover, QMLA
facilitates consideration of all terms independently, e.g. whether the
system is better described by a partially connected Ising lattice
*Ĥ*<sub>1</sub> or a nearest-neighbour connected Ising chain
*Ĥ*<sub>2</sub>:

*Ĥ*<sub>2</sub> = *α*<sub>1</sub>*σ̂*<sub>1</sub><sup>*z*</sup>*σ̂*<sub>2</sub><sup>*z*</sup> + *α*<sub>2</sub>*σ̂*<sub>1</sub><sup>*z*</sup>*σ̂*<sub>3</sub><sup>*z*</sup> + *α*<sub>1</sub>*σ̂*<sub>1</sub><sup>*z*</sup>*σ̂*<sub>4</sub><sup>*z*</sup>

*Ĥ*<sub>2</sub> = *α*<sub>1</sub>*σ̂*<sub>1</sub><sup>*z*</sup>*σ̂*<sub>2</sub><sup>*z*</sup> + *α*<sub>2</sub>*σ̂*<sub>2</sub><sup>*z*</sup>*σ̂*<sub>3</sub><sup>*z*</sup> + *α*<sub>3</sub>*σ̂*<sub>3</sub><sup>*z*</sup>*σ̂*<sub>4</sub><sup>*z*</sup>

Then, models exist in a *model space*, i.e. the space of all valid
combinations of the available terms. Any combination of terms is
permissible in a given model. QMLA can then be thought of as a search
through the model space for the set of terms which produce data that
best matches that of the system. Since these terms correspond to the
physical interactions affecting the system, the outcome can be thought
of as a complete characterisation.

# Model Training

Model traning is the process of optimising the parameters *α⃗* of a
given model against the system’s data. The model which is being learned
does not need to be the *true* model; any model can attempt to describe
any data. A A core hypothesis of QMLA is that models which better
reflect the true model will produce data more consistent with the system
data, when compared against less-physically-similar models.

In principle, any parameter-learning algorithm can fulfil the role of
training models in the QMLA framework, but in practice, [Quantum
Hamiltonian Learning (QHL)][] is used to perform Bayesian inference on
the parameterisation, and hence attempt to find the optimal
parameterisation for each model. This is performed using [QInfer][].

# Model Comparison

Two candidate models *Ĥ*<sub>1</sub>, *Ĥ*<sub>2</sub>, having undergone
model training, can be compared against each other to determine which
one better describes the system data. Bayes factors (BF) provide a
quantitative measure of the relative strength of the models at producing
the data. We take the BF *B*(*Ĥ*<sub>1</sub>, *Ĥ*<sub>2</sub>) between
two candidate models as evidence that one model is preferable. Evidence
is compiled in a series of pairwise comparisons; models are compared
with a number of competitors such that the strongest model from a pool
can be determined as that which won the highest number of pairwise
comparisons.

# Structure

QMLA is structured over several levels:

-   Models
    -   are individual candidates (e.g. Hamiltonians) which attempt to
        capture the physics of the `system`.
-   Layers/Branches:
    -   models are grouped in layers, which are thought of as branches
        on `exploration trees`.
-   Exploration trees
    -   are the objects on which the model search takes place: we think
        of models as *leaves* on *branches* of a tree. The model search
        is then the effort to find the single leaf on the tree which
        best describes the `system`. They grow and are pruned according
        to rules set out in the exploration strategy.
-   Exploration Strategies (ES)
    -   are bespoke sets of rules which decide how QMLA ought to proceed
        ateach step. For example, given the result of training/comparing
        a previous set of models, the ES determines the next set of
        candidate models to be considered.
-   Instance:
    -   a single implementation of the QMLA protocol, whether to run the
        entire model search or another subroutine the framework. Within
        an instance, several exploration trees can grow independently in
        parallel: we can then think of QMLA as a search for the single
        best leaf among a forest of trees, each of which corresponds to
        a unique exploration strategy.
-   Run
    -   many instances which pertain to the same problem. QMLA is run
        independently for a number of instances, allowing for analysis
        of the algorithm’s performance overall, e.g. that QMLA finds a
        particular model in 50% of 100 instances.

# Outputs

`QMLA` automatically performs a series of analyses and produces
associated plots. These are stored in a unique folder generated for the
`run` upon launch: this folder is specified by the date and time of the
launch and is located, relative to the `QMLA` main project directory in,
e.g., `launch/results/Jan_01/12_34`.

# User Interface

In order to tailor `QMLA` to a user’s needs, they must design a bespoke
`Exploration Strategy (ES)`. That is, the user must write a class
building upon and inheriting from the base ES, encompassing all of the
logic required to achieve their use case, for example by incorporating a
genetic algorithm within the method called upon for constructing new
candidates, `ExplorationStrategy.generate_models`. Then, that class must
be available to QMLA’s `get_exploration_class()` function, by ensuring
it is included in one of the `import` statements in
`qmla/exploration_strategies/__init__.py`. Finally, instruct `QMLA` to
use that `ES` for a run in the launch script. These steps are laid out
in full in the [QMLA Tutorial][].

[Quantum Hamiltonian Learning (QHL)]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.112.190501
[QInfer]: http://qinfer.org/
[QMLA Tutorial]: https://quantum-model-learning-agent.readthedocs.io/en/latest/tutorial.html
[full documentation is available]: https://quantum-model-learning-agent.readthedocs.io/en/latest/