Quantum Model Learning Agent
----------------------------

The class which controls everything is :class:`~qmla.QuantumModelLearningAgent`. 


An :term:`instance` of this class is used to run one of the available algorithms; many 
independent instances can operate simultaneously and be analysed together (e.g. 
to see the *average* reproduction of dynamics following model learning). 
This is referred to as a :term:`Run`

The QMLA class provides methods for each of the available algorithms, as well 
as routines required therein, and methods for analysis and plotting. 

Models are assigned a unique ``model_ID`` upon generation. 
QMLA considers a set of models as a `layer` or a `branch`. 
Models can exist on multiple branches. 
For each Exploration Strategy included in the instance, a Exploration Strategy tree is built. 
On a given tree, the associated Exploration Strategy decides which models to 
consider. The first branch of the tree holds the initial models
:math:`\mu^1 = \{ M_1^1, \dots M_n^1\}` 
for that ES. 
After the initial models have been learned through :term:`QHL`, and compared on the 
first branch, the ES uses the available information (e.g. the number of pairwise 
wins each model has) to construct a new set of models, 
:math:`\mu^2 = \{ M_1^2, \dots M_n^2\}`. 
Subsequent branches 
:math:`\mu^i`
similarly construct models 
based on the information available to the ES so far. 

Each branch is resident on its associated ES tree, but also the branch is known
to QMLA. Branches are assigned unique IDs by QMLA, such that QMLA has a 
birds-eye view of all of the mdoels on all branches on all ES trees. 
QMLA calls on the ES (via the ES tree) for a set of models to place on
its next branch, completely indifferent to how those models are generated, 
or whether they have been learned already. 
This allows for completely self-contained logic in the ES: 
QMLA will simply learn and compare
the models it is presented - it is up to the ES to decide how to interpret them. 
As such the core QMLA algorithm can be thought of as a simple loop: 
while the ES tree continues to return models, place those models on a branch, learn them 
and compare them. 
When all ES trees indicate they are finished, compare the champions of each tree against each other, 
to determine a global champion. 





