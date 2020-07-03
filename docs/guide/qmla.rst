Quantum Model Learning Agent
----------------------------

The class which controls everything is :class:`~qmla.QuantumModelLearningAgent`. 


An :term:`Instance` of this class is used to run one of the available algorithms; many 
independent instances can operate simultaneously and be analysed together (e.g. 
to see the *average* reproduction of dynamics following model learning). 
This is referred to as a :term:`Run`

The QMLA class provides methods for each of the available algorithms, as well 
as routines required therein, and methods for analysis and plotting. 

Models are assigned a unique `model_ID` upon generation. 
QMLA considers a set of models as a `layer` or a `branch`. 
Models can exist on multiple branches. 
For each growth rule included in the instance, a growth rule tree is built. 
On a given tree, the associated growth rule decides which models to 
consider. The first branch of the tree holds the initial models
:math:`\mu^1 = \{ M_1^1, \dots M_n^1\}` 
for that GR. 
After the initial models have been learned through :term:`QHL`, and compared on the 
first branch, the GR uses the available information (e.g. the number of pairwise 
wins each model has) to construct a new set of models, 
:math:`\mu^2 = \{ M_1^2, \dots M_n^2\}`. 
Subsequent branches 
:math:`\mu^i`
similarly construct models 
based on the information available to the GR so far. 

Each branch is resident on its associated GR tree, but also the branch is known
to **QMLA**. Branches are assigned unique IDs by QMLA, such that QMLA has a 
birds-eye view of all of the mdoels on all branches on all GR trees. 
**QMLA** calls on the GR (via the GR tree) for a set of models to place on
its next branch, completely indifferent to how those models are generated, 
or whether they have been learned already. 
This allows for completely self-contained logic in the GR: 
**QMLA** will simply learn and compare
the models it is presented - it is up to the GR to decide how to interpret them. 
As such the core **QMLA** class can be thought of as a simple loop: 
while the GR tree continues to return models, place those models on a branch, learn them 
and compare them. 
When all GR trees indicate they are finished, compare the champions of each tree against each other, 
to determine a global champion. 





