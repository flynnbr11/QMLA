# Quantum Model Learning Agent instructions



## Packages required:

### Python 
TODO: add package installations used
* numpy 1.16.2
* pandas 0.20.3
* scipy 1.0.0
* matplotlib 3.0.3
* QInfer 1.0 
* Qutip 4.2.0
* pickle 4.0 (standard)
* random (standard)
* time (standard)
* copy (standard)
* argparse 1.1
* redis 2.10.6
* rq 0.10.6
* redis-py-cluster 1.3.4
* itertools (standard)
* psutil 5.4.3
* networkx 2.3
* ipyparallel 6.0.2
* scikit-learn 0.19.1

### Other
* redis


## Overview 
QMLA is a Python package for learning the model describing data obtained by probing a quantum system. 

QMLA is designed to run locally or else on a cluster operating SunGridEngine. In both cases, shared memory is achieved using a redis-server. 
Before running a QMLA instance, start the redis-server from the top directory of this package:
```
redis-server
```

In a separate terminal, go to QMD/ExperimentalSimulations, the file launch_qmd.sh contains the top-level controls of interest. 

* Binary settings - top level
** num_tests: how many independent QMLA instances to run
** qhl_test: don't perform full QMLA tree, just run QHL on known/set true model. 
** multiple_qhl: perform QHL on preset list of models
** do_further_qhl: perform QHL refinement stage on preferred models after QMLA has finished.
** exp_data: (1) use set of experimental data; (0) generate data according to simulated  
** simulate_experiment: use the growth rule set by experimental data but generate true data according to set model

* QHL parmaeters:
** prt: number of particles
** exp: number of epochs/experiments to run
** pgh: (deprecated) parmameters related to heuristic used for generating experimental times. 
** ra: resampler a parameter
** rt: resampler t parameter


The user need not change controls listed under QMD Settings as these do not materially impact how QMLA proceeds (they are used for consistency).


## Growth rules
The main control the user must change for their purposes is which growth rule is used. 
If 
* exp_data==1 => exp_growth_rule is assigned
* exp_data==0 => sim_growth_rule is assigned (pure simulation) 
* simulate_experimetn => exp_growth_rule is used, generating data according to the set true model
s