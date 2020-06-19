# Quantum Model Learning Agent 

## Packages required:

### Python 
* numpy 1.16.2
* pandas 0.20.3
* scipy 1.0.0
* matplotlib 3.0.3
* QInfer 1.0 
* Cython
* Qutip 4.2.0
* pickle 4.0 (standard)
* random (standard)
* time (standard)
* copy (standard)
* argparse 1.1
* fermilib==0.1a3
* redis 2.10.6
* rq 0.10.6
* redis-py-cluster 1.3.4
* itertools (standard)
* psutil 5.4.3
* fpdf
* networkx 2.3
* ipyparallel 6.0.2
* scikit-learn 0.19.1

### Other
* redis


## Overview 
QMLA is a Python package for learning the model describing data obtained by probing a quantum system. 
There are three levels to QMLA: 
* model: individual models are trained through Bayesian inference (QHL).
* instance: QMLA protocol is implemented, involving learning and comparing models, and generating new candidate models.
* run: a group of instances are run, targeting the same underlying system. 

QMLA is designed to run locally or else on a cluster operating PBS. 
In both cases, shared memory is achieved using a redis-server. 
To run locally, initialise the redis-server in a terminal:
```
redis-server
```

In a separate terminal window, go to QMD/Launch, which contains scripts `local_launch.sh` and `parallel_launch.sh`. 
These implement QMLA locally or on a cluster respectively; they contain the top-level controls of interest:

* Run level
    * `num_tests`: how many independent QMLA instances to run

* Instance level
    * `growth_rule`: which QMLA growth rule to implement.
    * `qhl_test`: don't perform full QMLA tree, just run QHL on known/set true model. 
    * `multiple_qhl`: perform QHL on preset list of models
    * `do_further_qhl`: perform QHL refinement stage on preferred models after QMLA has finished.
    * `alt_growth_rules`: growth rules to concurrently implement in the same instance. 
        The main growth rule specifies the target system, 
        while alternative growth rules can consider different classes of models. 

* model learning parmaeters:
    * `prt`: number of particles used for model learning in the Bayesian inference framework.
    * `exp`: number of epochs/experiments to run for the Bayesian inference. 


The user need not change controls listed underneath, as these do not materially impact how QMLA proceeds 
(they are used for consistency e.g. for creating paths to store results).


## Growth rules
The main control the user must change for their purposes is which `growth_rule` is used. 
Growth rules are the mechanism by which the user can control how QMLA progresses: 
they specify the logic used to combine previously considered models, 
to produce unseen models to test against the system. 
They also contain key attributes for every stage of QMLA, which the user should be familiar with before 
constructing their own.


User growth rules can be written as sub-classes of the GrowthRule class. 
In order to write your own growth rule, users should familiarise themselves with 
the GrowthRule class to fully understand the available degrees of freedom.
User growth rules must inherit from it, and users should ensure that all the inherited methods
are suitable for their system and their intended learning procedure.

Briefly, some of the crucial aspects are
* how to generate probes used for parameter learning
* how to compute expectation values/likelihoods
* generating models from prior results
* selecting which experiment design heuristic to use
* selecting which prior distribution to use
* setting the true parameters of the target system
* setting parameters for 

### Programming convention
Note that Python files in this package follow Pep8. 
Before commiting new files, ensure to run autopep8 on the file first. 
```
autopep8 --in-place --aggressive <filename>
```
