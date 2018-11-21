Jupyter Notebook
QMD README
Last Checkpoint: 44 minutes ago
(unsaved changes)
Python 3
Python 3 
File
Edit
View
Insert
Cell
Kernel
Help


# QMD 
Quantum Model Development framework. 
​
​
## Instructions for adding a new growth generator
​
In order to add a growth rule to QMD, you must define how it works. 
​
Mostly this involves filling in various Dicts in UserFunctions, which direct wrapper functions to 
the appropriate functions, or values. 
​
​
### User Functions
​
Dictionaries to fill in:
* `expec_val_function_dict`: which function to call to get the expectation value. 
    This is called by `expectation_value_wrapper`, which takes arguments
    * `ham` Hamiltonian
    * `t` time
    * `state` state to evolve according to Hamiltonian and time
* `default_true_operators_by_generator`: Name of Hamiltonian to be used as the "true" Hamiltonian for the purpose of simulation. Must follow naming convention. 
* `fixed_axis_generators`: which growth generators have a fixed axis
    - e.g. a fixed magnetic field in the x-plane, relative to the basis inherently set by the name of the true Hamiltonian. If new generator does not have a fixed axis, do not fill in this dict. 
* `fixed_axes_by_generator`: which axis to fix. 
* `max_spawn_depth_info`: for growth rules which terminate after a fixed number of steps, here define the number of steps allowed. 
* `max_num_qubits_info`: maximum number of qubits permitted according to this growth generator. In some cases, growth is terminated when models reach this size. 
* `model_generation_functions`: which function creates a list of model names to form the next layer. These functions are stored in QML_lib/ModelGeneration. This is called by `new_model_generator` and must take arguments
    * `model_list`: champions of layer N, used to create list of models for layer N+1
    * `spawn_step`: doesn't have to be used, but can be useful to generate models specically based on here in the tree development the algorithm has reached. 
* `tree_finished_functions`: function to check whether the QMD tree should terminate at that step. Default is to terminate after a given number of steps. 
* `get_name_branch_map`: function which returns a dictionary mapping all generated models' names to a branch number. Default is to put models on a branch with ID equal to the number of parameters that model has. An alternative (prebuilt) is to assign branch ID of the number of qubits. These functions are stored in QML_lib/ModelNames
* `latex_naming_functions`: function to take a model name (string according to model naming convention), to a LaTex representation of that name. These can be specific to growth rules, and are stored in ModelNames.
* `initial_models`: default set of models to form the first layer, which are the seeds for growing the tree. 
#### Probe dict generators
QMD cycles through a set of probes at each epoch to learn parameters during the QHL phase for each model. 
These are generated once and stored in a dict. 
* `experimental_probe_dict_generator`: In the case experimental data is being used, the probes must correspond to the probes used during the experiment.
* `simulated_probe_dict_generator` and `special_probe_functions`: In simulated data, there may be special cases where probes are specific, e.g. |+>^n. In general, random probes are used for simulated data. 
​
​

​
