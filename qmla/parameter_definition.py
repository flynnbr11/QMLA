import numpy as np
import os
import pickle
import math

import scipy
import matplotlib.pyplot as plt

import qmla.construct_models
import qmla.shared_functionality.prior_distributions

pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    'set_shared_parameters'
]

def set_shared_parameters(
    growth_class,
    run_info_file=None,
    all_growth_rules=[],
    run_directory='',
    num_particles=100,
    probe_max_num_qubits_all_growth_rules=12,
    generate_evaluation_experiments=True,
):
    r"""
    Set up parameters for this `run` of QMLA.
    A run consists of any number of independent QMLA instances;
    for consistency they must share the same information.
    Parameters, such as true model (system) parameters
    and probes to use for plotting purposes,
    are shared by all QMLA instances within a given run.

    This function does not return anything, but stores data 
    required for the run to the ``run_info_file`` path.
    The data pickled are:

    :RunData true_model: 
        name of true model, i.e. the model we call the system, 
        against which candidate models are tested
    :RunData params_list: 
        list of parameters of the true model
    :RunData params_dict: 
        dict of parameters of the true model
    :RunData growth_generator: 
        growth rule (name) of true model
    :RunData all_growth_rules: 
        list of all growth rules (names) which are to 
        be performed by each instance
    :RunData evaluation_probes: 
        proebs to use during evaluation experiments
    :RunData evaluation_times: 
        times to use during evaluation experiments

    :param GrowthRule growth_class: growth rule of true model, from
        which to extract key info, e.g. true parameter ranges and prior.
    :param str run_info_file:
        path to which to store system information
    :param list all_growth_rules: 
        list of instances of :class:`~qmla.growth_rules.GrowthRule`
        which are the alternative growth rules, 
        i.e. which are performed during each instance, 
        but which do not specify the true model (system). 
    :param str run_directory: 
        path to which all results/information pertaining
        to this unique QMLA run are stored
    :param int num_paritlces: 
        number of particles used during model learning
    :param int probe_max_num_qubits_all_growth_rules: 
        largest system size for which to generate plot probes
    :param bool generate_evaluation_experiments:
        whether to construct an evaluation dataset which
        can be used to objectively evaluate models. 
        Evaluation data consists of experiments 
        (i.e. probes and evolution times) which were not 
        typically used in model learning, therefore each model 
        can be compared fairly on this data set. 

    """
    if growth_class.growth_generation_rule not in all_growth_rules:
        all_growth_rules.append(growth_class.growth_generation_rule)

    # Generate true parameters from an instance of the GR
    true_params_info = growth_class.generate_true_parameters()
    # Add stuff to run info
    true_params_info['all_growth_rules'] = all_growth_rules
    true_params_info['growth_generator'] = growth_class.growth_generation_rule

    if run_info_file is not None:
        import pickle
        pickle.dump(
            true_params_info,
            open(run_info_file, 'wb')
        )
    else:
        return true_params_info

    # Generate evaluation data set
    evaluation_data = growth_class.generate_evaluation_data(
        probe_maximum_number_qubits=probe_max_num_qubits_all_growth_rules,
        num_times = int(250), 
        run_directory = run_directory,
    )
    evaluation_data_path = os.path.join(
        run_directory, 
        'evaluation_data.p'
    )
    pickle.dump(
        evaluation_data, 
        open(evaluation_data_path, 'wb')
    )

    # Cosntruct true model
    true_params_dict = true_params_info['params_dict']
    true_ham = None
    for k in list(true_params_dict.keys()):
        param = true_params_dict[k]
        mtx = qmla.construct_models.compute(k)
        if true_ham is not None:
            true_ham += param * mtx
        else:
            true_ham = param * mtx

    try:
        qmla.utilities.plot_evaluation_dataset(
            evaluation_data = evaluation_data, 
            true_hamiltonian = true_ham,
            expectation_value_function = growth_class.expectation_value,
            save_to_file=os.path.join(
                run_directory, 
                'evaluation', 
                'dynamics.png'
            )
        )
    except:
        print("Failed to plot evaluation dataset.")


    
