import numpy as np
import os
import pickle
import math

import scipy
import matplotlib.pyplot as plt

import qmla.database_framework as database_framework
import qmla.shared_functionality.prior_distributions

pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    'set_shared_parameters'
]

def set_shared_parameters(
    growth_class,
    pickle_file=None,
    all_growth_rules=[],
    results_directory='',
    num_particles=100,
    probe_max_num_qubits_all_growth_rules = 12, 
    generate_evaluation_experiments=True, 
):
    r"""
    Set up parameters for this `run` of QMLA. 
    A run consists of any number of independent QMLA instances;
    for consistency they must share the same information.
    Parameters, such as true model parameters
    and probes to use for plotting purposes, 
    are shared by all instances of QMLA within a given run. 

    :param GrowthRule growth_class: growth rule of true model, from
        which to extract key info, e.g. true parameter ranges and prior.
    :param 
    """

    # Generate true model data.
    true_model = growth_class.true_model
    true_prior = growth_class.get_prior(
        model_name=true_model,
        log_file=growth_class.log_file,
        log_identifier='[Param definition]'
    )

    # Dissect true model into separate terms.
    terms = database_framework.get_constituent_names_from_name(
        true_model
    )
    latex_terms = [
        growth_class.latex_name(name=term) for term in terms
    ]
    true_model_latex = growth_class.latex_name(
        name=true_model,
    )
    num_terms = len(terms)
    
    # Generate and store true parameters.
    true_model_terms_params = []
    true_params_dict = {}
    true_params_dict_latex_names = {}

    if growth_class.fixed_true_terms:
        # TODO move this to GR - generate if required; take from pickle if possible
        true_params_dict = growth_class.true_params_dict
        true_params_list = growth_class.true_params_list
    else:
        # # sample from a distribution to get candidate true parameters
        # # use a wider distribution than initiated for QHL
        # # => true parameters within 3 sigma of learning distribution
        widen_prior_factor = 2  
        old_cov_mtx = true_prior.cov
        new_cov_mtx = widen_prior_factor * old_cov_mtx
        true_prior.__setattr__('cov', new_cov_mtx)
        sampled_list = true_prior.sample()

        # Either use randomly sampled parameter or that set in growth rule
        for i in range(num_terms):
            term = terms[i]
            
            try:
                # if this term is set in growth rule true_model_terms_params, use that value
                true_param = growth_class.true_model_terms_params[term]
            except:
                # otherwise, use value sampled from true prior
                true_param = sampled_list[0][i]
            
            true_model_terms_params.append(true_param)
            true_params_dict[terms[i]] = true_param
            true_params_dict_latex_names[latex_terms[i]] = true_param

        # Plot the true prior. 
        true_prior.__setattr__('cov', old_cov_mtx)
        try:
            true_prior_plot_file = os.path.join(results_directory, 'true_prior.png')
            qmla.shared_functionality.prior_distributions.plot_prior(
                model_name=true_model_latex,
                model_name_individual_terms=latex_terms,
                prior=true_prior,
                plot_file=true_prior_plot_file,
                true_model_terms_params=true_params_dict_latex_names
            )
        except BaseException:
            print("[ParameterDefinition] plotting prior failed.")
            pass
    
    print("[ParameterDefinition] Shared parameters set.")
    if growth_class.growth_generation_rule not in all_growth_rules: 
        all_growth_rules.append(growth_class.growth_generation_rule)

    evaluation_probes = None
    if generate_evaluation_experiments:
        # Generate test data to evaluate independent of training data.
        evaluation_probes = growth_class.generate_probes(
            store_probes=False, 
            probe_maximum_number_qubits = probe_max_num_qubits_all_growth_rules, 
            noise_level = growth_class.probe_noise_level,
            minimum_tolerable_noise = 0.0,
        )

        num_evaluation_times = int(max(num_particles, 50)) # use at least 50 times to evaluate
        evaluation_times = scipy.stats.reciprocal.rvs(
            growth_class.max_time_to_consider/100, 
            growth_class.max_time_to_consider, 
            size=num_evaluation_times
        ) # evaluation times generated log-uniformly
        available_probe_ids = list(range(growth_class.num_probes))
        list_len_fator = math.ceil(len(evaluation_times) / len(available_probe_ids))
        iterable_probe_ids = iter(available_probe_ids * list_len_fator)
        
        # Plot the times used for evaluation.
        plt.clf()
        plt.hist(
            evaluation_times,
            bins = list(np.linspace(0,growth_class.max_time_to_consider, 10))
        )
        plt.title('Times used for evaluation')
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        fig_path = os.path.join(
            results_directory, 
            'times_for_evaluation.png'
        )
        plt.savefig(fig_path)
        
    # Gather and store/return the true parameters.     
    true_params_info = {
        'params_list': true_model_terms_params,
        'params_dict': true_params_dict,
        'all_growth_rules': all_growth_rules,
        'evaluation_probes' : evaluation_probes,
        'evaluation_times' : evaluation_times,
        'true_model' : true_model, 
        'growth_generator' : growth_class.growth_generation_rule
    }
    if pickle_file is not None:
        import pickle
        pickle.dump(
            true_params_info,
            open(pickle_file, 'wb')
        )
    else: 
        return true_params_info
