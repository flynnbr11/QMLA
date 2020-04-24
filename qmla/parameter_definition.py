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
    true_prior=None,
    pickle_file=None,
    random_vals=False,
    all_growth_rules=[],
    # exp_data=False,
    results_directory='',
    num_particles=100,
    probe_max_num_qubits_all_growth_rules = 12, 
    generate_evaluation_experiments=True, 
    true_prior_plot_file=None,
):
    r"""
    Set up parameters for this run of QMLA. 
    Parameters, such as true model parameters
    and probes to use for plotting purposes, 
    are shared by all instances of QMLA within a given run. 

    """
    true_model = growth_class.true_model
    if true_prior is None: 
        true_prior = growth_class.get_prior(
            model_name=true_model,
            log_file=growth_class.log_file,
            log_identifier='[Param definition]'
        )

    terms = database_framework.get_constituent_names_from_name(
        true_model
    )

    latex_terms = []
    for term in terms:
        lt = growth_class.latex_name(
            name=term
        )
        latex_terms.append(lt)
    true_model_latex = growth_class.latex_name(
        name=true_model,
    )

    num_terms = len(terms)
    true_model_terms_params = []
    true_params_dict = {}
    true_params_dict_latex_names = {}

    if growth_class.fixed_true_terms:
        # TODO move this to GR - generate if required; take from pickle if possible
        true_params_dict = growth_class.true_params_dict
        true_params_list = growth_class.true_params_list
    else:
        print("[param def] NOT using fixed params")
        # sample from wider distribution than initiated for QML
        widen_prior_factor = 2  # should mean true values within 3 sigma of learning distribution
        old_cov_mtx = true_prior.cov
        new_cov_mtx = widen_prior_factor * old_cov_mtx
        true_prior.__setattr__('cov', new_cov_mtx)
        sampled_list = true_prior.sample()
        try:
            fixed_true_params = growth_class.true_model_terms_params
        except BaseException:
            fixed_true_params = set_true_params

        for i in range(num_terms):
            if random_vals == True:
                print("[setQHL] using random vals")
                true_param = sampled_list[0][i]
            else:
                try:
                    term = terms[i]
                    true_param = fixed_true_params[term]
                except BaseException:
                    true_param = sampled_list[0][i]
            true_model_terms_params.append(true_param)
            true_params_dict[terms[i]] = true_param
            true_params_dict_latex_names[latex_terms[i]] = true_param

        true_prior.__setattr__('cov', old_cov_mtx)
        try:
            print("[ParameterDefinition] latex terms:", latex_terms)
            qmla.shared_functionality.prior_distributions.plot_prior(
                model_name=true_model_latex,
                model_name_individual_terms=latex_terms,
                prior=true_prior,
                plot_file=true_prior_plot_file,
                true_model_terms_params=true_params_dict_latex_names
            )
        except BaseException:
            print("[ParameterDefinition] plotting prior failed \n\n\n")
            pass
    
    print("[ParameterDefinition] parameters set")
    if growth_class.growth_generation_rule not in all_growth_rules: 
        all_growth_rules.append(growth_class.growth_generation_rule)

    if generate_evaluation_experiments:
        print("[ParameterDefinition] Setting evaluation stuff")
        evaluation_probes = growth_class.generate_probes(
            store_probes=False, 
            probe_maximum_number_qubits = probe_max_num_qubits_all_growth_rules, 
            # experimental_data = exp_data,
            noise_level = growth_class.probe_noise_level,
            minimum_tolerable_noise = 0.0,
        )
        print("evaluation Probes generated")

        num_evaluation_times = int(max(num_particles, 50)) # use at least 50 times to evaluate
        evaluation_times = scipy.stats.reciprocal.rvs(
            1e-1, 
            growth_class.max_time_to_consider, 
            size=num_evaluation_times
        ) # evaluation times generated log-uniformly
        available_probe_ids = list(range(growth_class.num_probes))
        print("evaluation times generated")
        list_len_fator = math.ceil(len(evaluation_times) / len(available_probe_ids))
        iterable_probe_ids = iter(available_probe_ids * list_len_fator)
        plt.clf()
        plt.hist(
            evaluation_times,
            bins = list(np.linspace(0,growth_class.max_time_to_consider, 10))
        )
        print("max time:", growth_class.max_time_to_consider)
        print("bins:", list(np.linspace(0,growth_class.max_time_to_consider, 10)))
        print("eval times:", evaluation_times)
        print("histogram plotted") 
        print("Results dir:", results_directory)
        plt.title('Times used for evaluation')
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        print("axes labels set") 
        fig_path = os.path.join(
            results_directory, 
            'times_for_evaluation.png'
        )
        print("Fig path:", fig_path)

        plt.savefig(fig_path)
        print("histogram plot saved") 
        
    else: 
        evaluation_probes = None
    print("[ParameterDefinition] evaluation stuff set ")
    true_params_info = {
        'params_list': true_model_terms_params,
        'params_dict': true_params_dict,
        'all_growth_rules': all_growth_rules,
        'evaluation_probes' : evaluation_probes,
        'evaluation_times' : evaluation_times
    }
    # print("True params info:", true_params_info)

    true_params_info['true_model'] = true_model
    true_params_info['growth_generator'] = growth_class.growth_generation_rule
    if pickle_file is not None:
        import pickle
        pickle.dump(
            true_params_info,
            open(pickle_file, 'wb')
        )
    else: 
        return true_params_info
