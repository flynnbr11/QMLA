import numpy as np
import pickle

import qmla.database_framework as database_framework
import qmla.prior_distributions as distributions

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
    exp_data=0,
    true_prior_plot_file=None,
):
"""

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
        distributions.plot_prior(
            model_name=true_model_latex,
            model_name_individual_terms=latex_terms,
            prior=true_prior,
            plot_file=true_prior_plot_file,
            true_model_terms_params=true_params_dict_latex_names
        )
    except BaseException:
        print("[ParameterDefinition] plotting prior failed \n\n\n")
        pass

    if growth_class.growth_generation_rule not in all_growth_rules: 
        all_growth_rules.append(growth_class.growth_generation_rule)

    true_params_info = {
        'params_list': true_model_terms_params,
        'params_dict': true_params_dict,
        'all_growth_rules': all_growth_rules,
    }
    if exp_data:
        print("\n\n\n[SetQHL] EXPDATA -- dont store true vals")
        # so as not to plot "true" params for exp data
        true_params_info['params_dict'] = None
        true_params_info['params_list'] = []
        print("true params info:\n", true_params_info)

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
