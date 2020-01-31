import numpy as np
import random
import pickle
import argparse
import os
import matplotlib.pyplot as plt

import qmla.DataBase as DataBase
import qmla.prior_distributions as Distributions
import qmla.get_growth_rule as get_growth_rule

pickle.HIGHEST_PROTOCOL = 2

def create_qhl_params(
    true_op,
    true_prior,
    pickle_file=None,
    random_vals=False,
    growth_generator=None,
    unique_growth_classes=None,
    all_growth_classes=None,
    rand_min=-100,
    rand_max=-50,
    exp_data=0,
    growth_class=None,
    plus_probe_for_plot=False,
    true_prior_plot_file=None,
):
    terms = DataBase.get_constituent_names_from_name(
        true_op
    )

    latex_terms = []
    for term in terms:
        lt = growth_class.latex_name(
            name=term
        )
        latex_terms.append(lt)
    true_op_latex = growth_class.latex_name(
        name=true_op,
    )

    num_terms = len(terms)
    true_params = []
    true_params_dict = {}
    true_params_dict_latex_names = {}

    # sample from wider distribution than initiated for QML
    widen_prior_factor = 2  # should mean true values within 3 sigma of learning distribution
    old_cov_mtx = true_prior.cov
    new_cov_mtx = widen_prior_factor * old_cov_mtx
    true_prior.__setattr__('cov', new_cov_mtx)
    sampled_list = true_prior.sample()
    try:
        fixed_true_params = growth_class.true_params
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
        true_params.append(true_param)
        true_params_dict[terms[i]] = true_param
        true_params_dict_latex_names[latex_terms[i]] = true_param

    true_prior.__setattr__('cov', old_cov_mtx)
    try:
        Distributions.plot_prior(
            model_name=true_op_latex,
            model_name_individual_terms=latex_terms,
            prior=true_prior,
            plot_file=true_prior_plot_file,
            true_params=true_params_dict_latex_names
        )
    except BaseException:
        print("[SetQHLParams] plotting prior failed \n\n\n")
        pass

    true_params_info = {
        'params_list': true_params,
        'params_dict': true_params_dict,
        'all_growth_classes': all_growth_classes,
    }
    if exp_data:
        print("\n\n\n[SetQHL] EXPDATA -- dont store true vals")
        # so as not to plot "true" params for exp data
        true_params_info['params_dict'] = None
        true_params_info['params_list'] = []
        print("true params info:\n", true_params_info)

    true_params_info['true_op'] = true_op
    true_params_info['growth_generator'] = growth_generator
    if pickle_file is not None:
        import pickle
        pickle.dump(
            true_params_info,
            open(pickle_file, 'wb')
        )
    else: 
        return true_params_info
