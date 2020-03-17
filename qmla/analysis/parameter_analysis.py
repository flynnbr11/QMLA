import numpy as np
import sys
import os

import pickle
import pandas as pd
import matplotlib.pyplot as plt

import qmla.database_framework as database_framework
plt.switch_backend('agg')

__all__ = [
    'average_parameters_across_instances'
]

def rank_models(n): 
    # from
    # https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency
    return sorted(set(n), key=n.count)[::-1]


def average_parameters_across_instances(
    results_path,
    file_to_store=None, 
    top_number_models=3,
    average_type='median'
):
    r"""
    Find the median and standard deviation of parameters within all champion models
    across instances in this results directory. 

    :param results_path: path where results are stored in CSV. 
    :param file_to_store: path which is used to store the resulting priors. 
    :param top_number_models: Number of models to compute averages for 
        (top by number of instance wins). 

    :returns learned_priors: priors (median + std dev) of parameters
        of champion models. Can be stored.
    """
    
    results = pd.read_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(
        results.loc[:, 'NameAlphabetical']
    )
    winning_models = list(set(all_winning_models))
    if len(all_winning_models) > top_number_models:
        # restrict to the top N models, where N is user input
        winning_models = rank_models(all_winning_models)[0:top_number_models]

    params_dict = {}
    sigmas_dict = {}
    for mod in winning_models:
        params_dict[mod] = {}
        sigmas_dict[mod] = {}
        params = database_framework.get_constituent_names_from_name(mod)
        for p in params:
            params_dict[mod][p] = []
            sigmas_dict[mod][p] = []

    for i in range(len(winning_models)):
        mod = winning_models[i]
        learned_parameters = list(
            results[
                results['NameAlphabetical'] == mod
            ]['LearnedParameters']
        )
        final_sigmas = list(
            results[
                results['NameAlphabetical'] == mod
            ]['FinalSigmas']
        )
        num_wins_for_mod = len(learned_parameters)
        for i in range(num_wins_for_mod):
            params = eval(learned_parameters[i])
            sigmas = eval(final_sigmas[i])
            for k in list(params.keys()):
                params_dict[mod][k].append(params[k])
                sigmas_dict[mod][k].append(sigmas[k])

    average_params_dict = {}
    avg_sigmas_dict = {}
    std_deviations = {}
    learned_priors = {}
    for mod in winning_models:
        average_params_dict[mod] = {}
        avg_sigmas_dict[mod] = {}
        std_deviations[mod] = {}
        learned_priors[mod] = {}
        params = database_framework.get_constituent_names_from_name(mod)
        for p in params:
            avg_sigmas_dict[mod][p] = np.median(sigmas_dict[mod][p])
            averaging_weight = [1 / sig for sig in sigmas_dict[mod][p]]
            average_params_dict[mod][p] = np.average(
                params_dict[mod][p],
                weights=sigmas_dict[mod][p]
            )
            learned_priors[mod][p] = [
                average_params_dict[mod][p],
                avg_sigmas_dict[mod][p]
            ]

    if file_to_store is not None:
        pickle.dump(
            learned_priors,
            open(file_to_store, 'wb'),
            protocol=4
        )

    return learned_priors

