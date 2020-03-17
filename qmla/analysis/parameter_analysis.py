import numpy as np
import sys
import os

import pickle
import pandas as pd
import matplotlib.pyplot as plt

import qmla.database_framework as database_framework
from qmla.analysis.analysis_and_plot_functions import fill_between_sigmas
plt.switch_backend('agg')

__all__ = [
    'average_parameters_across_instances',
    'average_parameter_estimates'
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


def average_parameter_estimates(
    directory_name,
    results_path,
    results_file_name_start='results',
    growth_generator=None,
    unique_growth_classes=None,
    top_number_models=2,
    true_params_dict=None,
    save_to_file=None
):
    r"""
    Plots progression of parameter estimates against experiment number
    for the top models, i.e. those which win the most. 

    TODO: refactor this code - it should not need to unpickle
    all the files which have already been unpickled and stored in the summary
    results CSV.

    :param directory_name: path to directory where results .p files are stored.
    :param results_patha: path to CSV with all results for this run.
    :param growth_generator: the name of the growth generation rule used. 
    :param unique_growth_classes: dict with single instance of each growth rule class
        used in this run.
    :param top_number_models: Number of models to compute averages for 
        (top by number of instance wins). 
    :param true_params_dict: dict with true parameter for each parameter in the 
        true model.
    :param save_to_file: if not None, path to save PNG. 

    :returns None:
    """

    from matplotlib import cm
    plt.switch_backend('agg')  # to try fix plt issue on BC
    results = pd.read_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))

    os.chdir(directory_name)
    pickled_files = []
    for file in os.listdir(directory_name):
        if file.endswith(".p") and file.startswith(results_file_name_start):
            pickled_files.append(file)

    parameter_estimates_from_qmd = {}
    num_experiments_by_name = {}

    latex_terms = {}
    growth_rules = {}

    for f in pickled_files:
        fname = directory_name + '/' + str(f)
        result = pickle.load(open(fname, 'rb'))
        track_parameter_estimates = result['Trackplot_parameter_estimates']
    # for i in list(results.index):
    #     result = results.iloc(i)   
    #     track_parameter_estimates = eval(result['Trackplot_parameter_estimates'])

        alph = result['NameAlphabetical']
        if alph in parameter_estimates_from_qmd.keys():
            parameter_estimates_from_qmd[alph].append(
                track_parameter_estimates)
        else:
            parameter_estimates_from_qmd[alph] = [track_parameter_estimates]
            num_experiments_by_name[alph] = result['NumExperiments']

        if alph not in list(growth_rules.keys()):
            try:
                growth_rules[alph] = result['GrowthGenerator']
            except BaseException:
                growth_rules[alph] = growth_generator

    unique_growth_rules = list(set(list(growth_rules.values())))
    growth_classes = {}
    for g in list(growth_rules.keys()):
        try:
            growth_classes[g] = unique_growth_classes[growth_rules[g]]
        except BaseException:
            growth_classes[g] = None

    for name in winning_models:
        num_experiments = num_experiments_by_name[name]
        # epochs = range(1, 1+num_experiments)
        epochs = range(num_experiments_by_name[name] + 1)

        plt.clf()
        fig = plt.figure()
        ax = plt.subplot(111)

        parameters_for_this_name = parameter_estimates_from_qmd[name]
        num_wins_for_name = len(parameters_for_this_name)
        terms = sorted(database_framework.get_constituent_names_from_name(name))
        num_terms = len(terms)

        ncols = int(np.ceil(np.sqrt(num_terms)))
        nrows = int(np.ceil(num_terms / ncols))

        fig, axes = plt.subplots(
            figsize=(10, 7),
            nrows=nrows,
            ncols=ncols,
            squeeze=False,
        )
        row = 0
        col = 0
        axes_so_far = 0

        cm_subsection = np.linspace(0, 0.8, num_terms)
        colours = [cm.Paired(x) for x in cm_subsection]

        parameters = {}

        for t in terms:
            parameters[t] = {}

            for e in epochs:
                parameters[t][e] = []

        for i in range(len(parameters_for_this_name)):
            track_params = parameters_for_this_name[i]
            for t in terms:
                for e in epochs:
                    try:
                        parameters[t][e].append(track_params[t][e])
                    except:
                        parameters[t][e] = [track_params[t][e]]

        avg_parameters = {}
        std_devs = {}
        for p in terms:
            avg_parameters[p] = {}
            std_devs[p] = {}

            for e in epochs:
                avg_parameters[p][e] = np.median(parameters[p][e])
                std_devs[p][e] = np.std(parameters[p][e])

        for term in sorted(terms):
            ax = axes[row, col]
            axes_so_far += 1
            col += 1
            if (row == 0 and col == ncols):
                leg = True
            else:
                leg = False

            if col == ncols:
                col = 0
                row += 1
            # latex_terms[term] = database_framework.latex_name_ising(term)
            latex_terms[term] = growth_classes[name].latex_name(term)
            averages = np.array(
                [avg_parameters[term][e] for e in epochs]
            )
            standard_dev = np.array(
                [std_devs[term][e] for e in epochs]
            )

            param_lw = 3
            try:
                true_val = true_params_dict[term]
                true_term_latex = growth_classes[name].latex_name(term)
                ax.axhline(
                    true_val,
                    label=str('True value'),
                    ls='--',
                    color='red',
                    lw=param_lw

                )
            except BaseException:
                pass

            ax.axhline(
                0,
                linestyle='--',
                alpha=0.5,
                color='black',
                label='0'
            )
            fill_between_sigmas(
                ax,
                parameters[term],
                epochs,
                # colour = 'blue', 
                # alpha = 0.3,
                legend=leg,
                only_one_sigma=True, 
            )
            ax.plot(
                [e + 1 for e in epochs],
                averages,
                # marker='o',
                # markevery=0.1,
                # markersize=2*param_lw,
                lw=param_lw,
                label=latex_terms[term],
                color='blue'
            )

            # ax.scatter(
            #     [e + 1 for e in epochs],
            #     averages,
            #     s=max(1, 50 / num_experiments),
            #     label=latex_terms[term],
            #     color='black'
            # )


            latex_term = growth_classes[name].latex_name(term)
            ax.set_title(str(latex_term))

        latex_name = growth_classes[name].latex_name(name)

        if save_to_file is not None:
            fig.suptitle(
                'Parameter Esimates for {}'.format(latex_name)
            )
            try:
                save_file = ''
                if save_to_file[-4:] == '.png':
                    partial_name = save_to_file[:-4]
                    save_file = str(partial_name + '_' + name + '.png')
                else:
                    save_file = str(save_to_file + '_' + name + '.png')
                # plt.tight_layout()
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_file, bbox_inches='tight')
            except BaseException:
                print("Filename too long. Defaulting to idx")
                save_file = ''
                if save_to_file[-4:] == '.png':
                    partial_name = save_to_file[:-4]
                    save_file = str(
                        partial_name +
                        '_' +
                        str(winning_models.index(name)) +
                        '.png'
                    )
                else:
                    save_file = str(save_to_file + '_' + name + '.png')
                # plt.tight_layout()
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_file, bbox_inches='tight')

