import os
import sys 
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import seaborn as sns

import qmla.get_growth_rule

__all__ = [
    'generational_analysis',
    'r_sqaured_average',
    'average_quadratic_losses',
    'all_times_learned_histogram',
    'volume_average',
    'plot_bayes_factors_v_true_model'
]


def generational_analysis(combined_results, save_directory=None):
    if not os.path.exists(save_directory):
        try:
            os.makedirs(save_directory)
        except:
            pass

    generational_scores = None

    for k in combined_results.index:
        single_instance_gen_ll = eval(combined_results['GenerationalLogLikelihoods'][k])
        single_instance_gen_f_score= eval(combined_results['GenerationalFscore'][k])
        instance_id = combined_results['QID'][k]

        for gen in list(single_instance_gen_ll.keys()):
            this_gen_ll = single_instance_gen_ll[gen]
            # this_gen_log_abs_ll = [np.log(np.abs(ll)) for ll in this_gen_ll]
            this_gen_f_score = single_instance_gen_f_score[gen]
            # this_gen_data = list(zip(this_gen_ll, this_gen_log_abs_ll, this_gen_f_score))
            this_gen_data = list(zip(this_gen_ll, this_gen_f_score))

            df = pd.DataFrame(
                data = this_gen_data,
                columns = ['log_likelihood', 'f_score']
            )

            df['gen'] = gen
            # df['instance'] = k
            df['instance'] = instance_id
            if generational_scores is None:
                generational_scores = df
            else: 
                generational_scores = generational_scores.append(
                    df, 
                    ignore_index=True
                )

    num_instances = len(generational_scores.instance.unique())
    fig = plt.figure(
        figsize=(15, 8),
        tight_layout=True
    )
    gs = GridSpec(
        2,
        1,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    
    sns.boxplot(
        data = generational_scores, 
        x = 'gen',
        y = 'f_score',
        showfliers=False, 
        ax = ax1, 
    )
    # sns.swarmplot(
    #     data = generational_scores, 
    #     x = 'gen',
    #     y = 'f_score',
    #     # showfliers=False, 
    #     color='grey',
    #     ax = ax1, 
    # )

    ax1.set_ylabel('F score')
    ax1.set_xlabel('Generation')
    ax1.set_title('F score V Generation')
    ax1.set_ylim(0,1)
    ax1.axhline(0.5, ls='--',color='black')

    sns.pointplot(
        data = generational_scores, 
        x = 'gen',
        y = 'f_score',
#         showfliers=False, 
        hue='instance',
        ci=None, 
        ax = ax2, 
    )
    ax2.set_title("F score individual instances")
    ax2.axhline(0.5, ls='--', color='black')
    ax2.set_ylim(0,1)
    ax2.legend(
        title='Instance',
        ncol=min(8, num_instances)
    )

    if save_directory is not None:
        plt.savefig(
            os.path.join(
                save_directory, 
                "generational_measures_f_scores.png"
            )
        )
        
    plt.clf()
    gs = GridSpec(
        2,
        1,
    )
    ax3 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])

    sns.boxplot(
        data = generational_scores, 
        x = 'gen',
        y = 'log_likelihood',
        showfliers=False, 
        ax = ax3, 
    )

    ax3.set_title('Log likelihood V generation')
    ax3.set_ylabel('Log likelihood')
    ax3.set_xlabel('Generation')

    sns.pointplot(
        data = generational_scores, 
        x = 'gen',
        y = 'log_likelihood',
        # showfliers=False, 
        ci=None, 
        hue='instance',
        ax = ax4, 
    )

    ax4.set_title('Log likelihood individual instances')
    ax4.set_ylabel('Log likelihood')
    ax4.set_xlabel('Generation')
    ax4.legend(
        title='Instance',
        ncol=min(8, num_instances)
    )

    if save_directory is not None:
        plt.savefig(
            os.path.join(
                save_directory, 
                "generational_measures_log_likelihoods.png"
            )
        )


def r_sqaured_average(
    results_path,
    growth_class,
    growth_classes_by_name,
    top_number_models=2,
    save_to_file=None
):
    from matplotlib import cm
    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    # results = pd.DataFrame.from_csv(
    results = pd.read_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    def rank_models(n): return sorted(set(n), key=n.count)[::-1]
    # from
    # https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

    r_sq_by_model = {}

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0, 0.8, num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [cm.viridis(x) for x in cm_subsection]

    i = 0
    for i in range(len(names)):
        name = names[i]
        r_squared_values = list(
            results[results['NameAlphabetical'] == name]['RSquaredByEpoch']
        )

        r_squared_lists = {}
        num_wins = len(r_squared_values)
        for j in range(num_wins):
            rs = eval(r_squared_values[j])
            for t in list(rs.keys()):
                try:
                    r_squared_lists[t].append(rs[t])
                except BaseException:
                    r_squared_lists[t] = [rs[t]]

        times = sorted(list(r_squared_lists.keys()))
        means = np.array(
            [np.mean(r_squared_lists[t]) for t in times]
        )
        std_dev = np.array(
            [np.std(r_squared_lists[t]) for t in times]
        )

        gr_class = growth_classes_by_name[name]
        # TODO need growth rule of given name to get proper latex term
        term = gr_class.latex_name(name)
        # term = growth_class.latex_name(name) # TODO need growth rule of given
        # name to get proper latex term
        r_sq_by_model[term] = means
        plot_label = str(term + ' (' + str(num_wins) + ')')
        colour = colours[i]
        ax.plot(
            times,
            means,
            label=plot_label,
            marker='o'
        )
        ax.fill_between(
            times,
            means - std_dev,
            means + std_dev,
            alpha=0.2
        )
        ax.legend(
            bbox_to_anchor=(1.0, 0.9),
            title='Model (# instances)'
        )
    print("[AnalyseMultiple - r sq] r_sq_by_model:", r_sq_by_model)

    plt.xlabel('Epoch')
    plt.ylabel('$R^2$')
    plt.title('$R^2$ average')

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def average_quadratic_losses(
    results_path,
    growth_classes,
    growth_generator,
    top_number_models=2,
    fill_alpha=0.3,  # to shade area of 1 std deviation
    save_to_file=None
):
    from matplotlib import cm
    # results = pd.DataFrame.from_csv(
    results = pd.read_csv(
        results_path,
        index_col='QID'
    )
    sigmas = {  # standard sigma values
        1: 34.13,
        2: 13.59,
        3: 2.15,
        4: 0.1,
    }

    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    def rank_models(n): return sorted(set(n), key=n.count)[::-1]
    # from
    # https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))

    cm_subsection = np.linspace(
        0, 0.8, top_number_models
    )
    colour_list = [cm.Accent(x) for x in cm_subsection]

    plot_colours = {}
    for mod in winning_models:
        plot_colours[mod] = colour_list[winning_models.index(mod)]
        winning_models_quadratic_losses = {}

    fig = plt.figure()
    plt.clf()
    ax = plt.subplot(111)

    for mod in winning_models:
        winning_models_quadratic_losses[mod] = (
            results.loc[results['NameAlphabetical']
                        == mod]['QuadraticLosses'].values
        )

        list_this_models_q_losses = []
        for i in range(len(winning_models_quadratic_losses[mod])):
            list_this_models_q_losses.append(
                eval(winning_models_quadratic_losses[mod][i])
            )

        list_this_models_q_losses = np.array(
            list_this_models_q_losses
        )

        num_experiments = np.shape(list_this_models_q_losses)[1]
        avg_q_losses = np.empty(num_experiments)

        for i in range(num_experiments):
            avg_q_losses[i] = np.average(list_this_models_q_losses[:, i])

        latex_name = growth_classes[growth_generator].latex_name(name=mod)
        epochs = range(1, num_experiments + 1)

        ax.semilogy(
            epochs,
            avg_q_losses,
            label=latex_name,
            color=plot_colours[mod]
        )

        upper_one_sigma = [
            np.percentile(np.array(list_this_models_q_losses[:, t]), 50 + sigmas[1]) for t in range(num_experiments)
        ]
        lower_one_sigma = [
            np.percentile(np.array(list_this_models_q_losses[:, t]), 50 - sigmas[1]) for t in range(num_experiments)
        ]

        ax.fill_between(
            epochs,
            lower_one_sigma,
            upper_one_sigma,
            alpha=fill_alpha,
            facecolor=plot_colours[mod],
            #         label='$1 \sigma$ '
        )

    ax.set_xlim(1, num_experiments)
    ax.legend(bbox_to_anchor=(1, 1))
    plt.title('Quadratic Losses Averages')
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def all_times_learned_histogram(
    results_path="combined_results.csv",
    top_number_models=2,
    save_to_file=None
):
    from matplotlib import cm
    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    # results = pd.DataFrame.from_csv(
    results = pd.read_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    def rank_models(n): return sorted(set(n), key=n.count)[::-1]
    # from
    # https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0, 0.8, num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [cm.viridis(x) for x in cm_subsection]

    times_by_model = {}
    max_time = 0
    for i in range(len(names)):
        name = names[i]
        model_colour = colours[i]
        times_by_model[name] = []
        this_model_times_separate_runs = list(
            results[results['NameAlphabetical'] == name]['TrackTimesLearned']
        )

        num_wins = len(this_model_times_separate_runs)
        for j in range(num_wins):
            this_run_times = eval(this_model_times_separate_runs[j])
            times_by_model[name].extend(this_run_times)
            if max(this_run_times) > max_time:
                max_time = max(this_run_times)
        times_this_model = times_by_model[name]
        model_label = str(
            list(results[results['NameAlphabetical'] == name]['ChampLatex'])[0]
        )

        plt.hist(
            times_this_model,
            color=model_colour,
            # histtype='stepfilled',
            histtype='step',
            # histtype='bar',
            fill=False,
            label=model_label
        )

    # presuming all models used same heuristics .... TODO change if models can
    # use ones
    heuristic_type = list(
        results[results['NameAlphabetical'] == names[0]]['Heuristic'])[0]

    plt.legend()
    plt.title("Times learned on [{}]".format(heuristic_type))
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()
    plt.semilogy()
    if max_time > 100:
        plt.semilogx()
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def volume_average(
    results_path,
    growth_class,
    top_number_models=2,
    save_to_file=None
):
    from matplotlib import cm
    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    # results = pd.DataFrame.from_csv(
    results = pd.read_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    def rank_models(n): return sorted(set(n), key=n.count)[::-1]
    # from
    # https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0, 0.8, num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [cm.viridis(x) for x in cm_subsection]

    i = 0
    for i in range(len(names)):
        name = names[i]
        volume_values = list(
            results[results['NameAlphabetical'] == name]['TrackVolume']
        )

        volume_lists = {}
        num_wins = len(volume_values)
        for j in range(num_wins):
            rs = eval(volume_values[j])
            for t in list(rs.keys()):
                try:
                    volume_lists[t].append(rs[t])
                except BaseException:
                    volume_lists[t] = [rs[t]]

        times = sorted(list(volume_lists.keys()))
        means = np.array(
            [np.mean(volume_lists[t]) for t in times]
        )

        std_dev = np.array(
            [np.std(volume_lists[t]) for t in times]
        )

        term = growth_class.latex_name(name)
        plot_label = str(term + ' (' + str(num_wins) + ')')
        colour = colours[i]
        ax.plot(
            times,
            means,
            label=plot_label,
            marker='o',
            markevery=10
        )
        ax.fill_between(
            times,
            means - std_dev,
            means + std_dev,
            alpha=0.2
        )
        ax.legend(
            bbox_to_anchor=(1.0, 0.9),
            title='Model (# instances)'
        )
    plt.semilogy()
    plt.xlabel('Epoch')
    plt.ylabel('Volume')
    plt.title('Volume average')

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def plot_bayes_factors_v_true_model(
    results_csv_path,
    correct_mod="xTiPPyTiPPzTiPPxTxPPyTyPPzTz",
    growth_generator=None,
    save_to_file=None
):

    from matplotlib import cm
    # TODO saved fig is cut off on edges and don't have axes titles.

    growth_class = qmla.get_growth_rule.get_growth_generator_class(
        growth_generation_rule=growth_generator
    )

    correct_mod = growth_class.latex_name(
        name=correct_mod
    )
    results_csv = os.path.abspath(results_csv_path)
    # qmd_res = pd.DataFrame.from_csv(results_csv)
    qmd_res = pd.read_csv(results_csv)

    mods = list(
        set(list(
            qmd_res.index
        ))
    )
    if correct_mod not in mods:
        return False

    mods.pop(mods.index(correct_mod))
    othermods = mods
    correct_subDB = qmd_res.ix[correct_mod]
    all_BFs = []

    for competitor in othermods:
        BF_values = np.array((correct_subDB[competitor]))
        BF_values = BF_values[~np.isnan(BF_values)]

        all_BFs.append(BF_values)
    num_models = len(othermods)
    n_bins = 30
    # nrows=5
    # ncols=3
    ncols = int(np.ceil(np.sqrt(num_models)))
    nrows = int(np.ceil(num_models / ncols))

    fig, axes = plt.subplots(figsize=(20, 10), nrows=nrows, ncols=ncols)
    cm_subsection = np.linspace(0.1, 0.9, len(all_BFs))
    colors = [cm.viridis(x) for x in cm_subsection]

    for row in range(nrows):
        for col in range(ncols):
            # Make a multiple-histogram of data-sets with different length.
            idx = row * ncols + col
            if idx < len(all_BFs):
                hist, bins, _ = axes[row, col].hist(
                    np.log10(all_BFs[idx]), n_bins, color=colors[idx], label=othermods[idx])

                try:
                    maxBF = 1.1 * np.max(np.abs(np.log10(all_BFs[idx])))
                except BaseException:
                    maxBF = 10
                axes[row, col].legend()
                axes[row, col].set_xlim(-maxBF, maxBF)


#    fig.text(0.07, 0.5, 'Occurences', va='center', rotation='vertical')
#    fig.text(0.5, 0.07, '$log_{10}$ Bayes Factor', ha='center')

    plt.title("Bayes factors of true model.")
    if save_to_file is not None:
        print("Saving BF V true model to {}".format(save_to_file))
        fig.savefig(save_to_file, bbox_inches='tight')
    else: 
        print("BF V true model -- save to file is None")
