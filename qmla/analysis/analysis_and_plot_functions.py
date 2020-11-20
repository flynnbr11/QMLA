from matplotlib.gridspec import GridSpec
from matplotlib import cm
import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import copy
import random

import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import ticker
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.ticker import Formatter
from matplotlib import colors as mcolors
import matplotlib.text as mpl_text
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap
from matplotlib.patches import FancyArrowPatch, Circle, ArrowStyle
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from inspect import currentframe, getframeinfo
import sklearn
import seaborn as sns

try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure
 

import qmla.get_exploration_strategy as get_exploration_strategy
import qmla.shared_functionality.experimental_data_processing
import qmla.shared_functionality.expectation_value_functions
import qmla.construct_models as construct_models

frameinfo = getframeinfo(currentframe())

# __all__ = [
#     'format_exponent', 
#     'flatten'
# ]

def flatten(l): return [item for sublist in l for item in sublist]

def format_exponent(n):
    a = '%E' % n
    val = a.split('E')[0].rstrip('0').rstrip('.')
    val = np.round(float(val), 2)
    exponent = a.split('E')[1]
    return str(val) + 'E' + exponent

def fill_between_sigmas(
    ax,
    distribution,
    times,
    legend=False,
    only_one_sigma=True,
):
    # to draw distributions on a given axis, ax.
    # where distribution must be a dict
    # distribution[t] = [...], a list of values for the distribution at that
    # time

    sigmas = {  # standard sigma values
        1: 34.13,
        2: 13.59,
        3: 2.15,
        4: 0.1,
    }

    upper_one_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 +
            sigmas[1]) for t in times]
    lower_one_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 -
            sigmas[1]) for t in times]
    upper_two_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 +
            sigmas[1] +
            sigmas[2]) for t in times]
    lower_two_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 -
            sigmas[1] -
            sigmas[2]) for t in times]
    upper_three_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 +
            sigmas[1] +
            sigmas[2] +
            sigmas[3]) for t in times]
    lower_three_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 -
            sigmas[1] -
            sigmas[2] -
            sigmas[3]) for t in times]
    upper_four_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 +
            sigmas[1] +
            sigmas[2] +
            sigmas[3] +
            sigmas[4]) for t in times]
    lower_four_sigma = [
        np.percentile(
            np.array(
                distribution[t]),
            50 -
            sigmas[1] -
            sigmas[2] -
            sigmas[3] -
            sigmas[4]) for t in times]

    fill_alpha = 0.3
    one_sigma_colour = 'blue'
    two_sigma_colour = 'green'
    three_sigma_colour = 'orange'
    four_sigma_colour = 'red'
    ax.fill_between(
        # times,
        [t + 1 for t in times],
        upper_one_sigma,
        lower_one_sigma,
        alpha=fill_alpha,
        facecolor=one_sigma_colour,
        label=r'$1 \sigma$ '
    )

    if only_one_sigma == False:
        ax.fill_between(
            # times,
            [t + 1 for t in times],
            upper_two_sigma,
            upper_one_sigma,
            alpha=fill_alpha/2, # less strong for smaller proportions
            facecolor=two_sigma_colour,
            label=r'$2 \sigma$ '
        )
        ax.fill_between(
            # times,
            [t + 1 for t in times],
            lower_one_sigma,
            lower_two_sigma,
            alpha=fill_alpha/2,
            facecolor=two_sigma_colour,
        )

        ax.fill_between(
            # times,
            [t + 1 for t in times],
            upper_three_sigma,
            upper_two_sigma,
            alpha=fill_alpha/2,
            facecolor=three_sigma_colour,
            label=r'$3 \sigma$ '
        )
        ax.fill_between(
            # times,
            [t + 1 for t in times],
            lower_two_sigma,
            lower_three_sigma,
            alpha=fill_alpha/2,
            facecolor=three_sigma_colour,
        )

        ax.fill_between(
            # times,
            [t + 1 for t in times],
            upper_four_sigma,
            upper_three_sigma,
            alpha=fill_alpha/2,
            facecolor=four_sigma_colour,
            label=r'$4 \sigma$ '
        )
        ax.fill_between(
            # times,
            [t + 1 for t in times],
            lower_three_sigma,
            lower_four_sigma,
            alpha=fill_alpha/2,
            facecolor=four_sigma_colour,
        )

    if legend == True:
        ax.legend(
            loc='center right',
            bbox_to_anchor=(1.5, 0.5),
            #             title=''
        )


def plot_parameter_estimates(
    qmd,
    model_id,
    save_to_file=None
):
    from matplotlib import cm
    mod = qmd.get_model_storage_instance_by_id(model_id)
    name = mod.model_name

    if name not in list(qmd.model_name_id_map.values()):
        print(
            "True model ", name,
            "not in studied models",
            list(qmd.model_name_id_map.values())
        )
        return False
    terms = construct_models.get_constituent_names_from_name(name)
    num_terms = len(terms)

    term_positions = {}
    param_estimate_by_term = {}
    std_devs = {}

    for t in range(num_terms):
        term_positions[terms[t]] = t
        term = terms[t]
        param_position = term_positions[term]
        param_estimates = mod.track_param_means[:, param_position]
        #std_dev = mod.cov_matrix[param_position,param_position]
        std_dev = mod.track_covariance_matrices[:, param_position, param_position]
        param_estimate_by_term[term] = param_estimates
        std_devs[term] = std_dev

    cm_subsection = np.linspace(0, 0.8, num_terms)
    colours = [cm.magma(x) for x in cm_subsection]
    # TODO use color map as list
    num_epochs = mod.num_experiments
    ncols = int(np.ceil(np.sqrt(num_terms)))
    nrows = int(np.ceil(num_terms / ncols))
    fig, axes = plt.subplots(
        figsize=(10, 7),
        nrows=nrows,
        ncols=ncols,
        squeeze=False
    )
    row = 0
    col = 0
    axes_so_far = 0
    i = 0
#    for term in list(param_estimate_by_term.keys()):
    for term in terms:
        ax = axes[row, col]
        colour = colours[i % len(colours)]
        i += 1
        try:
            y_true = qmd.true_param_dict[term]
            true_term_latex = qmd.exploration_class.latex_name(
                name=term
            )
            true_term_latex = true_term_latex[:-1] + '_{0}' + '$'

            ax.axhline(
                y_true,
                label=str(true_term_latex),
                color='red',
                linestyle='--'
            )
        except BaseException:
            pass
        y = np.array(param_estimate_by_term[term])
        s = np.array(std_devs[term])
        x = range(1, 1 + len(param_estimate_by_term[term]))
        latex_term = mod.exploration_class.latex_name(term)
        latex_term = latex_term[:-1] + r'^{\prime}' + '$'
        # print("[pQMD] latex_term:", latex_term)
        ax.scatter(
            x,
            y,
            s=max(1, 50 / num_epochs),
            label=str(latex_term),
            color=colour
        )
#        ax.set_yscale('symlog')
        # print("[pQMD] scatter done" )
        ax.fill_between(
            x,
            y + s,
            y - s,
            alpha=0.2,
            facecolor='green',
            # label='$\sigma$'

        )
        # print("[pQMD] fill between done")
        ax.legend(loc=1, fontsize=20)
        axes_so_far += 1
        col += 1
        if col == ncols:
            col = 0
            row += 1
        # ax.set_title(str(latex_term))
        # print("[pQMD] title set")

#    ax = plt.subplot(111)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Parameter Estimate', fontsize=15)

    if save_to_file is not None:
        print(
            "[plot_parameter_estimates] saving to file",
            save_to_file,
            "type:", type(save_to_file)
        )
        plt.savefig(
            save_to_file,
            bbox_inches='tight'
        )
    # print("[pQMD] complete")


def plot_learned_models_dynamics(
    qmd,
    model_ids=None,
    include_expec_vals=True,
    include_bayes_factors=True,
    include_times_learned=True,
    include_param_estimates=False,
    save_to_file=None
):

    model_ids = list(sorted(set(model_ids)))  # only uniques values
    true_expec_vals = pickle.load(
        open(qmd.qmla_controls.system_measurements_file, 'rb'))
    times_to_plot = list(sorted(true_expec_vals.keys()))

    # TODO this is overwritten within for loop below so that large
    # Hamiltonians don't have to work out each time step
    true_exp = [true_expec_vals[t] for t in times_to_plot]
    qmd.log_print([
        "[Dynamics plot] plot probe file:", qmd.qmla_controls.probes_plot_file,
        "\n true expectation value path:", qmd.qmla_controls.system_measurements_file 
    ])
    plot_probes = pickle.load(
        open(qmd.qmla_controls.probes_plot_file, 'rb')
    )
    num_models_to_plot = len(model_ids)
    all_bayes_factors = qmd.all_bayes_factors
    max_time = max(times_to_plot)
    individual_terms_already_in_legend = []

#     ncols = int(np.ceil(np.sqrt(num_models_to_plot)))
# nrows = 3*int(np.ceil(num_models_to_plot/ncols)) + 1 # 1 extra row for
# "master"

    ncols = (
        include_expec_vals +
        include_bayes_factors +
        include_times_learned +
        include_param_estimates
    )
#     ncols = 4
    nrows = num_models_to_plot

    fig = plt.figure(
        figsize=(18, 10),
        # constrained_layout=True,
        tight_layout=True
    )
    gs = GridSpec(
        nrows,
        ncols,
        # figure=fig # not available on matplotlib 2.1.1 (on BC)
    )

    row = 0
    col = 0

    for mod_id in model_ids:
        qmd.log_print([
            "Plotting dynamics for model {}".format(mod_id)
        ])
        reduced = qmd.get_model_storage_instance_by_id(mod_id)
        reduced.compute_expectation_values(
            times=qmd.times_to_plot
        )
#         exploration_rule = reduced.exploration_strategy_of_true_model
        desc = str(
            "ID:{}\n".format(mod_id) +
            reduced.model_name_latex
        )
        times_to_plot = list(sorted(true_expec_vals.keys()))
        plot_colour = 'blue'
        name_colour = 'black'
        dynamics_label = str(mod_id)
        try:
            true_model_id = qmd.true_model_id
        except BaseException:
            true_model_id = -1
        if (
            mod_id == qmd.champion_model_id
            and
            mod_id == true_model_id
        ):
            plot_colour = 'green'
            name_colour = 'green'
            dynamics_label += ' [true + champ]'
            desc += str('\n[True + Champ]')
        elif mod_id == qmd.champion_model_id:
            plot_colour = 'orange'
            name_colour = 'orange'
            dynamics_label += ' [champ]'
            desc += str('\n[Champ]')
        elif mod_id == true_model_id:
            plot_colour = 'green'
            name_colour = 'green'
            dynamics_label += ' [true]'
            desc += str('\n[True]')

        ############ --------------- ############
        ############ Plot dynamics in left most column ############
        ############ --------------- ############
        if include_expec_vals is True:
            ham = reduced.learned_hamiltonian
            dim = np.log2(np.shape(ham)[0])
            probe = plot_probes[reduced.probe_num_qubits]
            qmd.log_print(
                [
                "[plot_learned_models_dynamics]",
                "\n\tModel ", reduced.model_name_latex,
                "\n\tnum qubits:", dim,
                "\n\tprobe:", probe
                ]
            )
            # expec_vals = {}
            if dim > 4:
                times_to_plot = times_to_plot[0::5]

            times_to_plot = sorted(list(true_expec_vals.keys()))
            true_exp = [true_expec_vals[t] for t in times_to_plot]

            # choose an axis to plot on
            ax = fig.add_subplot(gs[row, col])
            # first plot true dynamics
            ax.plot(
                times_to_plot,
                true_exp,
                c='r'
            )

            # now plot learned dynamics
            expec_vals = reduced.expectation_values
            sim_times = sorted(list(expec_vals.keys()))
            sim_exp = [expec_vals[t] for t in sim_times]

            # print(
            #     "[plotDynamics]",
            #     "\nsim exp:", sim_exp,
            #     "\nsim_times:", sim_times
            # )
            ax.plot(
                sim_times,
                sim_exp,
                marker='o',
                markevery=10,
                c=plot_colour,
                # label = dynamics_label
                label=desc
            )
            # qmd.log_print([
            #     "[Dynamics plot]",
            #     "sim_exp:", sim_exp[0:20],
            #     "true exp:", true_exp[0:20]
            # ])
    #         ax.legend()
            ax.set_ylim(-0.05, 1.05)

            if row == 0:
                ax.set_title('Expectation Values')
            if ncols == 1:
                ax.legend()

            col += 1
            if col == ncols:
                col = 0
                row += 1
        ############ --------------- ############
        ############ Plot Bayes factors ############
        ############ --------------- ############
        if include_bayes_factors == True:
            bayes_factors_this_mod = []
            bf_opponents = []
            for b in model_ids:
                if b != mod_id:
                    if b in list(all_bayes_factors[mod_id].keys()):
                        # bf_opponents.append(
                        #     qmd.get_model_storage_instance_by_id(b).model_name_latex
                        # )
                        bayes_factors_this_mod.append(
                            np.log10(all_bayes_factors[mod_id][b][-1]))
                        bf_opponents.append(str(b))
            ax = fig.add_subplot(gs[row, col])
            ax.bar(
                bf_opponents,
                bayes_factors_this_mod,
                color=plot_colour
            )
            ax.axhline(0, color='black')
            if row == 0:
                ax.set_title('Bayes Factors [$log_{10}$]')

            col += 1
            if col == ncols:
                col = 0
                row += 1
        ############ --------------- ############
        ############ Plot times learned over ############
        ############ --------------- ############
        if include_times_learned == True:
            ax = fig.add_subplot(gs[row, col])
            if row == 0:
                ax.set_title('Times learned')
            ax.yaxis.set_label_position("right")

            times_learned_over = sorted(qmla.utilities.flatten(reduced.times_learned_over))
            qmd.log_print(["[single instance plot] Times for bin:", times_learned_over])
            n, bins, patches = ax.hist(
                times_learned_over,
                # histtype='step',
                color=plot_colour,
                # fill=False,
                label=desc
            )
            ax.legend()
            # ax.semilogy()
            for bin_value in bins:
                ax.axvline(
                    bin_value,
                    linestyle='--',
                    alpha=0.3
                )
            plot_time_max = max(times_to_plot)
            max_time = max(times_learned_over)
            if max_time > plot_time_max:
                ax.axvline(
                    plot_time_max,
                    color='red',
                    linestyle='--',
                    label='Dynamics plot cutoff'
                )
                ax.legend()
            ax.set_xlim(0, max_time)

            col += 1
            if col == ncols:
                col = 0
                row += 1

        ############ --------------- ############
        ############ Plot parameters estimates ############
        ############ --------------- ############
        if include_param_estimates == True:
            ax = fig.add_subplot(gs[row, col])
            name = reduced.model_name
            terms = construct_models.get_constituent_names_from_name(name)
            num_terms = len(terms)

            term_positions = {}
            param_estimate_by_term = {}
            std_devs = {}

            for t in range(num_terms):
                term_positions[terms[t]] = t
                term = terms[t]
                param_position = term_positions[term]
                param_estimates = reduced.track_param_means[:, param_position]
                #std_dev = mod.cov_matrix[param_position,param_position]
                # std_dev = reduced.track_covariance_matrices[
                #     :,param_position,param_position
                # ]
                std_dev = reduced.track_param_uncertainties[:, param_position]
                param_estimate_by_term[term] = param_estimates
                std_devs[term] = std_dev

            cm_subsection = np.linspace(0, 0.8, num_terms)
            colours = [cm.magma(x) for x in cm_subsection]
            # TODO use color map as list
            num_epochs = reduced.num_experiments

            i = 0
        #    for term in list(param_estimate_by_term.keys()):
            for term in terms:
                colour = colours[i % len(colours)]
                i += 1
                try:
                    y_true = qmd.true_param_dict[term]
                    true_term_latex = qmd.exploration_class.latex_name(
                        name=term
                    )

                    ax.axhline(
                        y_true,
                        ls='--',
                        label=str(
                            true_term_latex +
                            ' True'),
                        color=colour)
                except BaseException:
                    pass
                y = np.array(param_estimate_by_term[term])
                s = np.array(std_devs[term])
                x = range(1, 1 + len(param_estimate_by_term[term]))
                latex_term = qmd.exploration_class.latex_name(
                    name=term
                )
                if latex_term not in individual_terms_already_in_legend:
                    individual_terms_already_in_legend.append(latex_term)
                    plot_label = str(latex_term)
                else:
                    plot_label = ''
                # print("[pQMD] latex_term:", latex_term)
                ax.plot(
                    x,
                    y,
                    #                 s=max(1,50/num_epochs),
                    label=plot_label,
                    color=colour
                )
        #        ax.set_yscale('symlog')
                # print("[pQMD] scatter done" )
    #             ax.fill_between(
    #                 x,
    #                 y+s,
    #                 y-s,
    #                 alpha=0.2,
    #                 facecolor=colour
    #             )

                ax.legend()
            if row == 0:
                ax.set_title('Parameter Estimates')

            col += 1
            if col == ncols:
                col = 0
                row += 1
    if save_to_file is not None:
        plt.savefig(
            save_to_file,
            bbox_inches='tight'
        )

def plot_distribution_progression(
    qmd,
    model_id=None, true_model=False,
    num_steps_to_show=2, show_means=True,
    renormalise=True,
    save_to_file=None
):
    # Plots initial and final prior distribution over parameter space
    # with num_steps_to_show intermediate distributions
    # Note only safe/tested for QHL, ie on true model (single parameter).
    from scipy import stats
    plt.clf()
    if true_model:
        try:
            mod = qmd.get_model_storage_instance_by_id(qmd.true_model_id)
        except BaseException:
            print("True model not present in this instance of QMD.")
    elif model_id is not None:
        mod = qmd.get_model_storage_instance_by_id(model_id)
    else:
        print("Either provide a model id or set true_model=True to generate \
              plot of distribution development."
              )
    true_parameters = mod.true_model_params
    num_experiments = np.shape(mod.particles)[2]
    max_exp_num = num_experiments - 1
    num_intervals_to_show = num_steps_to_show
    increment = int(num_experiments / num_intervals_to_show)

    nearest_five = round(increment / 5) * 5
    if nearest_five == 0:
        nearest_five = 1

    steps_to_show = list(range(0, num_experiments, nearest_five))

    if max_exp_num not in steps_to_show:
        steps_to_show.append(max_exp_num)

    # TODO use a colourmap insted of manual list
    colours = ['gray', 'rosybrown', 'cadetblue']
    true_colour = 'k'
    initial_colour = 'b'
    final_colour = 'r'
    steps_to_show = sorted(steps_to_show)

    ax = plt.subplot(111)

    if show_means:
        for t in true_parameters:
            ax.axvline(
                t,
                label='True param',
                c=true_colour,
                linestyle='dashed')

    for i in steps_to_show:
        # previous step which is shown on plot already
        j = steps_to_show.index(i) - 1
        if not np.all(mod.particles[:, :, i] == mod.particles[:, :, j]):
            # don't display identical distributions between steps
            particles = mod.particles[:, :, i]
            particles = sorted(particles)
            colour = colours[i % len(colours)]

            # TODO if renormalise False, DON'T use a stat.pdf to model
            # distribution
            if renormalise:
                fit = stats.norm.pdf(
                    particles, np.mean(particles), np.std(particles))
                max_fit = max(fit)
                fit = fit / max_fit
            else:
                fit = mod.weights[:, i]

            if i == max_exp_num:
                colour = final_colour
                label = 'Final distribution'
                if show_means:
                    ax.axvline(np.mean(particles),
                               label='Final Mean', color=colour, linestyle='dashed'
                               )
            elif i == min(steps_to_show):
                colour = initial_colour
                if show_means:
                    ax.axvline(np.mean(particles), label='Initial Mean',
                               color=colour, linestyle='dashed'
                               )
                label = 'Initial distribution'
            else:
                label = str('Step ' + str(i))

            ax.plot(particles, fit, label=label, color=colour)

    plt.legend(bbox_to_anchor=(1.02, 1.02), ncol=1)
    plt.xlabel('Parameter estimate')
    plt.ylabel('Probability Density (relative)')
    title = str(
        'Probability density function of parameter for ' +
        mod.model_name_latex)
    plt.title(title)
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def plot_distribution_progression_of_model(
    mod,
    num_steps_to_show=2,
    show_means=True,
    renormalise=True,
    save_to_file=None
):
    # Plots initial and final prior distribution over parameter space
    # with num_steps_to_show intermediate distributions
    # Note only safe/tested for QHL, ie on true model (single parameter).

    from scipy import stats
    plt.clf()

    true_parameters = mod.true_model_params
    num_experiments = np.shape(mod.particles)[2]
    max_exp_num = num_experiments - 1
    num_intervals_to_show = num_steps_to_show
    increment = int(num_experiments / num_intervals_to_show)

    nearest_five = round(increment / 5) * 5
    if nearest_five == 0:
        nearest_five = 1

    # steps_to_show = list(range(0,num_experiments,nearest_five))

    resampled_epochs = mod.epochs_after_resampling
    steps_to_show = list(range(0, resampled_epochs, num_steps_to_show))
    steps_to_show = [int(s) for s in steps_to_show]

    if max_exp_num not in steps_to_show:
        steps_to_show.append(max_exp_num)

    # TODO use a colourmap insted of manual list
    colours = ['gray', 'rosybrown', 'cadetblue']
    true_colour = 'k'
    initial_colour = 'b'
    final_colour = 'r'
    steps_to_show = sorted(steps_to_show)
    print(
        "[plot_distribution_progression]",
        "num exp:", num_experiments,
        "increment:", increment,
        "steps to show", steps_to_show,
        "resampled epochs:", mod.epochs_after_resampling,
    )

    ax = plt.subplot(111)

    if show_means:
        for t in true_parameters:
            ax.axvline(
                t,
                label='True param',
                c=true_colour,
                linestyle='dashed'
            )

    for i in steps_to_show:
        print(
            "[plot_distribution_progression]",
            "i,", i
        )

        particles = mod.particles[:, :, i]
        particles = sorted(particles)
        colour = colours[i % len(colours)]

        # TODO if renormalise False, DON'T use a stat.pdf to model distribution
        if renormalise:
            fit = stats.norm.pdf(
                particles,
                np.mean(particles),
                np.std(particles)
            )
            max_fit = max(fit)
            fit = fit / max_fit
        else:
            fit = mod.weights[:, i]

        if i == max_exp_num:
            colour = final_colour
            label = 'Final distribution'
            if show_means:
                ax.axvline(
                    np.mean(particles),
                    label='Final Mean',
                    color=colour,
                    linestyle='dashed'
                )
        elif i == min(steps_to_show):
            colour = initial_colour
            if show_means:
                ax.axvline(
                    np.mean(particles),
                    label='Initial Mean',
                    color=colour,
                    linestyle='dashed'
                )
            label = 'Initial distribution'
        else:
            label = str('Step ' + str(i))

        ax.plot(particles, fit, label=label, color=colour)

    plt.legend(bbox_to_anchor=(1.02, 1.02), ncol=1)
    plt.xlabel('Parameter estimate')
    plt.ylabel('Probability Density (relative)')
    title = str(
        'Probability density function of parameter for ' +
        mod.model_name_latex
    )
    plt.title(title)
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def r_squared_from_epoch_list(
    qmd,
    model_ids=[],
    epochs=[],
    min_time=0,
    max_time=None,
    save_to_file=None,
):
    exp_times = sorted(list(qmd.experimental_measurements.keys()))
    if max_time is None:
        max_time = max(exp_times)

    min_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(exp_times, 0)
    max_time = qmla.shared_functionality.experimental_data_processing.nearest_experimental_time_available(exp_times, max_time)
    min_data_idx = exp_times.index(min_time)
    max_data_idx = exp_times.index(max_time)
    exp_times = exp_times[min_data_idx:max_data_idx]
    exp_data = [
        qmd.experimental_measurements[t] for t in exp_times
    ]
    # plus = 1/np.sqrt(2) * np.array([1,1])
    # probe = np.array([0.5, 0.5, 0.5, 0.5+0j]) # TODO generalise probe
    # picking probe based on model instead
    datamean = np.mean(exp_data[0:max_data_idx])
    datavar = np.sum((exp_data[0:max_data_idx] - datamean)**2)

    fig = plt.figure()
    ax = plt.subplot(111)
    model_ids = list(set(model_ids))
    for model_id in model_ids:
        mod = qmd.get_model_storage_instance_by_id(model_id)
        r_squared_by_epoch = {}

        mod_num_qubits = construct_models.get_num_qubits(mod.model_name)
        probe = qmla.shared_functionality.expectation_value_functionsn_qubit_plus_state(mod_num_qubits)
        epochs.extend([0, qmd.num_experiments - 1])
        if len(mod.epochs_after_resampling) > 0:
            epochs.extend(mod.epochs_after_resampling)

        epochs = sorted(set(epochs))
        for epoch in epochs:
            # Construct new Hamiltonian to get R^2 from
            # Hamiltonian corresponds to parameters at that epoch
            ham = np.tensordot(mod.track_param_means[epoch], mod.model_terms_matrices, axes=1)
            sum_of_residuals = 0
            for t in exp_times:
                sim = qmd.exploration_class.expectation - value(
                    ham=ham,
                    t=t,
                    state=probe
                )
                true = qmd.experimental_measurements[t]
                diff_squared = (sim - true)**2
                sum_of_residuals += diff_squared

            Rsq = 1 - sum_of_residuals / datavar
            r_squared_by_epoch[epoch] = Rsq

        r_squareds = [r_squared_by_epoch[e] for e in epochs]

        plot_label = str(mod.model_name_latex)
        ax.plot(epochs, r_squareds, label=plot_label, marker='o')
    ax.legend(bbox_to_anchor=(1, 0.5),)
    ax.set_ylabel('$R^2$')
    ax.set_xlabel('Epoch')
    ax.set_title('$R^2$ Vs Epoch (with resampling epochs)')
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def plot_quadratic_loss(
    qmd,
    champs_or_all='champs',
    save_to_file=None
):

    # plot Quad loss for single QMD instance
    plt.clf()
    ax = plt.subplot(111)

    if qmd.qhl_mode is True:
        to_plot_quad_loss = [qmd.true_model_id]
        plot_title = str('Quadratic Loss for True operator (from QHL)')
    elif champs_or_all == 'champs':
        to_plot_quad_loss = qmd.branch_champions.values()
        plot_title = str('Quadratic Loss for Branch champions')
    else:
        to_plot_quad_loss = qmd.model_name_id_map.keys()
        plot_title = str('Quadratic Loss for all models')

    for i in sorted(list(to_plot_quad_loss)):
        mod = qmd.get_model_storage_instance_by_id(i)
        if len(mod.quadratic_losses_record) > 0:
            epochs = range(1, len(mod.quadratic_losses_record) + 1)
            model_name = mod.exploration_class.latex_name(
                name=qmd.model_name_id_map[i]
            )
            ax.plot(epochs, mod.quadratic_losses_record, label=str(model_name))
    ax.legend(bbox_to_anchor=(1, 1))

    ax.set_title(plot_title)

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def plot_volume_after_qhl(
    qmd,
    model_id=None,
    true_model=True,
    show_resamplings=True,
    save_to_file=None
):
    if true_model:
        try:
            mod = qmd.get_model_storage_instance_by_id(
                qmd.true_model_id
            )
        except BaseException:
            print("True model not present in QMD models.")
    elif model_id is not None:
        mod = qmd.get_model_storage_instance_by_id(model_id)
    else:
        print("Must either provide model_id or set true_model=True for volume plot.")

    try:
        y = mod.volume_by_epoch
    except AttributeError:
        print("Model not considered.")
        raise

    x = range(qmd.num_experiments)

    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Volume')
    plt.semilogy(x, y, label='Volume')

    resamplings = mod.epochs_after_resampling

    if show_resamplings and len(resamplings) > 0:
        plt.axvline(resamplings[0], linestyle='dashed',
                    c='grey', label='Resample point'
                    )
        for r in resamplings[1:]:
            plt.axvline(r, linestyle='dashed', c='grey')

    plt.legend()
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')