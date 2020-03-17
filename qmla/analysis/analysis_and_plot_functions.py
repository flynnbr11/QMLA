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

import qmla.get_growth_rule as get_growth_rule
import qmla.model_naming as model_naming
import qmla.experimental_data_processing as expdt
# from qmla import experimental_data_processing check that is called as experimental_data_processing.method.
import qmla.expectation_values as expectation_values
import qmla.database_framework as database_framework

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



def ExpectationValuesTrueSim(
    qmd,
    model_ids=None, champ=True,
    times=None,
    max_time=3, t_interval=0.01,
    linspace_times=False,
    upper_x_lim=None,
    plus_probe=True,
    true_plot_type='scatter',
    save_to_file=None
):
    use_experimental_data = qmd.use_experimental_data
    experimental_measurements_dict = qmd.experimental_measurements

    try:
        qmd.champion_model_id
    except BaseException:
        qmd.champion_model_id = qmd.highest_model_id + 1

    if model_ids is None and champ == True:
        model_ids = [qmd.champion_model_id]
    elif model_ids is not None and champ == True:
        if type(model_ids) is not list:
            model_ids = [model_ids]
        if qmd.champion_model_id not in model_ids:
            model_ids.append(qmd.champion_model_id)

    if qmd.champion_model_id in model_ids and champ == False:
        model_ids.remove(qmd.champion_model_id)

    # plus_plus = np.array([0.5-0.j,  0.5-0.j, 0.5-0.j, 0.5-0.j]) # TODO
    # generalise probe
    plot_probe_dict = pickle.load(
        open(qmd.probes_plot_file, 'rb')
    )
    probe_id = random.choice(range(qmd.probe_number))
    # names colours from
    # https://matplotlib.org/2.0.0/examples/color/named_colors.html
    true_colour = colors.cnames['lightsalmon']  # 'b'
    true_colour = 'r'  # 'b'

    champion_colour = colors.cnames['cornflowerblue']  # 'r'
    sim_colours = ['g', 'c', 'm', 'y', 'k']
    global_min_time = 0
    plt.clf()
    if (
        (experimental_measurements_dict is not None)
        and
        (use_experimental_data == True)
    ):
        times = sorted(list(experimental_measurements_dict.keys()))
        true_expec_values = [
            experimental_measurements_dict[t] for t in times
        ]
    else:
        if times is None:
            times = np.arange(0, max_time, t_interval)
        else:
            times = times
        true = qmd.true_model_name
        true_op = database_framework.Operator(true)

        true_model_terms_params = qmd.true_param_list
        true_model_terms_matrices = qmd.true_model_constituent_operators
        true_dim = true_op.num_qubits

        # if plus_probe:
        #     # true_probe = plus_plus
        #     true_probe = expectation_values.n_qubit_plus_state(true_dim)
        # else:
        #     true_probe = qmd.probes_system[(probe_id,true_dim)]

        true_probe = plot_probe_dict[true_dim]

        time_ind_true_ham = np.tensordot(true_model_terms_params, true_model_terms_matrices, axes=1)
        true_expec_values = []

        # print("true ham:", time_ind_true_ham)
        for t in times:
            if qmd.use_time_dependent_true_model:
                # Multiply time dependent parameters by time value
                params = copy.copy(qmd.true_param_list)
                for i in range(
                    len(params) - qmd.num_time_dependent_true_params,
                    len(params)
                ):
                    params[i] = params[i] * t
                true_ham = np.tensordot(params, true_model_terms_matrices, axes=1)
            else:
                true_ham = time_ind_true_ham

            try:
                expec = qmd.growth_class.expectation_value(
                    ham=true_ham,
                    t=t,
                    state=true_probe
                )

            except UnboundLocalError:
                print("[PlotQMD]\n Unbound local error for:",
                      "\nParams:", params,
                      "\nTimes:", times,
                      "\ntrue_ham:", true_ham,
                      "\nt=", t,
                      "\nstate=", true_probe
                      )

            true_expec_values.append(expec)

    ax1 = plt.subplot(311)
    if true_plot_type == 'plot':
        #        plt.subplot(211)
        plt.plot(times,
                 true_expec_values,
                 label='True Expectation Value',
                 color=true_colour
                 )
    else:
        plt.scatter(times,
                    true_expec_values,
                    label='True Expectation Value',
                    marker='o', s=2, color=true_colour
                    )

        # If we want a linspace plot of expec value
        if linspace_times:
            max_exp_time = max(times)
            min_exp_time = min(times)
            num_times = len(times)
            times = list(
                np.linspace(min_exp_time, max_exp_time, num_times)
            )

        ChampionsByBranch = {
            v: k for k, v in qmd.branch_champions.items()
        }
        max_time_learned = 0
        for i in range(len(model_ids)):
            mod_id = model_ids[i]
            sim = qmd.model_name_id_map[mod_id]
            mod = qmd.get_model_storage_instance_by_id(mod_id)
            sim_ham = mod.learned_hamiltonian
            times_learned = mod.times_learned_over
            sim_dim = database_framework.get_num_qubits(mod.model_name)
            # if plus_probe:
            #     sim_probe = expectation_values.n_qubit_plus_state(sim_dim)
            # else:
            #     sim_probe = qmd.probes_system[(probe_id,sim_dim)]
            sim_probe = plot_probe_dict[sim_dim]
            colour_id = int(i % len(sim_colours))
            sim_col = sim_colours[colour_id]

            sim_expec_values = []
            present_expec_values_times = sorted(
                list(mod.expectation_values.keys())
            )
            for t in times:

                try:
                    sim_expec_values.append(
                        mod.expectation_values[t]
                    )
                except BaseException:
                    expec = qmd.growth_class.expectation_value(
                        ham=sim_ham,
                        t=t,
                        state=sim_probe
                    )

                    sim_expec_values.append(expec)

            if mod_id == qmd.champion_model_id:
                models_branch = ChampionsByBranch[mod_id]
                champ_sim_label = str(mod.model_name_latex + ' (Champion)')
                sim_label = champ_sim_label
                sim_col = champion_colour
                time_hist_label = 'Times learned by Champion Model'
            elif mod_id in list(qmd.branch_champions.values()):
                models_branch = ChampionsByBranch[mod_id]
                # sim_label = 'Branch '+str(models_branch)+' Champion'
                sim_label = mod.model_name_latex
            else:
                # sim_label = 'Model '+str(mod_id)
                sim_label = mod.model_name_latex

            plt.subplot(311)
            plt.plot(
                times,
                sim_expec_values,
                label=sim_label,
                color=sim_col
            )

            num_bins = len(set(times_learned))
            unique_times_learned = sorted(list(set(times_learned)))
            if max(unique_times_learned) > max_time_learned:
                max_time_learned = max(unique_times_learned)
            unique_times_count = []
            if min(unique_times_learned) < global_min_time:
                global_min_time = min(unique_times_learned)

            for u in unique_times_learned:
                unique_times_count.append(list(times_learned).count(u))
            r_squared = mod.r_squared()
            r_squared_of_t = mod.r_squared_of_t

            exp_times = sorted(list(mod.r_squared_of_t.keys()))
            r_sq_of_t = [mod.r_squared_of_t[t] for t in exp_times]
            bar_hist = 'hist'

            ax2 = plt.subplot(
                312,
                # sharex=ax1
            )
            if bar_hist == 'bar':
                plt.bar(
                    unique_times_learned,
                    unique_times_count,
                    color=sim_col,
                    label=str(mod.model_name_latex),
                    width=0.001
                )
            elif bar_hist == 'hist':
                plt.hist(
                    times_learned,
                    bins=num_bins,
                    label=str('Occurences (max time:' +
                              str(np.round(max_time_learned, 2)) +
                              ')'
                              ),
                    color=sim_col,
                    histtype='step',
                    fill=False
                )
            ax3 = ax2.twinx()
            ax3.set_ylabel('$R^2$')
            plt.plot(
                exp_times, r_sq_of_t,
                label='$R^2(t)$',
                color=sim_col,
                linestyle='dashed'
            )

        ax3.legend(
            bbox_to_anchor=(1.3, 0.5),
            loc=2
        )
        ax1.legend(
            title='Expectation Values',
            bbox_to_anchor=(1.0, 1.1),
            loc=2
        )
        ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True  # labels along the bottom edge are off
        )

        ax1.set_xlim(global_min_time, max(times))
        if max_time_learned > max(times):
            ax2.semilogx()
            ax2.axvline(
                max(times),
                color='red',
                label='Max exp. val. time shown'
            )

        max_time_plot = max(max(times), max_time_learned)
        ax2.set_xlim(global_min_time, max_time_plot + 0.1)

        ax1.set_ylabel('Exp Value')
        ax2.set_ylabel('Occurences')
        ax2.set_xlabel('Time (microseconds)')
        ax2.set_yscale('log')

        ax2.set_title(str('Times learned upon'))
        ax2.axvline(0, color='black')
        ax2.axhline(0, color='black')
        ax2.legend(
            bbox_to_anchor=(1.2, 1.1),
            loc=2
        )
        ax1.axvline(0, color='black')
        plot_title = str(
            str(qmd.num_particles) + ' particles.\n'
            + str(qmd.num_experiments) + ' experiments.'
        )
        plt.title(plot_title)
#        plt.figlegend()

        plt.tight_layout()
        if save_to_file is not None:
            plt.savefig(save_to_file, bbox_inches='tight')


def plot_learned_models_dynamics(
    qmd,
    model_ids=None,
    include_expec_vals=True,
    include_bayes_factors=True,
    include_times_learned=True,
    include_param_estimates=False,
    save_to_file=None
):
    if qmd.qhl_mode == True:
        model_ids = [qmd.true_model_id]
        include_bayes_factors = False
    elif qmd.qhl_mode_multiple_models == True:
        model_ids = list(qmd.qhl_mode_multiple_models_model_ids)
        include_bayes_factors = False
    elif model_ids is None:
        model_ids = list(qmd.branch_champions.values())

    model_ids = list(sorted(set(model_ids)))  # only uniques values
    true_expec_vals = pickle.load(
        open(qmd.qmla_controls.true_expec_path, 'rb'))
    times_to_plot = list(sorted(true_expec_vals.keys()))
    # times_to_plot = list(sorted(qmd.plot_times))

    # TODO this is overwritten within for loop below so that large
    # Hamiltonians don't have to work out each time step
    true_exp = [true_expec_vals[t] for t in times_to_plot]
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
        reduced = qmd.get_model_storage_instance_by_id(mod_id)
        reduced.compute_expectation_values(
            times=qmd.times_to_plot
        )
#         growth_generator = reduced.growth_rule_of_true_model
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
            # print(
            #     "[plot_learned_models_dynamics]",
            #     "\n\tModel ", reduced.model_name_latex,
            #     "\n\tnum qubits:", dim,
            #     "\n\tprobe:", probe

            # )
            expec_vals = {}
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
    #         ax.legend()

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
    #         ax.set_ylabel(
    #             desc,
    #             color=name_colour,
    #             rotation=0,
    #         )
            ax.yaxis.set_label_position("right")

            times_learned_over = reduced.times_learned_over
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
            terms = database_framework.get_constituent_names_from_name(name)
            num_terms = len(terms)

            term_positions = {}
            param_estimate_by_term = {}
            std_devs = {}

            for t in range(num_terms):
                term_positions[terms[t]] = t
                term = terms[t]
                param_position = term_positions[term]
                param_estimates = reduced.track_mean_params[:, param_position]
                #std_dev = mod.cov_matrix[param_position,param_position]
                # std_dev = reduced.track_covariance_matrices[
                #     :,param_position,param_position
                # ]
                std_dev = reduced.track_param_dist_widths[:, param_position]
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
                    if use_experimental_data == False:
                        y_true = qmd.true_param_dict[term]
                        # true_term_latex = database_framework.latex_name_ising(term)
                        true_term_latex = qmd.growth_class.latex_name(
                            name=term
                        )

                        ax.axhline(
                            y_true,
                            label=str(
                                true_term_latex +
                                ' True'),
                            color=colour)
                except BaseException:
                    pass
                y = np.array(param_estimate_by_term[term])
                s = np.array(std_devs[term])
                x = range(1, 1 + len(param_estimate_by_term[term]))
                latex_term = qmd.growth_class.latex_name(
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


def ExpectationValuesQHL_TrueModel(
    qmd,
    max_time=3, t_interval=0.01,
    upper_x_lim=None,
    true_plot_type='scatter',
    save_to_file=None,
    debug_print=False
):
    model_ids = [qmd.true_model_id]
#    probe_id = random.choice(range(qmd.probe_number))
    probe_id = 10

    experimental_measurements_dict = qmd.experimental_measurements
    use_experimental_data = qmd.use_experimental_data
    # names colours from
    # https://matplotlib.org/2.0.0/examples/color/named_colors.html
    true_colour = colors.cnames['lightsalmon']  # 'b'
    true_colour = 'r'  # 'b'

    champion_colour = colors.cnames['cornflowerblue']  # 'r'
    sim_colours = ['g', 'c', 'm', 'y', 'k']

    plt.clf()
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Expectation Value')

    if (experimental_measurements_dict is not None) and (
            use_experimental_data == True):
        times = sorted(list(experimental_measurements_dict.keys()))
        true_expec_values = [
            experimental_measurements_dict[t] for t in times
        ]
    else:
        times = np.arange(0, max_time, t_interval)
        true = qmd.true_model_name
        true_op = database_framework.Operator(true)

        true_model_terms_params = qmd.true_param_list
#        true_model_terms_matrices = true_op.constituents_operators
        true_model_terms_matrices = qmd.true_model_constituent_operators
        true_dim = true_op.num_qubits
        true_probe = qmd.probes_system[(probe_id, true_dim)]
        time_ind_true_ham = np.tensordot(true_model_terms_params, true_model_terms_matrices, axes=1)
        true_expec_values = []

        for t in times:
            if qmd.use_time_dependent_true_model:
                # Multiply time dependent parameters by time value
                params = copy.copy(qmd.true_param_list)
                for i in range(
                    len(params) - qmd.num_time_dependent_true_params,
                    len(params)
                ):
                    params[i] = params[i] * t
                true_ham = np.tensordot(params, true_model_terms_matrices, axes=1)
            else:
                true_ham = time_ind_true_ham

            try:
                expec = qmd.growth_class.expectation_value(
                    ham=true_ham,
                    t=t,
                    state=true_probe
                )

            except UnboundLocalError:
                print("[PlotQMD]\n Unbound local error for:",
                      "\nParams:", params,
                      "\nTimes:", times,
                      "\ntrue_ham:", true_ham,
                      "\nt=", t,
                      "\nstate=", true_probe
                      )

            true_expec_values.append(expec)

    if true_plot_type == 'plot':
        plt.plot(times, true_expec_values, label='True Expectation Value',
                 color=true_colour
                 )
    else:
        plt.scatter(times, true_expec_values, label='True Expectation Value',
                    marker='o', s=8, color=true_colour
                    )

    ChampionsByBranch = {v: k for k, v in qmd.branch_champions.items()}
    for i in range(len(model_ids)):
        mod_id = model_ids[i]
        sim = qmd.model_name_id_map[mod_id]
        sim_op = database_framework.Operator(sim)
        mod = qmd.get_model_storage_instance_by_id(mod_id)
        sim_params = list(mod.final_learned_params[:, 0])
        sim_ops = sim_op.constituents_operators
        sim_ham = np.tensordot(sim_params, sim_ops, axes=1)
        if debug_print:
            print("Times:\n", times)
            print("SIM HAM:\n", sim_ham)
        sim_dim = sim_op.num_qubits
        sim_probe = qmd.probes_system[(probe_id, sim_dim)]
        colour_id = int(i % len(sim_colours))
        sim_col = sim_colours[colour_id]


        sim_expec_values = []
        for t in times:            
            ex_val = qmd.growth_class.expectation_value(
                ham=sim_ham,
                t=t,
                state=sim_probe
            )
            sim_expec_values.append(ex_val)

        if mod_id == qmd.true_model_id:
            sim_label = 'Simulated Model'
            sim_col = champion_colour
        else:
            sim_label = 'Model ' + str(mod_id)

        plt.plot(
            times,
            sim_expec_values,
            label=sim_label,
            color=sim_col
        )

    ax = plt.subplot(111)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    qty = 0.1
    ax.set_position([box.x0, box.y0 + box.height * qty,
                     box.width, box.height * (1.0 - qty)])

    handles, labels = ax.get_legend_handles_labels()
    label_list = list(labels)
    handle_list = list(handles)

    new_labels = []
    new_handles = []

    special_labels = []
    special_handles = []

    special_terms = [
        'True Expectation Value',
        'Champion Model',
        'Simulated Model']

    for i in range(len(label_list)):
        if label_list[i] in special_terms:
            special_labels.append(label_list[i])
            special_handles.append(handle_list[i])
        else:
            new_labels.append(label_list[i])
            new_handles.append(handle_list[i])

    special_handles = tuple(special_handles)
    special_labels = tuple(special_labels)

    extra_lgd = True
    if len(new_handles) == 0:
        #        print("No models other than champ/true")
        extra_lgd = False

    new_handles = tuple(new_handles)
    new_labels = tuple(new_labels)

    all_expec_values = sim_expec_values + true_expec_values
    lower_y_lim = max(0, min(all_expec_values))
    upper_y_lim = max(all_expec_values)
    plt.ylim(lower_y_lim, upper_y_lim)

    if upper_x_lim is not None:
        plt.xlim(0, upper_x_lim)
    else:
        plt.xlim(0, max(times))
    plt.xlim(0, max_time)

    if extra_lgd:
        lgd_spec = ax.legend(special_handles, special_labels,
                             loc='upper center', bbox_to_anchor=(1, 1), fancybox=True,
                             shadow=True, ncol=1
                             )
        lgd_new = ax.legend(new_handles, new_labels,
                            loc='upper center', bbox_to_anchor=(1.15, 0.75),
                            fancybox=True, shadow=True, ncol=1
                            )
        plt.gca().add_artist(lgd_spec)
    else:
        lgd_spec = ax.legend(special_handles, special_labels,
                             loc='upper center', bbox_to_anchor=(1, 1), fancybox=True,
                             shadow=True, ncol=1
                             )

    latex_name_for_title = qmd.growth_class.latex_name(name=qmd.true_model_name)
    plt.title(
        str(
            "QHL test for " +
            latex_name_for_title
            + ". [" + str(qmd.num_particles) + " prt; "
            + str(qmd.num_experiments) + "exp]"
        )
    )
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


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

    # for i in steps_to_show:
    #     j = steps_to_show.index(i) - 1 # previous step which is shown on plot already
    #     if (
    #         steps_to_show.index(i)==0
    #         or
    #         not np.all(mod.particles[:,:,i] == mod.particles[:,:,j])
    #     ):
    #         # don't display identical distributions between steps
    #         print(
    #             "[plot_distribution_progression]",
    #             "i,j:", i,j
    #         )

    #         particles = mod.particles[:,:,i]
    #         particles = sorted(particles)
    #         colour = colours[i%len(colours)]

    #         # TODO if renormalise False, DON'T use a stat.pdf to model distribution
    #         if renormalise:
    #             fit = stats.norm.pdf(
    #                 particles,
    #                 np.mean(particles),
    #                 np.std(particles)
    #             )
    #             max_fit = max(fit)
    #             fit = fit/max_fit
    #         else:
    #             fit = mod.weights[:,i]

    #         if i==max_exp_num:
    #             colour = final_colour
    #             label = 'Final distribution'
    #             if show_means:
    #                 ax.axvline(
    #                     np.mean(particles),
    #                     label='Final Mean',
    #                     color=colour,
    #                     linestyle='dashed'
    #                 )
    #         elif i==min(steps_to_show):
    #             colour = initial_colour
    #             if show_means:
    #                 ax.axvline(
    #                     np.mean(particles),
    #                     label='Initial Mean',
    #                     color=colour,
    #                     linestyle='dashed'
    #                 )
    #             label='Initial distribution'
    #         else:
    #             label=str('Step '+str(i))

    #         ax.plot(particles, fit, label=label, color=colour)

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


def r_squared_plot(
    results_csv_path=None,
    save_to_file=None
):
    # For use in QHL parameter sweep
    # qhl_results = pd.DataFrame.from_csv(
    qhl_results = pd.read_csv(
        results_csv_path, index_col='ConfigLatex')

    piv = pd.pivot_table(qhl_results,
                         values=['Time', 'RSquaredTrueModel'],
                         index=['ConfigLatex'],
                         aggfunc={
                             'Time': [np.mean, np.median, min, max],
                             'RSquaredTrueModel': [np.median, np.mean]
                         }
                         )

    time_means = list(piv['Time']['mean'])
    time_mins = list(piv['Time']['min'])
    time_maxs = list(piv['Time']['max'])
    time_medians = list(piv['Time']['median'])

    r_squared_medians = list(piv['RSquaredTrueModel']['median'])
    r_squared_means = list(piv['RSquaredTrueModel']['mean'])
    num_models = len(time_medians)
    configs = piv.index.tolist()

    plt.clf()
    fig, ax = plt.subplots()
    if num_models <= 5:
        plot_height = num_models
    else:
        plot_height = num_models / 2

    fig.set_figheight(plot_height)
    # fig.set_figwidth(num_models/4)

    ax2 = ax.twiny()
    width = 0.5  # the width of the bars
    ind = np.arange(len(r_squared_medians))  # the x locations for the groups
    use_log_times = False
    time_colour = 'b'
    if use_log_times:
        times_to_use = [np.log10(t) for t in time_medians]
        ax2.set_xlabel('Time ($log_{10}$ seconds)')
    else:
        times_to_use = time_medians
        ax2.set_xlabel('Median Time (seconds)')

    ax2.barh(ind, times_to_use, width / 4, color=time_colour, label='Time')

    times_to_mark = [60, 600, 3600, 14400, 36000]
    if use_log_times:
        times_to_mark = [np.log10(t) for t in times_to_mark]

    max_time = max(times_to_use)
    for t in times_to_mark:
        if t < max_time:
            ax2.axvline(x=t, color=time_colour)

    ax.barh(ind, r_squared_medians, width, color='grey', align='center',
            label='$R^2$'
            )
    #    ax.axvline(x=max_x/2, color='g', label='50% Models correct')
    ax.set_yticks(ind)
    ax.set_yticklabels(configs, minor=False)
    ax.set_ylabel('Configurations')
    ax.set_xlim(min(r_squared_medians) - 0.2, 1)
    ax.axvline(0, label='$R^2=0$')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center',
               bbox_to_anchor=(0.5, -0.2), ncol=2
               )

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

    min_time = expdt.nearest_experimental_time_available(exp_times, 0)
    max_time = expdt.nearest_experimental_time_available(exp_times, max_time)
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

        mod_num_qubits = database_framework.get_num_qubits(mod.model_name)
        probe = expectation_values.n_qubit_plus_state(mod_num_qubits)
        epochs.extend([0, qmd.num_experiments - 1])
        if len(mod.epochs_after_resampling) > 0:
            epochs.extend(mod.epochs_after_resampling)

        epochs = sorted(set(epochs))
        for epoch in epochs:
            # Construct new Hamiltonian to get R^2 from
            # Hamiltonian corresponds to parameters at that epoch
            ham = np.tensordot(mod.track_mean_params[epoch], mod.model_terms_matrices, axes=1)
            sum_of_residuals = 0
            for t in exp_times:
                sim = qmd.growth_class.expectation - value(
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
            model_name = mod.growth_class.latex_name(
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


def BayF_IndexDictToMatrix(model_naming, AllBayesFactors,
                           StartBayesFactors=None):

    size = len(model_naming)
    Bayf_matrix = np.zeros([size, size])

    for i in range(size):
        for j in range(size):
            try:
                Bayf_matrix[i, j] = AllBayesFactors[i][j][-1]
            except BaseException:
                Bayf_matrix[i, j] = 1

    return Bayf_matrix


class SquareCollection(collections.RegularPolyCollection):
    """Return a collection of squares."""

    def __init__(self, **kwargs):
        super(
            SquareCollection,
            self).__init__(
            4,
            rotation=np.pi /
            4.,
            **kwargs)

    def get_transform(self):
        """Return transform scaling circle areas to data space."""
        ax = self.axes
        pts2pixels = 72.0 / ax.figure.dpi
        scale_x = pts2pixels * ax.bbox.width / ax.viewLim.width
        scale_y = pts2pixels * ax.bbox.height / ax.viewLim.height
        return transforms.Affine2D().scale(scale_x, scale_y)


class IndexLocator(ticker.Locator):

    def __init__(self, max_ticks=21):
        self.max_ticks = max_ticks

    def __call__(self):
        """Return the locations of the ticks."""
        dmin, dmax = self.axis.get_data_interval()
        if dmax < self.max_ticks:
            step = 1
        else:
            step = np.ceil(dmax / self.max_ticks)
        return self.raise_if_exceeds(np.arange(0, dmax, step))


def hinton(inarray, max_value=None, use_default_ticks=True,
           skip_diagonal=True, skip_which=None, grid=True, white_half=0.,
           where_labels='bottomleft'
           ):
    """Plot Hinton diagram for visualizing the values of a 2D array.

    Plot representation of an array with positive and negative values
    represented by white and black squares, respectively. The size of each
    square represents the magnitude of each value.

    AAG modified 04/2018

    Parameters
    ----------
    inarray : array
        Array to plot.
    max_value : float
        Any *absolute* value larger than `max_value` will be represented by a
        unit square.
    use_default_ticks: boolean
        Disable tick-generation and generate them outside this function.
    skip_diagonal: boolean
        remove plotting of values on the diagonal
    skip_which: None, upper, lower
        whether to plot both upper and lower triangular
        matrix or just one of them
    grid: Boolean
        to remove the grid from the plot
    white_half : float
        adjust the size of the white "coverage" of the "skip_which"
        part of the diagram
    where_labels: "bottomleft", "topright"
        move the xy labels and ticks to the corresponding position
    """

    ax = plt.gca()
    ax.set_facecolor('silver')
    # make sure we're working with a numpy array, not a numpy matrix
    inarray = np.asarray(inarray)
    height, width = inarray.shape
    if max_value is None:
        finite_inarray = inarray[np.where(inarray > -np.inf)]
        max_value = 2**np.ceil(np.log(np.max(np.abs(finite_inarray))) / np.log(2))
    values = np.clip(inarray / max_value, -1, 1)
    rows, cols = np.mgrid[:height, :width]

    pos = np.where(np.logical_and(values > 0, np.abs(values) < np.inf))
    neg = np.where(np.logical_and(values < 0, np.abs(values) < np.inf))

    # if skip_diagonal:
    # for mylist in [pos,neg]:
    # diags = np.array([ elem[0] == elem[1] for elem in mylist ])
    # diags = np.where(diags == True)
    # print(diags)
    # for elem in diags:
    # del(mylist[elem])
    # del(mylist[elem])

    for idx, color in zip([pos, neg], ['white', 'black']):
        if len(idx[0]) > 0:
            xy = list(zip(cols[idx], rows[idx]))

            circle_areas = np.pi / 2 * np.abs(values[idx])
            if skip_diagonal:
                diags = np.array([elem[0] == elem[1] for elem in xy])
                diags = np.where(diags == True)

                for delme in diags[0][::-1]:
                    circle_areas[delme] = 0

            if skip_which is not None:
                if skip_which is 'upper':
                    lows = np.array([elem[0] > elem[1] for elem in xy])
                if skip_which is 'lower':
                    lows = np.array([elem[0] < elem[1] for elem in xy])
                lows = np.where(lows == True)

                for delme in lows[0][::-1]:
                    circle_areas[delme] = 0

            squares = SquareCollection(sizes=circle_areas,
                                       offsets=xy, transOffset=ax.transData,
                                       facecolor=color, edgecolor=color)
            ax.add_collection(squares, autolim=True)

    if white_half > 0:
        for i in range(width):
            for j in range(i):

                xy = [(i, j)] if skip_which is 'upper' else [(j, i)]

                squares = SquareCollection(sizes=[white_half],
                                           offsets=xy, transOffset=ax.transData,
                                           facecolor='white', edgecolor='white')
                ax.add_collection(squares, autolim=True)

    ax.axis('scaled')
    # set data limits instead of using xlim, ylim.
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    if grid:
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    if use_default_ticks:
        ax.xaxis.set_major_locator(IndexLocator())
        ax.yaxis.set_major_locator(IndexLocator())

    if where_labels is 'topright':
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()


def format_fn(tick_val, tick_pos, labels):

    if int(tick_val) in range(len(labels)):
        return labels[int(tick_val)]
    else:
        return ''


class QMDFuncFormatter(Formatter):
    """
    Use a user-defined function for formatting.

    The function should take in two inputs (a tick value ``x`` and a
    position ``pos``), and return a string containing the corresponding
    tick label.
    """

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __call__(self, x, pos=None):
        """
        Return the value of the user defined function.

        `x` and `pos` are passed through as-is.
        """
        return self.func(x, pos, self.args)


def plotHinton(
    model_names,
    bayes_factors,
    growth_generator=None,
    save_to_file=None
):
    """
    Deprecated -- using old functionality e.g. get_latex_name 
    should come from growth class; 
    but retained in case plotting logic wanted later
    """

    hinton_mtx = BayF_IndexDictToMatrix(model_names, bayes_factors)
    log_hinton_mtx = np.log10(hinton_mtx)
    # labels = [database_framework.latex_name_ising(name) for name in model_names.values()]
    labels = [
        get_latex_name(name, growth_generator)
        for name in model_names.values()
    ]

    fig, ax = plt.subplots(figsize=(7, 7))

    hinton(
        log_hinton_mtx,
        use_default_ticks=True,
        skip_diagonal=True,
        where_labels='topright',
        skip_which='upper')
    ax.xaxis.set_major_formatter(QMDFuncFormatter(format_fn, labels))
    ax.yaxis.set_major_formatter(QMDFuncFormatter(format_fn, labels))
    plt.xticks(rotation=90)

    # savefigs(expdire, "EXP_CompareModels_BFhinton"+mytimestamp+".pdf")

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')
    plt.show()


###### Tree diagram #####



# static coloring property definitions
losing_node_colour = 'r'
branch_champ_node_colour = 'b'
overall_champ_node_colour = 'g'


def qmdclassTOnxobj(
    qmd,
    modlist=None,
    directed=True,
    only_adjacent_branches=True
):

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    positions = {}
    branch_x_filled = {}
    branch_mod_count = {}

    max_branch_id = qmd.branch_highest_id
    max_mod_id = qmd.highest_model_id
    if modlist is None:
        modlist = range(max_mod_id)
    for i in range(max_branch_id + 1):
        branch_x_filled[i] = 0
        branch_mod_count[i] = 0

    for i in modlist:
        mod = qmd.get_model_storage_instance_by_id(i)
        name = mod.model_name
        branch = qmd.get_model_data_by_field(name=name, field='branch_id')
        branch_mod_count[branch] += 1
        latex_term = mod.model_name_latex

        G.add_node(i)
        G.node[i]['label'] = latex_term
        G.node[i]['status'] = 0.2
        G.node[i]['info'] = 'Non-winner'

    # Set x-coordinate for each node based on how many nodes
    # are on that branch (y-coordinate)
    most_models_per_branch = max(branch_mod_count.values())
    for i in modlist:
        mod = qmd.get_model_storage_instance_by_id(i)
        name = mod.model_name
        branch = qmd.get_model_data_by_field(name=name, field='branch_id')
        num_models_this_branch = branch_mod_count[branch]
        pos_list = available_position_list(
            num_models_this_branch,
            most_models_per_branch
        )
        branch_filled_so_far = branch_x_filled[branch]
        branch_x_filled[branch] += 1

        x_pos = pos_list[branch_filled_so_far]
        y_pos = branch
        positions[i] = (x_pos, y_pos)
        G.node[i]['pos'] = (x_pos, y_pos)

    # set node colour based on whether that model won a branch
    for b in list(qmd.branch_champions.values()):
        if b in modlist:
            G.node[b]['status'] = 0.45
            G.node[b]['info'] = 'Branch Champion'

    G.node[qmd.champion_model_id]['status'] = 0.9
    G.node[qmd.champion_model_id]['info'] = 'Overall Champion'

    edges = []
    for a in modlist:
        for b in modlist:
            is_adj = adjacent_branch_test(qmd, a, b)
            if is_adj or not only_adjacent_branches:
                if a != b:
                    unique_pair = database_framework.unique_model_pair_identifier(a, b)
                    if ((unique_pair not in edges)
                        and (unique_pair in qmd.bayes_factor_pair_computed)
                        ):
                        edges.append(unique_pair)
                        vs = [int(stringa) for stringa
                              in unique_pair.split(',')
                              ]

                        thisweight = np.log10(
                            qmd.all_bayes_factors[float(vs[0])][float(vs[1])][-1]
                        )

                        if thisweight < 0:
                            # flip negative valued edges and move
                            # them to positive
                            thisweight = - thisweight
                            flipped = True
                            G.add_edge(vs[1], vs[0],
                                       weight=thisweight, flipped=flipped,
                                       winner=b,
                                       loser=a,
                                       adj=is_adj
                                       )
                        else:
                            flipped = False
                            G.add_edge(vs[0], vs[1],
                                       weight=thisweight, flipped=flipped,
                                       winner=a,
                                       loser=b,
                                       adj=is_adj
                                       )
    return G


def plot_qmla_single_instance_tree(
    qmd,
    save_to_file=None,
    only_adjacent_branches=True,
    id_labels=True,
    modlist=None
):

    G = qmdclassTOnxobj(
        qmd,
        only_adjacent_branches=only_adjacent_branches,
        modlist=modlist)

    arr = np.linspace(0, 50, 100).reshape((10, 10))
    cmap = plt.get_cmap('viridis')
    new_cmap = truncate_colormap(cmap, 0.35, 1.0)

    plotTreeDiagram(
        G,
        n_cmap=plt.cm.pink_r,
        e_cmap=new_cmap,
        arrow_size=0.02,
        # arrow_size = 8.0,
        nonadj_alpha=0.1, e_alphas=[],
        label_padding=0.4, pathstyle="curve",
        id_labels=id_labels, save_to_file=save_to_file)




### Parameter Estimate Plot ###
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
    use_experimental_data=False,
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
    terms = database_framework.get_constituent_names_from_name(name)
    num_terms = len(terms)

    term_positions = {}
    param_estimate_by_term = {}
    std_devs = {}

    for t in range(num_terms):
        term_positions[terms[t]] = t
        term = terms[t]
        param_position = term_positions[term]
        param_estimates = mod.track_mean_params[:, param_position]
        #std_dev = mod.cov_matrix[param_position,param_position]
        std_dev = mod.track_covariance_matrices[:, param_position, param_position]
        param_estimate_by_term[term] = param_estimates
        std_devs[term] = std_dev

    cm_subsection = np.linspace(0, 0.8, num_terms)
    colours = [cm.magma(x) for x in cm_subsection]
#    colours = [ cm.Set1(x) for x in cm_subsection ]

#    colours = ['b','r','g','orange', 'pink', 'grey']

    # TODO use color map as list
    # num_epochs = qmd.num_experiments
    num_epochs = mod.num_experiments
#    fig = plt.figure()
#    ax = plt.subplot(111)

    # ncols=3
    # nrows=3 # TODO  -- make safe
    ncols = int(np.ceil(np.sqrt(num_terms)))
    nrows = int(np.ceil(num_terms / ncols))

#    nrows=int(np.ceil( num_terms/ncols ))

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
            if use_experimental_data == False:
                y_true = qmd.true_param_dict[term]
                # true_term_latex = database_framework.latex_name_ising(term)
                true_term_latex = qmd.growth_class.latex_name(
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
        latex_term = mod.growth_class.latex_name(term)
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
    # plt.legend(bbox_to_anchor=(1.1, 1.05))
    # # TODO put title at top; Epoch centred bottom; Estimate centre y-axis
    # plt.title(str("Parameter estimation for model " +
    #     database_framework.latex_name_ising(name)+" ["+str(qmd.num_particles)
    #     +" prt;" + str(qmd.num_experiments) + "exp]"
    #     )
    # )

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


### Radar Plot ###

def plotRadar(qmd, modlist, save_to_file=None, plot_title=None):
    """
    Deprecated -- using old functionality e.g. UserFunctions; 
    but retained in case plotting logic wanted later
    """
  
    from matplotlib import cm as colmap

    labels = [
        get_latex_name(
            name=qmd.model_name_id_map[l],
            growth_generator=qmd.growth_rule_of_true_model
        ) for l in modlist
    ]
    size = len(modlist)
    theta = custom_radar_factory(size, frame='polygon')

    fig, ax = plt.subplots(
        figsize=(
            12, 6), subplot_kw=dict(
            projection='radar'))

#    cmap = colmap.get_cmap('viridis')
    cmap = colmap.get_cmap('RdYlBu')
    colors = [cmap(col) for col in np.linspace(0.1, 1, size)]

    required_bayes = {}
    scale = []

    for i in modlist:
        required_bayes[i] = {}
        for j in modlist:
            if i is not j:
                try:
                    val = qmd.all_bayes_factors[i][j][-1]
                    scale.append(np.log10(val))
                except BaseException:
                    val = 1.0
                required_bayes[i][j] = val

    [scale_min, scale_max] = [min(scale), max(scale)]
    many_circles = 4
    low_ini = scale_min
    shift_ini = 1
    shift = 6
    ax.set_rgrids(list(shift_ini +
                       np.linspace(low_ini + 0.05, 0.05, many_circles)),
                  labels=list(np.round(np.linspace(low_ini + 0.05, 0.05, many_circles),
                                       2)), angle=180
                  )

    for i in modlist:
        dplot = []
        for j in modlist:
            if i is not j:
                try:
                    bayes_factor = qmd.all_bayes_factors[i][j][-1]
                except BaseException:
                    bayes_factor = 1.0

                log_bayes_factor = np.log10(bayes_factor)
                dplot.append(shift + log_bayes_factor)
            else:
                dplot.append(shift + 0.0)
        ax.plot(theta, np.array(dplot), color=colors[int(i % len(colors))],
                linestyle='--', alpha=1.
                )
        ax.fill(theta, np.array(dplot),
                facecolor=colors[int(i % len(colors))], alpha=0.25
                )

    ax.plot(theta, np.repeat(shift, len(labels)), color='black',
            linestyle='-', label='BayesFactor=1'
            )

    ax.set_varlabels(labels, fontsize=15)
    try:
        ax.tick_params(pad=50)
    except BaseException:
        pass

    legend = ax.legend(labels, loc=(1.5, .35),
                       labelspacing=0.1, fontsize=14)

    if plot_title is not None:
        plt.title(str(plot_title))

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


class IndexLocator(ticker.Locator):

    def __init__(self, max_ticks=10):
        self.max_ticks = max_ticks

    def __call__(self):
        """Return the locations of the ticks."""
        dmin, dmax = self.axis.get_data_interval()
        if dmax < self.max_ticks:
            step = 1
        else:
            step = np.ceil(dmax / self.max_ticks)
        return self.raise_if_exceeds(np.arange(0, dmax, step))


def custom_radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=None, frac=1.0):
            self.set_thetagrids(np.degrees(theta), labels,
                                fontsize=fontsize, frac=frac
                                )

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
    return verts


#### Cumulative Bayes CSV and InterQMD Tree plotting ####


def updateAllBayesCSV(qmd, all_bayes_csv):
    import os
    import csv
    data = get_bayes_latex_dict(qmd)
    names = list(data.keys())
    fields = ['ModelName']
    fields += names

    all_models = []
    if os.path.isfile(all_bayes_csv) is False:
        # all_models += ['ModelName']
        # all_models += names
        # print("file exists:", os.path.isfile(all_bayes_csv))
        # print("creating CSV")
        with open(all_bayes_csv, 'a+') as bayes_csv:
            writer = csv.DictWriter(
                bayes_csv,
                fieldnames=fields
            )
            writer.writeheader()
    else:
        # print("file exists:", os.path.isfile(all_bayes_csv))
        current_csv = csv.DictReader(open(all_bayes_csv))
        current_fieldnames = current_csv.fieldnames
        new_models = list(
            set(fields) - set(current_fieldnames)
        )

        if len(new_models) > 0:

            import pandas
            csv_input = pandas.read_csv(
                all_bayes_csv,
                index_col='ModelName'
            )
            a = list(csv_input.keys())
            # print("pandas says existing models are:\n", a)
            empty_list = [np.NaN] * len(list(csv_input[a[0]].values))

            for new_col in new_models:
                csv_input[new_col] = empty_list
            # print("writing new pandas CSV: ", csv_input)
            csv_input.to_csv(all_bayes_csv)

    with open(all_bayes_csv) as bayes_csv:
        reader = csv.DictReader(
            bayes_csv,
        )
        fields = reader.fieldnames

    with open(all_bayes_csv, 'a') as bayes_csv:
        writer = csv.DictWriter(
            bayes_csv,
            fieldnames=fields,
        )
        for f in names:
            single_model_dict = data[f]
            single_model_dict['ModelName'] = f
            writer.writerow(single_model_dict)


def get_bayes_latex_dict(qmd):
    latex_dict = {}
    # print("get bayes latex dict")

    latex_write_file = open(
        # str(qmd.results_directory + 'LatexMapping.txt'),
        qmd.latex_name_map_file_path,
        'a+'
    )
    for i in list(qmd.all_bayes_factors.keys()):
        mod = qmd.model_name_id_map[i]
        latex_name = qmd.get_model_storage_instance_by_id(i).model_name_latex
        mapping = (mod, latex_name)
        print(mapping, file=latex_write_file)

    for i in list(qmd.all_bayes_factors.keys()):
        mod_a = qmd.get_model_storage_instance_by_id(i).model_name_latex
        latex_dict[mod_a] = {}
        for j in list(qmd.all_bayes_factors[i].keys()):
            mod_b = qmd.get_model_storage_instance_by_id(j).model_name_latex
            latex_dict[mod_a][mod_b] = qmd.all_bayes_factors[i][j][-1]
    return latex_dict



# Manipulate QMD output
def BayesFactorsCSV(qmd, save_to_file, names_ids='latex'):

    import csv
    fields = ['ID', 'Name']
    if names_ids == 'latex':
        # names = [database_framework.latex_name_ising(qmd.model_name_id_map[i]) for i in

        names = []
        for mod_name in list(qmd.model_name_id_map.values()):
            names.append(
                qmd.branch_growth_rule_instances[
                    qmd.models_branches[
                        qmd.model_id_to_name_map[mod_name]
                    ]
                ].latex_name(name=mod_name)
            )

    elif names_ids == 'nonlatex':
        names = [
            qmd.model_name_id_map[i]
            for i in
            range(qmd.highest_model_id)
        ]
    elif names_ids == 'ids':
        names = range(qmd.highest_model_id)
    else:
        print("BayesFactorsCSV names_ids must be latex, nonlatex, or ids.")

    fields.extend(names)

    with open(save_to_file, 'w') as csvfile:

        fieldnames = fields
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()
        for i in range(qmd.highest_model_id):
            model_bf = {}
            for j in qmd.all_bayes_factors[i].keys():
                if names_ids == 'latex':
                    other_model_name = qmd.branch_growth_rule_instances[
                        qmd.models_branches[j]
                    ].latex_name(name=qmd.model_name_id_map[j])

                elif names_ids == 'nonlatex':
                    other_model_name = qmd.model_name_id_map[j]
                elif names_ids == 'ids':
                    other_model_name = j
                model_bf[other_model_name] = qmd.all_bayes_factors[i][j][-1]

            # if names_ids=='latex':
                # model_bf['Name'] = database_framework.latex_name_ising(qmd.model_name_id_map[i])
            try:
                model_bf['Name'] = qmd.branch_growth_rule_instances[
                    qmd.models_branches[i]
                ].latex_name(name=qmd.model_name_id_map[i])
            except BaseException:
                model_bf['Name'] = qmd.model_name_id_map[i]
            model_bf['ID'] = i
            writer.writerow(model_bf)


# Overall multiple QMD analyses


def genetic_algorithm_f_score_fitness_plots(
    results_path, 
    save_directory=None
):
    combined_results = pd.read_csv(results_path)
    results_by_fscore = pd.DataFrame()

    for i in combined_results.index:
        ratings_list = eval(dict(combined_results['GrowthRuleStorageData'])[i])['f_score_fitnesses']

        for result in ratings_list: 
            results_by_fscore = (
                results_by_fscore.append(
                    pd.Series(
                    {
                        'f_score' : round_nearest(result[0], 0.05),
                        'fitness_by_win_ratio' : np.round(result[1], 2),
                        'fitness_by_rating' : np.round(result[2], 2),
                        'original_fitness' : int(result[3]),
                        # 'rating_to_wins_ratio' : np.round(result[4])
                    }), 
                    ignore_index=True
                )
            )    


    fig = plt.figure(
        figsize=(18, 10),
        # constrained_layout=True,
        tight_layout=True
    )
    gs = GridSpec(
        3,
        1,
    )
    sns.set_style('darkgrid')
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
    # sns.violinplot(
        x = 'f_score', 
        y = 'fitness_by_rating',
        data = results_by_fscore,
        ax = ax1,
    #     label = 'Rating'
    )
    ax1.legend()
    ax1.set_ylim(-0.1,1.1)
    ax1.set_title('Fitness by ELO rating')

    ax2 = fig.add_subplot(gs[1, 0])
    sns.boxplot(
    # sns.violinplot(
        x = 'f_score', 
        y = 'fitness_by_win_ratio',
        data = results_by_fscore,
        ax = ax2,
    #     label = 'Rating'
    )
    ax2.set_title('Fitness by win ratio')
    ax2.set_ylim(-0.1,1.1)

    ax3 = fig.add_subplot(gs[2, 0])
    sns.distplot(
        results_by_fscore['f_score'],
        bins = np.arange(0,1.01, 0.05),
        ax = ax3,
        kde=False, 
    )
    ax3.set_xlim(0,1)
    ax3.set_title('Number of models')


    
    if save_directory is not None:
        save_to_file = os.path.join(
            save_directory, 
            'genetic_alg_fitnesses_by_f_score.png'
        )
        plt.savefig(save_to_file)

    ### Separate plot
    plt.clf()
    fig = plt.figure(
        figsize=(18, 10),
        # constrained_layout=True,
        tight_layout=True
    )
    gs = GridSpec(
        3,
        1,
    )

    ax4 = fig.add_subplot(gs[0, 0])

    sns.lineplot(
        x = 'fitness_by_win_ratio',
        y = 'fitness_by_rating',
        data= results_by_fscore,
        ax = ax4,
    )

    highest_fit = max( 
        results_by_fscore['fitness_by_win_ratio'].max(), 
        results_by_fscore['fitness_by_rating'].max()
    )
    ax4.plot(
        np.linspace(0,highest_fit),
        np.linspace(0,highest_fit),
        label = 'x=y',
        ls = '--'
    )
    ax4.legend()
    ax4.set_title('Fitnes comparison')

    ax5 = fig.add_subplot(gs[1, 0])
    sns.boxplot(
        x='f_score',
        y='original_fitness', 
        data=results_by_fscore,
        ax = ax5,
    )
    sns.swarmplot(
        x='f_score',
        y='original_fitness', 
        data=results_by_fscore,
        ax = ax5,
    )
    ax5.set_title('Original ELO rating')
    # ax5.set_xlim(0,1)

    # ax6 = fig.add_subplot(gs[2, 0])
    # sns.boxplot(
    #     x='f_score',
    #     y='rating_to_wins_ratio', 
    #     data=results_by_fscore,
    #     ax = ax6,
    # )    
    # ax6.set_title('Ratio of fitness by rating:wins')
    # ax6.axhline(1, color='black', ls='dotted')
    # ax6.set_xlabel('F-score')
    # ax6.set_ylabel('Ratio')
    # # ax6.set_xlim(0,1)



    if save_directory is not None:
        save_to_file = os.path.join(
            save_directory, 
            'genetic_alg_fitness_comparison.png'
        )
        plt.savefig(save_to_file)

