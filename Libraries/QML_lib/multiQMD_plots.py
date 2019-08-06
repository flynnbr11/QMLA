
import sys, os
import pickle
import matplotlib.pyplot as plt
import pandas
plt.switch_backend('agg')
from matplotlib.lines import Line2D

import argparse
import numpy as np

import DataBase
import PlotQMD as ptq
import ModelNames
# import UserFunctions 
import GrowthRules



def parameter_sweep_analysis(
    directory_name, 
    results_csv, 
    save_to_file=None, 
    use_log_times=False, 
    use_percentage_models=False
):

    import os, csv
    if not directory_name.endswith('/'):
        directory_name += '/'

    qmd_cumulative_results = pandas.DataFrame.from_csv(results_csv,
        index_col='ConfigLatex'
    )
    piv = pandas.pivot_table(
        qmd_cumulative_results, 
        values=['CorrectModel', 'Time', 'Overfit', 'Underfit', 'Misfit'], 
        index=['ConfigLatex'], 
        aggfunc={
            'Time':[np.mean, np.median, min, max], 
            'CorrectModel' : [np.sum, np.mean],
            'Overfit' : [np.sum, np.mean],
            'Misfit' : [np.sum, np.mean],
            'Underfit' : [np.sum, np.mean] 
        }
    )

    time_means = list(piv['Time']['mean'])
    time_mins = list(piv['Time']['min'])
    time_maxs = list(piv['Time']['max'])
    time_medians = list(piv['Time']['median'])
    correct_count = list(piv['CorrectModel']['sum'])
    correct_ratio = list(piv['CorrectModel']['mean'])
    overfit_count = list(piv['Overfit']['sum'])
    overfit_ratio = list(piv['Overfit']['mean'])
    underfit_count = list(piv['Underfit']['sum'])
    underfit_ratio = list(piv['Underfit']['mean'])
    misfit_count = list(piv['Misfit']['sum'])
    misfit_ratio = list(piv['Misfit']['mean'])
    num_models = len(time_medians)

    configs = piv.index.tolist()
    percentages = [a*100 for a in correct_ratio]


    plt.clf()
    fig, ax = plt.subplots()
    if num_models <= 5 :
        plot_height = num_models
    else:
        plot_height = num_models/2
    
    fig.set_figheight(plot_height)
    #fig.set_figwidth(num_models/4)

    ax2 = ax.twiny()
    width = 0.5 # the width of the bars 
    ind = np.arange(len(correct_ratio))  # the x locations for the groups

    if use_log_times:
        times_to_use = [np.log10(t) for t in time_medians]
        ax2.set_xlabel('Time ($log_{10}$ seconds)')
    else:
        times_to_use = time_medians
        ax2.set_xlabel('Median Time (seconds)')

    if use_percentage_models:
        correct = [a*100 for a in correct_ratio]
        misfit = [a*100 for a in misfit_ratio]
        underfit = [a*100 for a in underfit_ratio]
        overfit = [a*100 for a in overfit_ratio]
        ax.set_xlabel('% Models')
    else:
        correct = correct_count
        misfit = misfit_count
        overfit = overfit_count
        underfit = underfit_count
        ax.set_xlabel('Number of Models')

    max_x = correct[0] + misfit[0] + overfit[0] + underfit[0]
    time_colour = 'b'
    ax2.barh(ind, times_to_use, width/4, color=time_colour, label='Time')
    
    times_to_mark = [60,600, 3600, 14400, 36000]
    if use_log_times:
        times_to_mark = [np.log10(t) for t in times_to_mark]

    max_time = max(times_to_use)
    for t in times_to_mark:
        if t < max_time:
            ax2.axvline(x=t, color=time_colour)


    left_pts = [0] * num_models
    ax.barh(ind, correct, width, color='g', align='center', 
        label='Correct Models', left=left_pts
    )
    left_pts = [sum(x) for x in zip(left_pts, correct)]

    ax.barh(ind, underfit, width, color='r', align='center', 
        label='Underfit Models', left=left_pts
    )
    left_pts = [sum(x) for x in zip(left_pts, underfit)]
    
    ax.barh(ind, misfit, width, color='orange', align='center', 
        label='Misfit Models', left=left_pts
    )
    left_pts = [sum(x) for x in zip(left_pts, misfit)]

    ax.barh(ind, overfit, width, color='y', align='center', 
        label='Overfit Models', left=left_pts
    )
    left_pts = [sum(x) for x in zip(left_pts, overfit)]

#    ax.axvline(x=max_x/2, color='g', label='50% Models correct')   
    ax.set_yticks(ind)
    ax.set_yticklabels(configs, minor=False)
    ax.set_ylabel('Configurations')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center',
        bbox_to_anchor=(0.5, -0.2), ncol=2
    )

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')
        
        
def average_parameters(
    results_path, 
    top_number_models=3,
    average_type='median'
):

    results = pandas.DataFrame.from_csv(
        results_path,
        index_col='QID'
    )

    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    rank_models = lambda n:sorted(set(n), key=n.count)[::-1] 
    # from https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency
    
    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))    


    params_dict = {}
    sigmas_dict = {}
    for mod in winning_models:
        params_dict[mod] = {}
        sigmas_dict[mod] = {}
        params = DataBase.get_constituent_names_from_name(mod)
        for p in params:
            params_dict[mod][p] = []
            sigmas_dict[mod][p] = []

    for i in range(len(winning_models)):
        mod = winning_models[i]
        learned_parameters = list(
            results[ results['NameAlphabetical']==mod ]
            ['LearnedParameters']
        )
        final_sigmas = list(
            results[ results['NameAlphabetical']==mod ]
            ['FinalSigmas']
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
        params = DataBase.get_constituent_names_from_name(mod)
        for p in params:
            # if average_type == 'median':
            #     average_params_dict[mod][p] = np.median(
            #         params_dict[mod][p]
            #     )
            # else:
            #     average_params_dict[mod][p] = np.mean(
            #         params_dict[mod][p]
            #     )
            # if np.std(params_dict[mod][p]) > 0:                
            #     std_deviations[mod][p] = np.std(params_dict[mod][p])
            # else:
            #     # if only one winner, give relatively broad prior. 
            #     std_deviations[mod][p] = 0.5 
            

            # learned_priors[mod][p] = [
            #     average_params_dict[mod][p], 
            #     std_deviations[mod][p]
            # ]


            avg_sigmas_dict[mod][p] = np.median(sigmas_dict[mod][p])
            averaging_weight = [1/sig for sig in sigmas_dict[mod][p]]
            # print("[mod][p]:", mod, p)
            # print("Attempting to avg this list:", params_dict[mod][p])
            # print("with these weights:", averaging_weight)

            average_params_dict[mod][p] = np.average(
                params_dict[mod][p], 
                weights=sigmas_dict[mod][p]
            )
            # print("avg sigmas dict type:", type(avg_sigmas_dict[mod][p]))
            # print("type average_params_dict:", type(average_params_dict[mod][p]))
            # print("avg sigmas dict[mod][p]:", avg_sigmas_dict[mod][p])
            # print("average_params_dict[mod][p]:", average_params_dict[mod][p])
            learned_priors[mod][p] = [
                average_params_dict[mod][p], 
                avg_sigmas_dict[mod][p]
            ]
    
    return learned_priors   

def average_parameter_estimates(
    directory_name, 
    results_path, 
    results_file_name_start='results',
    growth_generator=None, 
    top_number_models=2,
    true_params_dict=None,
    save_to_file=None
):
    from matplotlib import cm
    plt.switch_backend('agg') #  to try fix plt issue on BC
    results = pandas.DataFrame.from_csv(
        results_path,
        index_col='QID'
    )

    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    rank_models = lambda n:sorted(set(n), key=n.count)[::-1] 
    # from https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

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
        fname = directory_name+'/'+str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']
        track_parameter_estimates = result['TrackParameterEstimates']
        
        # num_experiments = result['NumExperiments']
        if alph in parameter_estimates_from_qmd.keys():
            parameter_estimates_from_qmd[alph].append(track_parameter_estimates)
        else:
            parameter_estimates_from_qmd[alph] = [track_parameter_estimates]
            num_experiments_by_name[alph] = result['NumExperiments']

        if alph not in list(growth_rules.keys()):
            try:
                growth_rules[alph] = result['GrowthGenerator']
            except:
                growth_rules[alph] = growth_generator

    unique_growth_rules = list(set(list(growth_rules.values())))
    unique_growth_classes = {}
    for g in unique_growth_rules:
        try:
            unique_growth_classes[g] = GrowthRules.get_growth_generator_class(
                growth_generation_rule = g
            )
        except:
            unique_growth_classes[g] = None
    growth_classes = {}
    for g in list(growth_rules.keys()):
        try:
            growth_classes[g] = unique_growth_classes[growth_rules[g]]
        except:
            growth_classes[g] = None
    # print("[AnalyseMultiple - param avg] unique_growth_rules:", unique_growth_rules)
    # print("[AnalyseMultiple - param avg] unique_growth_classes:", unique_growth_classes)
    # print("[AnalyseMultiple - param avg] growth classes:", growth_classes)

    for name in winning_models:
        num_experiments = num_experiments_by_name[name]
        # epochs = range(1, 1+num_experiments)
        epochs = range(num_experiments_by_name[name])

        plt.clf()
        fig = plt.figure()
        ax = plt.subplot(111)

        parameters_for_this_name = parameter_estimates_from_qmd[name]
        num_wins_for_name = len(parameters_for_this_name)
        terms = sorted(DataBase.get_constituent_names_from_name(name))
        num_terms = len(terms)

        ncols = int(np.ceil(np.sqrt(num_terms)))
        nrows = int(np.ceil(num_terms/ncols))

        fig, axes = plt.subplots(
            figsize = (10, 7), 
            nrows=nrows, 
            ncols=ncols,
            squeeze=False,
        )
        row = 0
        col = 0
        axes_so_far = 0

        cm_subsection = np.linspace(0,0.8,num_terms)
#        colours = [ cm.magma(x) for x in cm_subsection ]
        colours = [ cm.Paired(x) for x in cm_subsection ]

        parameters = {}

        for t in terms:
            parameters[t] = {}

            for e in epochs:
                parameters[t][e] = []

        for i in range( len( parameters_for_this_name )):
            track_params =  parameters_for_this_name[i]
            for t in terms:
                for e in epochs:
                    parameters[t][e].append( track_params[t][e] )

        avg_parameters = {}
        std_devs = {}
        for p in terms :
            avg_parameters[p] = {}
            std_devs[p] = {}

            for e in epochs:
                avg_parameters[p][e] = np.median(parameters[p][e])
                std_devs[p][e] = np.std(parameters[p][e])

        for term in sorted(terms):
            ax = axes[row, col]
            axes_so_far += 1
            col += 1
            if (row==0 and col==ncols):
                leg=True
            else:
                leg=False

            if col == ncols:
                col=0
                row+=1
            # latex_terms[term] = DataBase.latex_name_ising(term)
            latex_terms[term] = growth_classes[name].latex_name(term)
            averages = np.array( 
                [ avg_parameters[term][e] for e in epochs  ]
            )
            standard_dev = np.array(
                [ std_devs[term][e] for e in epochs]
            )
            
            try:
                true_val = true_params_dict[term]
                # true_term_latex = DataBase.latex_name_ising(term)
                true_term_latex = growth_classes[name].latex_name(term)
                ax.axhline(
                    true_val, 
                    # label=str(true_term_latex+ ' True'), 
                    # color=colours[terms.index(term)]
                    label=str('True value'), 
                    color='black'

                )
            except:
                pass

            ax.axhline(
                0, 
                linestyle='--', 
                alpha=0.5, 
                color='black', 
                label='0'
            )
            ptq.fill_between_sigmas(
                ax, 
                parameters[term], 
                # [e +1 for e in epochs],
                epochs, 
                legend=leg
            )
            
            ax.scatter(
                [e +1 for e in epochs],
#                epochs, 
                averages, 
                s=max(1,50/num_experiments),
                label=latex_terms[term],
                # color=colours[terms.index(term)]
                color='black'
            )

            # latex_term = DataBase.latex_name_ising(term)
            latex_term = growth_classes[name].latex_name(term)
            # latex_term = latex_terms[term]
            ax.set_title(str(latex_term))
            
        """
        plot_title= str(
            'Average Parameter Estimates '+ 
            # str(DataBase.latex_name_ising(name)) +
            ' [' +
            str(num_wins_for_name) + # TODO - num times this model won 
            ' instances].'
        )
        ax.set_ylabel('Parameter Esimate')
        ax.set_xlabel('Experiment')
        plt.title(plot_title)
        ax.legend(
            loc='center left', 
            bbox_to_anchor=(1, 0.5), 
            title='Parameter'
        )    
        """
        
        latex_name = growth_classes[name].latex_name(term)

        if save_to_file is not None:
            fig.suptitle(
                'Parameter Esimates for {}'.format(latex_name)
            )
            save_file=''
            if save_to_file[-4:] == '.png':
                partial_name = save_to_file[:-4]
                save_file = str(partial_name + '_' + name + '.png')
            else:
                save_file = str(save_to_file + '_' + name + '.png')
            # plt.tight_layout()
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_file, bbox_inches='tight')


def Bayes_t_test(
    directory_name, 
    dataset, 
    results_path,
    results_file_name_start='results',
    use_experimental_data=False,
    true_expectation_value_path=None,
    growth_generator = None, 
    plot_probe_file = None,
    top_number_models=2,
    save_true_expec_vals_alone_plot=True,
    collective_analysis_pickle_file=None, 
    save_to_file=None
):
    plt.switch_backend('agg')
    print("Drawing avg expectation values from file:", results_path)
    from matplotlib import cm
    from scipy import stats

    
    results = pandas.DataFrame.from_csv(
        results_path,
        index_col='QID'
    )

    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    rank_models = lambda n:sorted(set(n), key=n.count)[::-1] 
    # from https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency
    
    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))    


    cm_subsection = np.linspace(0,0.8,len(winning_models))
    colours = [ cm.viridis(x) for x in cm_subsection ]
    # colours = [ cm.Spectral(x) for x in cm_subsection ]

    # Relies on Results folder structure -- not safe?!
    # ie ExperimentalSimulations/Results/Sep_10/14_32/results_001.p, etc
    if use_experimental_data == True:
        os.chdir(directory_name)
        os.chdir("../../../../ExperimentalSimulations/Data/")
        experimental_measurements = pickle.load(
            open(str(dataset), 'rb')
        )
    elif true_expectation_value_path is not None:
        experimental_measurements = pickle.load(
            open(str(true_expectation_value_path), 'rb')
        )
    else:
        print("Either set \
            use_experimental_data=True or \
            provide true_expectation_value_path"
        )
        return False

    expectation_values_by_name = {}
    os.chdir(directory_name)
    pickled_files = []
    for file in os.listdir(directory_name):
        # if file.endswith(".p") and file.startswith("results"):
        if (
            file.endswith(".p") 
            and 
            file.startswith(results_file_name_start)
        ):
            pickled_files.append(file)

    growth_rules = {}
    for f in pickled_files:
        fname = directory_name+'/'+str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']
        expec_values = result['ExpectationValues']

        if alph in expectation_values_by_name.keys():
            expectation_values_by_name[alph].append(expec_values)
        else:
            expectation_values_by_name[alph] = [expec_values]

        if alph not in list(growth_rules.keys()):
            growth_rules[alph] = result['GrowthGenerator']

    unique_growth_rules = list(set(list(growth_rules.values())))
    unique_growth_classes = {}
    for g in unique_growth_rules:
        try:
            unique_growth_classes[g] = GrowthRules.get_growth_generator_class(
                growth_generation_rule = g
            )
        except:
            unique_growth_classes[g] = None
    growth_classes = {}
    for g in list(growth_rules.keys()):
        try:
            growth_classes[g] = unique_growth_classes[growth_rules[g]]
        except:
            growth_classes[g] = None

    # print("[BayesTTest - param avg] unique_growth_rules:", unique_growth_rules)
    # print("[BayesTTest - param avg] unique_growth_classes:", unique_growth_classes)
    # print("[BayesTTest - param avg] growth classes:", growth_classes)
    true_model = unique_growth_classes[growth_generator].true_operator

    collect_expectation_values = {
        'means' : {},
        'medians' : {},
        'true' : {},
        'mean_std_dev' : {},
        'success_rate' : {},
    }
    success_rate_by_term = {}
    nmod = len(winning_models)  
    ncols = int(np.ceil(np.sqrt(nmod)))
    nrows = int(np.ceil(nmod/ncols)) + 1 # 1 extra row for "master"

    # fig = plt.figure()
    # ax = plt.subplot(111)

    # fig, axes = plt.subplots(
    #     figsize = (20, 10), 
    #     nrows=nrows, 
    #     ncols=ncols,
    #     squeeze=False,
    #     sharex='col',
    #     sharey='row'
    # )

    fig = plt.figure(
        figsize = (15, 8), 
        # constrained_layout=True,
        tight_layout=True
    )
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        nrows, 
        ncols, 
        # figure=fig # not available on matplotlib 2.1.1 (on BC)
    )

    row = 1
    col = 0

    axes_so_far = 1
    # full_plot_axis = axes[0,0]
    full_plot_axis = fig.add_subplot(gs[0,:])
    # i=0


    for term in winning_models:
        # plt.clf()
        # ax.clf()
        # ax = axes[row, col]
        ax = fig.add_subplot(gs[row, col])
        expectation_values = {}
        num_sets_of_this_name = len(
            expectation_values_by_name[term]
        )
        for i in range(num_sets_of_this_name):
            learned_expectation_values = (
                expectation_values_by_name[term][i]
            )

            for t in list(experimental_measurements.keys()):
                try:
                    expectation_values[t].append(
                        learned_expectation_values[t]
                    )
                except:
                    try:
                        expectation_values[t] = [
                            learned_expectation_values[t]
                        ]
                    except:
                        # if t can't be found, move on
                        pass

        means = {}
        std_dev = {}
        true = {}
        t_values = {}
        # times = sorted(list(experimental_measurements.keys()))
        true_times = sorted(list(expectation_values.keys()))
        times = sorted(list(expectation_values.keys()))
        flag=True
        one_sample=True
        for t in times:
            means[t] = np.mean(expectation_values[t])
            std_dev[t] = np.std(expectation_values[t])
            true[t] = experimental_measurements[t]
            if num_sets_of_this_name > 1:
                expec_values_array = np.array(
                    [ [i] for i in expectation_values[t]]
                )
                # print("shape going into ttest:", np.shape(true_expec_values_array))
                if use_experimental_data==True:
                    t_val = stats.ttest_1samp( 
                        expec_values_array, # list of expec vals for this t
                        true[t], # true expec val of t
                        axis=0, 
                        nan_policy='omit'
                    )
                else:
                    true_dist = stats.norm.rvs(
                        loc = true[t],
                        scale=0.001,
                        size=np.shape(expec_values_array)
                    )
                    t_val = stats.ttest_ind( 
                        expec_values_array, # list of expec vals for this t
                        true_dist, # true expec val of t
                        axis=0, 
                        nan_policy='omit'
                    )


                # if flag==True and t>0:
                #     print("t=", t)
                #     print("true:", expec_values_array)
                #     print("true", true[t])
                #     print("t_val:", t_val)
                #     flag=False

                if np.isnan(float(t_val[1]))==False:
                    # t_values[t] = 1-t_val[1]
                    t_values[t] = t_val[1]
                else:
                    print("t_val is nan for t=",t)

        true_exp = [true[t] for t in times]
        num_runs = num_sets_of_this_name # TODO should this be the number of times this model won???
        success_rate = 0

        for t in times: 

            true_likelihood = true[t]
            mean = means[t]
            std = std_dev[t]
            credible_region = ( 2/np.sqrt(num_runs) ) * std

            if (
                ( true_likelihood  < (mean + credible_region) )
                and
                ( true_likelihood > (mean - credible_region) ) 
            ):
                success_rate += 1/len(times)

        mean_exp = np.array( [means[t] for t in times] )
        std_dev_exp = np.array( [std_dev[t] for t in times] )
        # name=DataBase.latex_name_ising(term)
        residuals = (mean_exp - true_exp)**2
        sum_residuals = np.sum(residuals)
        mean_true_val = np.mean(true_exp)
        true_mean_minus_val = (true_exp - mean_true_val)**2 
        sum_of_squares = np.sum(
            true_mean_minus_val
        )
        final_r_squared = 1 - sum_residuals/sum_of_squares
        # print(
        #     "[Analyse - avg dynamics]\n",
        #     "\nmean_true_val", mean_true_val,
        #     "\ntrue", true_exp[0:10],
        #     "\nmean", mean_exp[0:10],
        #     "\nresiduals", residuals[0:10],
        #     "\nsum of residuals", sum_residuals,
        #     "\ntrue_mean_minus_val", true_mean_minus_val, 
        #     "\nsum of square", sum_of_squares,
        # )

        name = growth_classes[term].latex_name(term)
        description = str(
                name + 
                ' (' + str(num_sets_of_this_name)  + ')'
                + ' [$R^2=$' +
                str(
                    # np.round(final_r_squared, 2)
                    # np.format_float_scientific(
                    #     final_r_squared, 
                    #     precision=2
                    # )
                    format_exponent(final_r_squared)
                ) 
                + ']'
            )
        if term == true_model:
            description += ' (True)'

        description_w_bayes_t_value = str(
                name + ' : ' + 
                str(round(success_rate, 2)) + 
                ' (' + str(num_sets_of_this_name)  + ').'
            )


        collect_expectation_values['means'][name] = mean_exp
        collect_expectation_values['mean_std_dev'][name] = std_dev_exp
        collect_expectation_values['success_rate'][name] = success_rate

#        ax.errorbar(times, mean_exp, xerr=std_dev_exp, label=description)
        # if num_sets_of_this_name > 1:
        #     bayes_t_values_avail_times = sorted(list(t_values.keys()))
        #     bayes_t_values = [t_values[t] for t in bayes_t_values_avail_times]
        #     median_b_t_val = np.median(bayes_t_values)
        #     # print("Bayes t values:", bayes_t_values)

        #     ax.plot(
        #         bayes_t_values_avail_times, 
        #         bayes_t_values,
        #         label=str(
        #             'Bayes t-value (median '+ 
        #             str(np.round(median_b_t_val,2))+
        #             ')'
        #         ),
        #         color=colours[winning_models.index(term)],
        #         linestyle='--',
        #         alpha=0.3
        #     )

        ax.plot(
            times, 
            mean_exp, 
            c = colours[winning_models.index(term)],
            label=description
        )
        ax.fill_between(
            times, 
            mean_exp-std_dev_exp, 
            mean_exp+std_dev_exp, 
            alpha=0.2,
            facecolor = colours[winning_models.index(term)],
        )
        ax.set_ylim(0,1)
        ax.set_xlim(0,max(times))

        success_rate_by_term[term] = success_rate

        # plt.title('Mean Expectation Values')
        # plt.xlabel('Time')
        # plt.ylabel('Expectation Value')
        # true_exp = [true[t] for t in times]
        # ax.set_xlim(0,1)
        # plt.xlim(0,1)


        ax.set_title('Mean Expectation Values')
        # if col == 0:
        #     ax.set_ylabel('Expectation Value')
        # if row == nrows-1:
        #     ax.set_xlabel('Time')
        # ax.set_xlim(0,1)
        # plt.xlim(0,1)

        ax.scatter(
            times, 
            true_exp, 
            color='r', 
            s=5, 
            label='True Expectation Value'
        )
        ax.plot(
            times, 
            true_exp, 
            color='r', 
            alpha = 0.3
        )

        # ax.legend(
        #     loc='center left', 
        #     bbox_to_anchor=(1, 0.5), 
        #     title=' Model : Bayes t-test (instances)'
        # )    
        
        # fill in "master" plot



        high_level_label = str(name)
        if term == true_model:
            high_level_label += ' (True)'


        full_plot_axis.plot(
            times, 
            mean_exp, 
            c = colours[winning_models.index(term)],
            label=high_level_label
        )
        """
        full_plot_axis.fill_between(
            times, 
            mean_exp-std_dev_exp, 
            mean_exp+std_dev_exp, 
            alpha=0.2,
            facecolor = colours[winning_models.index(term)],
        )
        """
        if axes_so_far == 1:
            full_plot_axis.scatter(
                times, 
                true_exp, 
                color='r', 
                s=5, 
                label='True Expectation Value'
            )
            full_plot_axis.plot(
                times, 
                true_exp, 
                color='r', 
                alpha = 0.3
            )
        full_plot_axis.legend(
            loc='center left', 
            bbox_to_anchor=(1, 0), 
        )
        # full_plot_axis.legend(
        #     ncol = ncols,
        #     loc='lower center', 
        #     bbox_to_anchor=(0.5, -1.3), 
        # )
        full_plot_axis.set_ylim(0,1)
        full_plot_axis.set_xlim(0,max(times))

        axes_so_far += 1
        col += 1
        if col == ncols:
            col=0
            row+=1
        # ax.set_title(str(name))
        ax.set_title(description)


        # if save_to_file is not None:
        #     save_file=''
        #     # save_file = save_to_file[:-4]
        #     save_file = str(
        #         save_to_file[:-4]+
        #         '_'+
        #         str(term) + '.png'
        #     )
        #     print("Saving to ",save_file )
        #     plt.savefig(save_file, bbox_inches='tight')

    # fig.set_xlabel('Time')
    # fig.set_ylabel('Expectation Value')

    fig.text(0.45, -0.04, 'Time', ha='center')
    fig.text(-0.04, 0.5, 'Expectation Value', va='center', rotation='vertical')
    
    if save_to_file is not None:
        fig.suptitle("Expectation Values of learned models.")
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_to_file, bbox_inches='tight')


    # Also save an image of the true expectation values without overlaying results
    if save_true_expec_vals_alone_plot == True:
        plt.clf()
        # plt.scatter(
        #     times, 
        #     true_exp, 
        #     color='r', 
        #     s=5, 
        #     label='True Expectation Value'
        # )
        plt.plot(
            times, 
            true_exp, 
            marker='o',
            color='r', 
            label='True System'
            # alpha = 0.3
        )
        plt.xlabel('Time')
        plt.ylabel('Expectation Value')
        plt.legend()
        true_only_fig_file = str(
            save_to_file[:-4]
            + '_true_expec_vals.png'
        )
        plt.title("Expectation Values of True model.")
        plt.savefig(
            true_only_fig_file,
            bbox_inches='tight'
        )

    # add the combined analysis dict
    collect_expectation_values['times'] = true_times
    collect_expectation_values['true'] = true_exp


    if os.path.isfile(collective_analysis_pickle_file) is False:
        combined_analysis = {
            'expectation_values' : collect_expectation_values
        }
        pickle.dump(
            combined_analysis,
            open(collective_analysis_pickle_file, 'wb')
        )
    else:
        # load current analysis dict, add to it and rewrite it. 
        combined_analysis = pickle.load(
            open(collective_analysis_pickle_file, 'rb')
        ) 
        combined_analysis['expectation_values'] = collect_expectation_values
        pickle.dump(
            combined_analysis,
            open(collective_analysis_pickle_file, 'wb')
        )


def format_exponent(n):
    a = '%E' % n
    val = a.split('E')[0].rstrip('0').rstrip('.')
    val = np.round(float(val), 2)
    exponent = a.split('E')[1]
    
    return str(val) + 'E' + exponent

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

    results = pandas.DataFrame.from_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    rank_models = lambda n:sorted(set(n), key=n.count)[::-1] 
    # from https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency
    
    r_sq_by_model = {}

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))    

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0,0.8,num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [ cm.viridis(x) for x in cm_subsection ]

    i=0
    for i in range(len(names)):
        name = names[i]
        r_squared_values = list(
            results[ results['NameAlphabetical']==name]['RSquaredByEpoch']
        )

        r_squared_lists = {}
        num_wins = len(r_squared_values)
        for j in range(num_wins):
            rs = eval(r_squared_values[j])
            for t in list(rs.keys()):
                try:
                    r_squared_lists[t].append(rs[t])
                except:
                    r_squared_lists[t] = [rs[t]]

        times = sorted(list(r_squared_lists.keys()))
        means = np.array(
            [ np.mean(r_squared_lists[t]) for t in times]
        )
        std_dev = np.array(
            [ np.std(r_squared_lists[t]) for t in times]
        )

        # term = DataBase.latex_name_ising(name)
        gr_class = growth_classes_by_name[name]
        term = gr_class.latex_name(name) # TODO need growth rule of given name to get proper latex term
        # term = growth_class.latex_name(name) # TODO need growth rule of given name to get proper latex term
        r_sq_by_model[term] = means
        plot_label = str(term + ' ('+ str(num_wins) + ')')
        colour = colours[ i ]
        ax.plot(
            times, 
            means,
            label=plot_label,
            marker='o'
        )
        ax.fill_between(
            times, 
            means-std_dev,
            means+std_dev,
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

    results = pandas.DataFrame.from_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    rank_models = lambda n:sorted(set(n), key=n.count)[::-1] 
    # from https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency
    
    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))    

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0,0.8,num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [ cm.viridis(x) for x in cm_subsection ]

    i=0
    for i in range(len(names)):
        name = names[i]
        volume_values = list(
            results[ results['NameAlphabetical']==name]['TrackVolume']
        )

        volume_lists = {}
        num_wins = len(volume_values)
        for j in range(num_wins):
            rs = eval(volume_values[j])
            for t in list(rs.keys()):
                try:
                    volume_lists[t].append(rs[t])
                except:
                    volume_lists[t] = [rs[t]]

        times = sorted(list(volume_lists.keys()))
        means = np.array(
            [ np.mean(volume_lists[t]) for t in times]
        )

        std_dev = np.array(
            [ np.std(volume_lists[t]) for t in times]
        )

        # term = DataBase.latex_name_ising(name)
        term = growth_class.latex_name(name)
        plot_label = str(term + ' ('+ str(num_wins) + ')')
        colour = colours[ i ]
        ax.plot(
            times, 
            means,
            label=plot_label,
            marker='o',
            markevery=10
        )
        ax.fill_between(
            times, 
            means-std_dev,
            means+std_dev,
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


def all_times_learned_histogram(
    results_path = "summary_results.csv",
    top_number_models=2,
    save_to_file=None
):
    from matplotlib import cm
    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    results = pandas.DataFrame.from_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    rank_models = lambda n:sorted(set(n), key=n.count)[::-1] 
    # from https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))    

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0,0.8,num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [ cm.viridis(x) for x in cm_subsection ]

    times_by_model = {}
    max_time = 0
    for i in range(len(names)):
        name = names[i]
        model_colour = colours[i]
        times_by_model[name] = []
        this_model_times_separate_runs = list(
            results[ results['NameAlphabetical']==name]['TrackTimesLearned']
        )

        num_wins = len(this_model_times_separate_runs)
        for j in range(num_wins):
            this_run_times = eval(this_model_times_separate_runs[j])
            times_by_model[name].extend(this_run_times)
            if max(this_run_times) > max_time:
                max_time = max(this_run_times)
        times_this_model = times_by_model[name]
        model_label = str(
            list(results[results['NameAlphabetical']==name]['ChampLatex'])[0]
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

    # presuming all models used same heuristics .... TODO change if models can use ones
    heuristic_type = list(results[results['NameAlphabetical']==names[0]]['Heuristic'])[0]

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




        
def get_model_scores(directory_name):
#    sys.path.append(directory_name)

    os.chdir(directory_name)

    scores = {}
    growth_rules = {}

    pickled_files = []
    for file in os.listdir(directory_name):
        if file.endswith(".p") and file.startswith("results"):
            pickled_files.append(file)
    
    for f in pickled_files:
        fname = directory_name+'/'+str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']

        if alph in scores.keys():
            scores[alph] += 1
        else:
            scores[alph] = 1

        if alph not in list(growth_rules.keys()):
            growth_rules[alph] = result['GrowthGenerator']
    unique_growth_rules = list(set(list(growth_rules.values())))
    unique_growth_classes = {}
    for g in unique_growth_rules:
        try:
            unique_growth_classes[g] = GrowthRules.get_growth_generator_class(
                growth_generation_rule = g
            )
        except:
            unique_growth_classes[g] = None
    growth_classes = {}
    for g in list(growth_rules.keys()):
        try:
            growth_classes[g] = unique_growth_classes[growth_rules[g]]
        except:
            growth_classes[g] = None

    return scores, growth_rules, growth_classes, unique_growth_classes

def get_entropy(
    models_points, 
    growth_generator=None,
    inf_gain=False
):
    # TODO this calculation of entropy may not be correct
    # What is initial_entropy meant to be?
    num_qmd_instances = sum(list(models_points.values()))
    num_possible_qmd_instances = len(
        # ptq.ising_terms_rotation_hyperfine()
        UserFunctions.get_all_model_names(
            growth_generator = growth_generator,
            return_branch_dict = 'latex_terms'
        )
    )
    # TODO don't always want ising terms only

    
    model_fractions = {}
    for k in list(models_points.keys()):
        model_fractions[k] = models_points[k]/num_qmd_instances    
    
    initial_entropy = -1*np.log2(1/num_possible_qmd_instances)
    entropy = 0
    for i in list(models_points.keys()):
        success_prob = model_fractions[i]
        partial_entropy = success_prob * np.log2(success_prob)
        if np.isnan(partial_entropy):
            partial_entropy = 0 
        entropy -= partial_entropy
    
    if inf_gain:
        # information gain is entropy loss
        information_gain =  initial_entropy - entropy
        return information_gain
    else:
        return entropy


def plot_scores(
        scores, 
        growth_classes, 
        unique_growth_classes, 
        growth_rules, 
        entropy=None,
        inf_gain=None, 
        true_operator = None, 
        growth_generator = None,
        batch_nearest_num_params_as_winners = True,
        collective_analysis_pickle_file =  None, 
        save_file='model_scores.png'
    ):
    plt.clf()
    models = list(scores.keys())

    # print("[AnalyseMultiple - plot_scores] growth classes:",growth_classes )
    # print("[AnalyseMultiple - plot_scores] unique_growth_classes:",unique_growth_classes )
    latex_true_op = unique_growth_classes[growth_generator].latex_name(
        name = true_operator
    )    

    latex_model_names = [
        growth_classes[model].latex_name(model)
        for model in models
    ]

    latex_scores_dict = {}
    for mod in models:
        latex_mod = growth_classes[mod].latex_name(mod)
        latex_scores_dict[latex_mod] = scores[mod]

    batch_correct_models = []
    if batch_nearest_num_params_as_winners == True:
        num_true_params = len(
            DataBase.get_constituent_names_from_name(
                true_operator
            )
        )
        for mod in models:
            num_params = len(
                DataBase.get_constituent_names_from_name(mod)
            )

            if (
                num_true_params - num_params == 1
            ):
                # must be exactly one parameter smaller
                batch_correct_models.append(
                    mod
                )

    mod_scores = scores
    scores = list(scores.values())
    num_runs = sum(scores)
    fig, ax = plt.subplots()    
    width = 0.75 # the width of the bars 
    ind = np.arange(len(scores))  # the x locations for the groups
    colours = ['blue' for i in ind]
    batch_success_rate = correct_success_rate = 0
    for mod in batch_correct_models: 
        mod_latex = growth_classes[mod].latex_name(mod)
        mod_idx = latex_model_names.index(mod_latex)
        colours[mod_idx] = 'orange'
        batch_success_rate += mod_scores[mod]
    if true_operator in models:
        batch_success_rate += mod_scores[true_operator]
        correct_success_rate = mod_scores[true_operator]

    batch_success_rate /= num_runs
    correct_success_rate /= num_runs
    batch_success_rate *= 100
    correct_success_rate *= 100 #percent

    results_collection = {
        'type' : growth_generator, 
        'true_model' : latex_true_op, 
        'scores' : latex_scores_dict
    }
    print("[Analyse] results_collection", results_collection)

    # if save_results_collection is not None:
    #     print("[Analyse] save results collection:", save_results_collection)
    #     pickle.dump(
    #         results_collection, 
    #         open(
    #             save_results_collection, 
    #             'wb'
    #         )
    #     )
    if os.path.isfile(collective_analysis_pickle_file) is False:
        combined_analysis = {
            'scores' : results_collection
        }
        pickle.dump(
            combined_analysis,
            open(collective_analysis_pickle_file, 'wb')
        )
    else:
        # load current analysis dict, add to it and rewrite it. 
        combined_analysis = pickle.load(
            open(collective_analysis_pickle_file, 'rb')
        ) 
        combined_analysis['scores'] = results_collection
        pickle.dump(
            combined_analysis,
            open(collective_analysis_pickle_file, 'wb')
        )



    try:
        true_idx = latex_model_names.index(
            latex_true_op
        )
        colours[true_idx] = 'green'

    except:
        pass


    # ax.barh(ind, scores, width, color="blue")
    ax.barh(ind, scores, width, color=colours)
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(latex_model_names, minor=False)
    custom_lines = [
        Line2D([0], [0], color='green', lw=4),
        Line2D([0], [0], color='orange', lw=4),
        Line2D([0], [0], color='blue', lw=4),
    ]
    custom_handles = [
        'True ({}%)'.format(int(correct_success_rate)), 
        'True/Close ({}%)'.format(int(batch_success_rate)), 
        'Other'
    ]
    
    plot_title = str(
        'Number of QMD instances won by models.' 
    )

    if entropy is not None:
        plot_title += str( 
            '\n$\mathcal{S}$=' 
            + str(round(entropy, 2))
        )
    if inf_gain is not None:
        plot_title += str(
            '\t $\mathcal{IG}$=' 
            + str(round(inf_gain, 2))
        )
    plt.legend(custom_lines, custom_handles)
    plt.title(plot_title)
    plt.ylabel('Model')
    plt.xlabel('Number of wins')
    #plt.bar(scores, latex_model_names)
    
    plt.savefig(save_file, bbox_inches='tight')
    
    
def plot_tree_multi_QMD(
        results_csv, 
        all_bayes_csv, 
        latex_mapping_file,
        avg_type='medians',
        growth_generator=None,
        entropy=None, 
        inf_gain=None, 
        save_to_file=None
    ):
    try:
        qmd_res = pandas.DataFrame.from_csv(
            results_csv, 
            index_col='LatexName'
        )
    except ValueError:
        print(
            "Latex Name not in results CSV keys.", 
            "There aren't enough data for a tree of multiple QMD."
            "This may be because this run was for QHL rather than QMD."
        )
        raise


    mods = list(qmd_res.index)
    winning_count = {}
    for mod in mods:
        winning_count[mod]=mods.count(mod)

    ptq.cumulativeQMDTreePlot(
        cumulative_csv=all_bayes_csv, 
        wins_per_mod=winning_count,
        latex_mapping_file=latex_mapping_file, 
        growth_generator=growth_generator,
        only_adjacent_branches=True, 
        avg=avg_type, 
        entropy=entropy, 
        inf_gain=inf_gain,
        save_to_file=save_to_file
    )        


def count_model_occurences(
    latex_map, 
    true_operator_latex,
    save_counts_dict=None,
    save_to_file=None
):
    f = open(latex_map, 'r')
    l = str(f.read())
    terms = l.split("',")

    # for t in ["(", ")", "'", " "]:
    for t in ["'", " "]:
        terms = [a.replace(t, '') for a in terms]

    sep_terms = []
    for t in terms:
        sep_terms.extend(t.split("\n"))

    unique_models = list(set([s for s in sep_terms if "$" in s]))
    counts = {}
    for ln in unique_models:
        counts[ln] = sep_terms.count(ln)
    unique_models = sorted(unique_models)
    model_counts = [counts[m] for m in unique_models]
    unique_models = [
        a.replace("\\\\", "\\")
        for a in unique_models
    ] # in case some models have too many slashes. 
    max_count = max(model_counts)
    integer_ticks = list(range(max_count+1))
    colours = ['blue' for m in unique_models]
    
    if true_operator_latex in unique_models:
        true_idx = unique_models.index(true_operator_latex)
        colours[true_idx] = 'green'
    
    fig, ax = plt.subplots(
        figsize=(
            max(max_count*2, 5),
            len(unique_models)/4)
        )
    ax.plot(kind='barh')
    ax.barh(
        unique_models, 
        model_counts,
        color = colours
    )
    ax.set_xticks(integer_ticks)
    ax.set_title('# times each model generated')
    ax.set_xlabel('# occurences')
    ax.tick_params(
        top=True, 
        direction='in'
    ) 
    if save_counts_dict is not None:
        import pickle
        pickle.dump(
            counts, 
            open(
                save_counts_dict,
                'wb'
            )
        )
    
    try:
        if save_to_file is not None:
            plt.savefig(save_to_file)
    except:
        print(
            "[AnalyseMultiple - count model occurences] couldn't save plot to file", 
            save_to_file

        )
        raise