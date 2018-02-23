from __future__ import print_function # so print doesn't show brackets
import numpy as np
import itertools as itr

import os as os
import sys as sys 
import pandas as pd
import warnings
import time as time
import random
import pickle

sys.path.append(os.path.join("..", "Libraries","QML_lib"))
import Evo as evo
import DataBase 
from QMD import QMD #  class moved to QMD in Library
import QML
import ModelGeneration 
import BayesF
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
paulis = ['x', 'y', 'z'] # will be chosen at random. or uncomment below and comment within loop to hard-set

import time as time 
import argparse
parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')


def get_directory_name_by_time(just_date=False):
    import datetime
    # Directory name based on date and time it was generated 
    # from https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
    now =  datetime.date.today()
    year = now.strftime("%y")
    month = now.strftime("%b")
    day = now.strftime("%d")
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    date = str (str(day)+'_'+str(month)+'_'+str(year) )
    time = str(str(hour)+'_'+str(minute))
    name = str(date+'/'+time+'/')
    if just_date is False:
        return name
    else: 
        return str(date+'/')

### Set up command line arguments to alter script parameters. ###

parser.add_argument(
  '-t', '--num_tests', 
  help="Number of complete tests to average over.",
  type=int,
  default=1
)

parser.add_argument(
  '-e', '--num_experiments', 
  help='Number of experiments to use for the learning process',
  type=int,
  default=200
)

parser.add_argument(
  '-p', '--num_particles', 
  help='Number of particles to use for the learning process',
  type=int,
  default=300
)

parser.add_argument(
  '-q', '--num_qubits', 
  help='Number of qubits to run tests for.',
  type=int,
  default=2
)
parser.add_argument(
  '-pm', '--num_parameters', 
  help='Number of parameters to run tests for.',
  type=int,
  default=1
)

parser.add_argument(
  '-qle',
  help='True to perform QLE, False otherwise.',
  type=int,
  default=1
)
parser.add_argument(
  '-iqle',
  help='True to perform IQLE, False otherwise.',
  type=int,
  default=1
)
parser.add_argument(
  '-pt', '--plots',
  help='True: do generate all plots for this script; False: do not.',
  type=int,
  default=0
)
parser.add_argument(
  '-rt', '--resample_threshold',
  help='Resampling threshold for QInfer.',
  type=float,
  default=0.6
)
parser.add_argument(
  '-ra', '--resample_a',
  help='Resampling a for QInfer.',
  type=float,
  default=0.9
)
parser.add_argument(
  '-pgh', '--pgh_factor',
  help='Resampling threshold for QInfer.',
  type=float,
  default=1.0
)
parser.add_argument(
  '-vary_rt', '--vary_resample_threshold',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=bool,
  default=False
)
parser.add_argument(
  '-vary_ra', '--vary_resample_a',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=bool,
  default=False
)
parser.add_argument(
  '-vary_pgh', '--vary_pgh_factor',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=bool,
  default=False
)

arguments = parser.parse_args()
do_iqle = bool(arguments.iqle)
do_qle = bool(arguments.qle)
num_tests = arguments.num_tests
num_qubits = arguments.num_qubits
num_parameters = arguments.num_parameters
num_exp = arguments.num_experiments
num_part = arguments.num_particles
all_plots = bool(arguments.plots)
best_resample_threshold = arguments.resample_threshold
best_resample_a = arguments.resample_a
best_pgh = arguments.pgh_factor
vary_resample_a = arguments.vary_resample_threshold
vary_resample_thresh = arguments.vary_resample_a
vary_pgh_factor = arguments.vary_pgh_factor

#######

plot_time = get_directory_name_by_time(just_date=False) # rather than calling at separate times and causing confusion
save_figs = False
save_data = False

intermediate_plots = all_plots
do_summary_plots = all_plots
store_data = all_plots

global_true_op = ModelGeneration.random_model_name(num_dimensions=num_qubits, num_terms=num_parameters)  # choose a random initial Hamiltonian.

while global_true_op == 'i':
  global_true_op = ModelGeneration.random_model_name(num_dimensions=num_qubits, num_terms=num_parameters)  # choose a random initial Hamiltonian.


global paulis_list
paulis_list = {'i' : np.eye(2), 'x' : evo.sigmax(), 'y' : evo.sigmay(), 'z' : evo.sigmaz()}

warnings.filterwarnings("ignore", message='Negative weights occured', category=RuntimeWarning)


#####################################

### Plotting functions 

#######################################

start = time.time() 

def store_data_for_plotting(iqle_qle):
    #qlosses=[]
    if iqle_qle == 'qle':
        qlosses = qle_qlosses
        final_qlosses = qle_final_qloss
        qle_type = 'QLE'
        differences = qle_differences
    elif iqle_qle == 'iqle':
        qlosses = iqle_qlosses
        final_qlosses = iqle_final_qloss
        qle_type = 'IQLE'
        differences = iqle_differences

    else:
        print("Needs to either be QLE or IQLE")
        
        
    ### Format data to be used in plotting
        
    exp_values = {}
    for i in range(num_exp):
        exp_values[i] = []

    for i in range(num_tests): 
        for j in range(len(qlosses[i])):
            exp_values[j].append(qlosses[i][j])

    medians=[]        
    std_dev=[]
    means=[]
    mins=[]
    maxs = []
    for k in range(num_exp):
        medians.append(np.median(exp_values[k]) )
        means.append(np.mean(exp_values[k]) )
        mins.append(np.min(exp_values[k]) )
        maxs.append(np.max(exp_values[k]) )
        std_dev.append(np.std(exp_values[k]) )
    
    if iqle_qle == 'qle':
        qle_all_medians.append(medians)
        qle_all_means.append(means)
        qle_all_mins.append(mins)
        qle_all_maxs.append(maxs)
        qle_all_std_devs.append(std_dev)
        qle_descriptions_of_runs.append(description)
    elif iqle_qle == 'iqle':
        iqle_all_medians.append(medians)
        iqle_all_means.append(means)
        iqle_all_mins.append(mins)
        iqle_all_maxs.append(maxs)
        iqle_all_std_devs.append(std_dev)
        iqle_descriptions_of_runs.append(description)
        
#    old_descriptions_of_runs[len(old_descriptions_of_runs)] = description
    
    
def individual_plots(iqle_qle):            
    if iqle_qle == 'qle':
        qlosses = qle_qlosses
        final_qlosses = qle_final_qloss
        qle_type = 'QLE'
        differences = qle_differences
        medians = qle_all_medians
        means = qle_all_means
        mins = qle_all_mins
        maxs = qle_all_maxs
        std_devs = qle_all_std_devs

        
    elif iqle_qle == 'iqle':
        qlosses = iqle_qlosses
        final_qlosses = iqle_final_qloss
        qle_type = 'IQLE'
        differences = iqle_differences
        medians = iqle_all_medians
        means = iqle_all_means
        mins = iqle_all_mins
        maxs = iqle_all_maxs
        std_devs = iqle_all_std_devs
    else:
        print("Needs to either be QLE or IQLE")

    ##### Plots #####
    ### Overall results
    
    # Errors histogram
    # %matplotlib inline
    plot_description = 'Errors_histogram'
    plt.clf()
    plt.hist(differences, normed=False, bins=30)
    plt.ylabel('Count of '+ qle_type +' runs');
    plt.xlabel('Error');
    plt.title('Count of '+ qle_type +' runs by error margin '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    
    # Quadratic Loss histogram
    # %matplotlib inline
    plot_description='Final_quadratic_loss'
    plt.clf()
    plt.hist(final_qlosses, normed=False, bins=30)
    plt.ylabel('Count of ' +qle_type+' runs');
    plt.xlabel('Error');
    plt.title('Count of ' +qle_type+' runs by final Quadratic Loss '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)

    # Quadratic loss development of all tests 
    # %matplotlib inline
    plot_description='All_quadratic_loss'
    plt.clf()
    for i in range(num_tests):
        plt.semilogy( range(1,1+len(qlosses[i])), list(qlosses[i]))
    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type+' Quadratic Loss per experiment '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    
    
    
            
    ### Averages
    
    """
    # Median Quadratic loss with error bar
    # Not useful here but kept for use case of errorbar function
    # %matplotlib inline
    plot_description = 'Median_with_error_bar'
    plt.clf()
    y = medians
    x = range(1,1+num_exp)
    err = std_dev
    fig,ax = plt.subplots()
    ax.errorbar(x, y, yerr=err)
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')
    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Median Quadratic Loss '+title_appendix);
    plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    """
    # Median with shadowing
    # %matplotlib inline
    plot_description = 'Median_with_std_dev'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = medians[0]
    x = range(1,1+num_exp)
    #print("Different length lists")
    devs = std_devs[0]
    y_lower = [( y[i] - devs[i]) for i in range(len(y))]
    y_upper = [ (y[i] + devs[i] )for i in range(len(y))]
    #y_lower = np.array(y[0])  - np.array(std_devs[0]) 
    #y_upper = np.array(y[0])  + np.array(std_devs[0])
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Median Quadratic Loss '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    
    
    # Median with shadowing
    # %matplotlib inline
    plot_description = 'Median_min_max'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = medians[0]
    
    x = range(1,1+num_exp)
    y_lower = maxs[0]
    y_upper = mins[0]
    
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Median Quadratic Loss (Max/Min) '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
        
    
    # Mean with shadowing
    # %matplotlib inline
    plot_description = 'Mean_with_std_dev'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = means[0]
    devs =  std_devs[0]
    x = range(1,1+num_exp)
    y_lower = [( y[i] - devs[i]) for i in range(len(y))]
    y_upper = [ (y[i] + devs[i] )for i in range(len(y))]
    
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Mean Quadratic Loss '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)

    # Mean with shadowing for min/max
    # %matplotlib inline
    plot_description = 'Mean_min_max'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = means[0]
    x = range(1,1+num_exp)
    y_lower = mins[0]
    y_upper = maxs[0]
    
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Mean Quadratic Loss (Max/Min) '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)

    
def draw_summary_plots(iqle_qle):
    if iqle_qle == 'qle':
        qlosses = qle_qlosses
        final_qlosses = qle_final_qloss
        qle_type = 'QLE'
        differences = qle_differences
        all_qlosses = all_qle_final_qlosses
        descriptions_of_runs = qle_descriptions_of_runs
        all_medians = qle_all_medians
        all_means = qle_all_means
        all_mins = qle_all_mins
        all_maxs = qle_all_maxs
        all_std_devs = qle_all_std_devs
        
    elif iqle_qle == 'iqle':
        qlosses = iqle_qlosses
        final_qlosses = iqle_final_qloss
        qle_type = 'IQLE'
        differences = iqle_differences
        all_qlosses = all_iqle_final_qlosses
        descriptions_of_runs = iqle_descriptions_of_runs
        all_medians = iqle_all_medians
        all_means = iqle_all_means
        all_mins = iqle_all_mins
        all_maxs = iqle_all_maxs
        all_std_devs = iqle_all_std_devs
    else:
        print("Needs to either be QLE or IQLE")

    # Correlation diagram between Quadratic Loss and distance of true param from center of prior    
    distance_from_center = [ item - 0.5 for item in all_true_params_single_list]
    # %matplotlib inline
    plot_description = 'Correlation_distance_quad_loss_'+variable_parameter
    plt.clf()
    plt.xlabel('Distance from center of Prior distribution')
    plt.ylabel('Final Quadratic Loss')
    plt.gca().set_yscale('log')
    x = distance_from_center
    y = all_qlosses
    plt.scatter(x,y)        
    plt.title(qle_type + ' Correlation between distance from center and final quadratic loss');
    plt.savefig(summary_plot_directory+qle_type+'_'+ plot_description)

    
    # Median Quadratic loss with variable resampling threshold
    # %matplotlib inline
    plot_description='Median_Q_Loss_w_variable_'+variable_parameter
    plt.clf()
    for i in range(len(all_medians)):
        plt.semilogy( range(1,1+(num_exp)), list(all_medians[i]), label=descriptions_of_runs[i])

    ax = plt.subplot(111)
    ## Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ## Put a legend below current axis
    lgd=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=4)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type+' Median Quadratic Loss with variable '+ variable_parameter);
    plt.savefig(summary_plot_directory+qle_type+'_'+ plot_description, bbox_extra_artists=(lgd,), bbox_inches='tight')


    # Mean Quadratic loss with variable resampling threshold
    # %matplotlib inline
    plot_description='Mean_Q_Loss_w_variable_'+variable_parameter
    plt.clf()
    for i in range(len(all_medians)):
        plt.semilogy( range(1,1+(num_exp)), list(all_means[i]), label=descriptions_of_runs[i])
    ax = plt.subplot(111)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    lgd=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=4)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type+' Mean Quadratic Loss with variable '+ variable_parameter);
    plt.savefig(summary_plot_directory+qle_type+'_'+ plot_description, bbox_extra_artists=(lgd,), bbox_inches='tight')




##########################################

####### run loops 

#########################################


# This cell runs tests on IQLE and QLE and plots the resulting errors and quadratic losses


all_true_ops = []
all_true_params = []
all_true_params_single_list = []
all_qle_final_qlosses = []
all_iqle_final_qlosses = []

qle_all_medians = []
qle_all_means = []
qle_all_maxs = []
qle_all_mins = []
qle_all_std_devs = []

iqle_all_medians = []
iqle_all_means = []
iqle_all_maxs = []
iqle_all_mins = []
iqle_all_std_devs = []


iqle_descriptions_of_runs = []
qle_descriptions_of_runs = []
old_descriptions_of_runs = {}




if vary_resample_thresh or vary_resample_a or vary_pgh_factor:
    variable_parameter = 'vary'
else:
    variable_parameter = ''


a_options = [best_resample_a]
resample_threshold_options = [best_resample_threshold]
pgh_options = [best_pgh]
    
if vary_resample_thresh : 
    variable_parameter += '_thresh'
    resample_threshold_options = np.arange(0.35, 0.75, 0.1)

if vary_resample_a: 
    variable_parameter += '_a'
    a_options = np.arange(0.85, 0.99, 0.05)

if vary_pgh_factor:
    variable_parameter += '_pgh'
    pgh_options = np.arange(0.6, 1.5, 0.4)

# RUN QMD Loops

qmd_time = 0


for resample_thresh in resample_threshold_options:
    for resample_a in a_options:
        for pgh_factor in pgh_options:


            qle_list = []
            iqle_list = []
            true_param_list = []
            true_op_list =[]
            qle_differences = []
            iqle_differences = []
            qle_qlosses = []
            qle_final_qloss =[]
            iqle_qlosses =[]
            iqle_final_qloss =[]




            for i in range(num_tests):
                print("Test ", i)
                initial_op_list = ['x', 'y', 'z', 'xTy', 'xTz', 'yTz']
#                initial_op_list = ['x', 'y', 'z']
                true_params = [np.random.rand() for i in range(num_parameters)]
                true_param_list.append(true_params[0])
#                true_op = global_true_op
                true_op = random.choice(initial_op_list)
#                true_op = 'xTy'
                global_true_op = true_op
                true_op_list.append(true_op)
                                
                # (Note: not learning between models yet; just learning paramters of true model)


                qle_values = [] # qle True does QLE; False does IQLE
                if do_qle is True:
                    qle_values.append(True)
                if do_iqle is True:
                    qle_values.append(False)

                for qle in qle_values:
                    a = time.time() # track time in just QMD
                    qmd = QMD(
                        initial_op_list=initial_op_list, 
                        true_operator=true_op, 
                        true_param_list=true_params, 
                        num_particles=num_part,
                        qle=qle,
                        max_num_branches = 0,
                        max_num_qubits = 2, 
                        resample_threshold = resample_thresh,
                        resampler_a = resample_a,
                        pgh_prefactor = pgh_factor
                    )
                    # qmd.runAllActiveModelsIQLE(num_exp=num_exp)
                    qmd.runQMD(num_exp = num_exp, spawn=False)
#                    qmd.runAllActiveModelsIQLE(num_exp = num_exp)
                    b = time.time()
                    qmd_time += b-a

                    mod = qmd.getModelInstance(true_op)
                    if qle is True:
                        qle_list.append(mod.FinalParams[0][0])
                        qle_qlosses.append(mod.QLosses)
                        qle_final_qloss.append(mod.QLosses[-1])
                    else: 
                        iqle_list.append(mod.FinalParams[0][0])
                        iqle_qlosses.append(mod.QLosses)
                        iqle_final_qloss.append(mod.QLosses[-1])

                    champion = qmd.ChampionName
                    # Load QMD instance into pickle file to read in notebook.
                    pickle.dump(qmd, open("qmd_class.npy", "wb"))
                    # qmd.killModel(true_op)
                    del qmd

            all_true_params.append(true_param_list)
            all_true_params_single_list.extend(true_param_list)
            all_true_ops.append(true_op_list)
            all_qle_final_qlosses.extend(qle_final_qloss)
            all_iqle_final_qlosses.extend(iqle_final_qloss)



            for i in range(num_tests):
                if do_iqle:
                    iqle_diff =np.abs(true_param_list[i]-iqle_list[i])
                    iqle_differences.append(iqle_diff)
                if do_qle:
                    qle_diff =np.abs(true_param_list[i]-qle_list[i])
                    qle_differences.append(qle_diff)



            # Plotting
            title_appendix = str( '(single parameter, ' +str(num_tests)+' runs)')

            plot_title_appendix = str( str(num_exp)+'_exp_'+str(num_part)+'_particles_'+str(num_tests)+'_runs.png')
            #plot_directory = 'test_plots/'+plot_time
            param_details = str('a_'+str(resample_a)+'_thresh_'+str(resample_thresh)+'_pgh_'+str(pgh_factor))

            #plot_directory = 'test_plots/'+plot_time+'/'+plot_title_appendix+'/'+
            plot_directory = 'test_plots/'+plot_time+'/'+plot_title_appendix+'/'+param_details+'/'

            if intermediate_plots:
                if not os.path.exists(plot_directory):
                    os.makedirs(plot_directory)

            description = str('('+ str(resample_thresh)+ ', '+ str(resample_a) + ', '+ str(pgh_factor)+')')
            if do_iqle:
                if store_data: store_data_for_plotting('iqle')
                if intermediate_plots: individual_plots('iqle')

            if do_qle:  
                
                if store_data: print("storing data for QLE")
                if store_data: store_data_for_plotting('qle')
                if intermediate_plots: individual_plots('qle')

            names_to_save = ['true_param_list', 'true_op_list']
            lists_to_save = [true_param_list, true_op_list]

            if do_qle:
                names_to_save.extend(['qle_list', 'qle_qlosses', 'qle_final_qloss'])
                lists_to_save.extend([qle_list, qle_qlosses, qle_final_qloss])

            if do_iqle:
                names_to_save.extend(['iqle_list', 'iqle_qlosses', 'iqle_final_qloss'])
                lists_to_save.extend([iqle_list, iqle_qlosses, iqle_final_qloss])

            running_data = dict(zip(names_to_save, lists_to_save))


            for item in running_data.keys():
                filename = plot_directory+str(item)+'.txt'
                data_to_save = running_data[item]
                if type(data_to_save)==list:
                    if save_data: np.savetxt(filename, data_to_save, fmt="%s")    
                else: 
                    if save_data: np.savetxt(filename, data_to_save)  



c=time.time()
summary_plot_directory = 'test_plots/'+plot_time+'/'+plot_title_appendix+'/Summary_Plots/'
if do_summary_plots:
    if not os.path.exists(summary_plot_directory):
        os.makedirs(summary_plot_directory)

                
if do_iqle:
    if do_summary_plots: draw_summary_plots('iqle')
    
    
if do_qle:
    if do_summary_plots: draw_summary_plots('qle')
    

d=time.time()
end = time.time()

num_qle_types = do_iqle + do_qle
num_exponentiations = num_qle_types * num_part * num_exp*num_tests

print("\n\n\n")
if all_plots: print("Plots made with matplotlib.")
if not all_plots: print("Plots NOT drawn.")                                       
print(num_qubits, "Qubits")
print(num_qle_types, "Types of (I)QLE")
print(num_part, "Particles")
print(num_exp, "Experiments")
print(num_tests, "Tests")
print("Totalling ", num_exponentiations, " calls to ", num_qubits,"-qubit exponentiation function.")
print("QMD-time / num exponentiations = ", qmd_time/num_exponentiations)
print("Total time on QMD : ", qmd_time, "seconds.")
if all_plots: print("Total time on plotting : ", d-c, "seconds.")
print("Total time : ", end - start, "seconds.")
print("True model: ", global_true_op)
print("QMD Champion:", champion)









