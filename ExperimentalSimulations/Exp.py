from __future__ import print_function # so print doesn't show brackets
import os as os

import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import itertools as itr

import sys as sys 
import pandas as pd
import warnings
import time as time
import random
import pickle
pickle.HIGHEST_PROTOCOL = 2
sys.path.append(os.path.join("..", "Libraries","QML_lib"))

## Parse input variables to use in QMD; store in class global_variables. 
import GlobalVariables
global_variables = GlobalVariables.parse_cmd_line_args(sys.argv[1:])

import RedisSettings as rds
import Evo as evo
import DataBase 
import ExperimentalDataFunctions as expdt
from QMD import QMD #  class moved to QMD in Library
import QML
import ModelGeneration 
import matplotlib.pyplot as plt
#from pympler import asizeof
import matplotlib.pyplot as plt
import time as time 

###  START QMD ###
start = time.time()

"""
Set up and functions. 
"""

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time

def log_print(to_print_list, log_file):
    identifier = str(str(time_seconds()) +" [EXP]")
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, 
            str(to_print),
            file=write_log_file,
            flush=True
        )

log_file = global_variables.log_file
qle = global_variables.do_qle # True for QLE, False for IQLE
num_probes = 40

if global_variables.use_experimental_data == True:
# TODO reinstate regualar probes

    experimental_probe_dict = expdt.experimental_NVcentre_ising_probes(
        num_probes=num_probes
    )
else:
    experimental_probe_dict = None

"""
if global_variables.qhl_test:
    experimental_probe_dict = expdt.experimental_NVcentre_ising_probes_plusplus(
        num_probes=num_probes
    )
"""

# Load in experimental data
"""
experimental_measurements_dict = expdt.experimentalMeasurementDict(
    directory = str("Data/"+global_variables.dataset),
    max_time = global_variables.data_max_time + global_variables.data_time_offset
)
"""

experimental_measurements_dict = pickle.load(
    open(str('Data/'+global_variables.dataset), 'rb')
)

if global_variables.use_experimental_data is True:
    expec_val_plot_max_time = global_variables.data_max_time
else:
    expec_val_plot_max_time = 100    

"""
for t in list(experimental_measurements_dict.keys()):
    # Shift t-values by offset so t=0 corresponds to Pr(0)=1
    # Convert t from ns to ms; remove old records

    new_time = (t - global_variables.data_time_offset)/1000
    msmt = experimental_measurements_dict[t]
    experimental_measurements_dict.pop(t)
    experimental_measurements_dict[new_time] = msmt
"""




initial_op_list = ['xTi', 'yTi', 'zTi']

true_op = global_variables.true_operator
true_op_list = DataBase.get_constituent_names_from_name(true_op)
num_params = len(true_op_list)
true_params = []
random_true_params = False

if(
    global_variables.use_experimental_data==False and
    random_true_params == False
):
    if true_op == 'xTiPPyTiPPzTiPPxTxPPyTyPPzTz':
        for i in range(3):
            true_params.append(random.uniform(0,2)) # for xTi, yTi, zTi
        true_params.append(random.uniform(2.0, 3.0))    #xTx
        true_params.append(random.uniform(2.0, 3.0))    #yTy
        true_params.append(random.uniform(2.0, 3.0))    #zTz
    elif true_op == 'xTi':
        true_params = [1.0]

else: # i.e. purely random true parameters
    for i in range(num_params):
        true_params.append(random.uniform(0,2))

log_print(
    ["True params:", true_params], 
    log_file
)

if global_variables.custom_prior:

    prior_specific_terms = {

        'xTy' : [1.0, 0.5],
        'xTz' : [1.0, 0.5],
        'yTz' : [1.0, 0.5],


        # 'xTx' : [-2.7, 0.1], # true value 2.7
        # 'yTy' : [-2.7, 0.1], # true value 2.7
        # 'zTz' : [-2.1, 0.1], # true value 2.14
        # 'xTi' : [1.5, 1.0], # TODO Broaden, testing with small dist
        # 'yTi' : [1.5, 1.0],
        # 'zTi' : [1.5, 1.0],
        'xTx' : [0.0, 2.0], # true value 2.7
        'yTy' : [0.0, 2.0], # true value 2.7
        'zTz' : [0.0, 2.0], # true value 2.14
        'xTi' : [0.0, 2.0], # TODO Broaden, testing with small dist
        'yTi' : [0.0, 2.0],
        'zTi' : [0.0, 2.0],

        # # Values below correspond to simulated data
        # 'xTx' : [-1.5, 1.0], # true value 2.7
        # 'yTy' : [-1.5, 1.0], # true value 2.7
        # 'zTz' : [-1.5, 1.0], # true value 2.14
        # 'xTi' : [1.5, 1.0], # TODO Broaden, testing with small dist
        # 'yTi' : [1.5, 1.0],
        # 'zTi' : [1.5, 1.0],

        # Values below correspond to Andreas' inital QMD values for this data set
#        'xTx' : [-1.59329862 , 0.19606237], # true value 2.7
#        'yTy' : [-1.8172416  , 0.13556748], # true value 2.7
#        'zTz' : [-3.14796181  ,0.02902302], # true value 2.14
#        'xTi' : [ 0.33645139 , 0.09620414], # TODO Broaden, testing with small dist
#        'yTi' : [ 0.66585614 , 0.07126715],
#        'zTi' : [ 0.34865298  , 0.05232437],


        # Values below correspond to Andreas' final values for the NVB_Hahn data set
#         'xTx' : [-2.85, 0.01], # true value 2.7
#         'yTy' : [-2.76, 0.01], # true value 2.7
#         'zTz' : [-2.13, 0.01], # true value 2.14
#         'xTi' : [0.66, 0.01], # TODO Broaden, testing with small dist
#         'yTi' : [0.43, 0.01],
#         'zTi' : [0.55, 0.01],
    }
else:
    prior_specific_terms = {}

model_priors = None

if global_variables.further_qhl == True:

    qmd_results_model_scores_csv = str(
        global_variables.results_directory + 'average_priors.p'
    )
    print("QMD results CSV in ", qmd_results_model_scores_csv)
    model_priors = pickle.load(
        open(
            qmd_results_model_scores_csv,
            'rb'
        )
    )
    log_print(
        ["Futher QHL. Model_priors:\n", model_priors],
        log_file 
    )
    initial_op_list = list(model_priors.keys())


num_ops = len(initial_op_list)
do_qhl_plots = global_variables.qhl_test and False # TODO when to turn this on?
    
results_directory = global_variables.results_directory
long_id = global_variables.long_id
    
log_print(["\n QMD id", global_variables.qmd_id, 
    " on host ", global_variables.host_name, 
    "and port", global_variables.port_number,
    "has seed", rds.get_seed(global_variables.host_name,
    global_variables.port_number, global_variables.qmd_id,
    print_status=True),"\n", global_variables.num_particles,
    " particles for", global_variables.num_experiments, 
    "experiments and ", global_variables.num_times_bayes,
    "bayes updates\n Gaussian=", global_variables.gaussian, 
    "\n RQ=", global_variables.use_rq, "RQ log:",
    global_variables.log_file, "\n Bayes CSV:",
    global_variables.cumulative_csv], log_file
 )

""" 
Launch and run QMD
"""

qmd = QMD(
    initial_op_list=initial_op_list, 
    qhl_test = global_variables.qhl_test, 
    true_operator=true_op, 
    true_param_list=true_params, 
    use_time_dep_true_model = False, 
    true_params_time_dep = { 'xTi' : 0.01},
    num_particles=global_variables.num_particles,
    num_experiments = global_variables.num_experiments, 
    num_times_for_bayes_updates = global_variables.num_times_bayes,
    bayes_lower = global_variables.bayes_lower,
    bayes_upper = global_variables.bayes_upper,
    qle=qle,
    resample_threshold = global_variables.resample_threshold,
    resampler_a = global_variables.resample_a, 
    pgh_prefactor = global_variables.pgh_factor,
    store_particles_weights = False,
    bayes_time_binning=True, 
    qhl_plots=do_qhl_plots, 
    results_directory = results_directory,
    long_id = long_id, 
    num_probes=num_probes,
    probe_dict = experimental_probe_dict, 
    gaussian = global_variables.gaussian, 
    prior_specific_terms = prior_specific_terms,    
    model_priors = model_priors,
    use_experimental_data = global_variables.use_experimental_data,
    experimental_measurements = experimental_measurements_dict,
    max_num_branches = 0,
    max_num_qubits = 10, 
    parallel = True,
    use_rq = global_variables.use_rq,
    use_exp_custom=True, 
    compare_linalg_exp_tol=None,
    growth_generator = global_variables.growth_generation_rule,
    q_id = global_variables.qmd_id,
    host_name = global_variables.host_name,
    port_number = global_variables.port_number,
    rq_timeout = global_variables.rq_timeout,
    log_file = global_variables.log_file
)

if global_variables.qhl_test:
    qmd.runQHLTest()

    if global_variables.pickle_qmd_class:
        log_print(["QMD complete. Pickling result to",
            global_variables.class_pickle_file], log_file
        )
        qmd.delete_unpicklable_attributes()
        with open(global_variables.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmd, pkl_file , protocol=2)

    if global_variables.save_plots:
        
        print("Plotting things")
        qmd.plotParameterEstimates(true_model=True, 
            save_to_file= str(global_variables.plots_directory+
            'qhl_parameter_estimates_'+ str(global_variables.long_id) +
            '.png')
        )
        """
        qmd.plotExpecValuesQHLTrueModel(
            max_time=expec_val_plot_max_time, 
            t_interval=1,
            save_to_file = str( 
            global_variables.plots_directory+
            'qhl_expec_values_'+str(global_variables.long_id)+'.png')
        )
        """

        qmd.plotVolumeQHL(
            save_to_file = str( 
            global_variables.plots_directory+
            'qhl_volume_'+str(global_variables.long_id)+'.png')
        )

        true_mod_instance = qmd.reducedModelInstanceFromID(
            qmd.TrueOpModelID
        )
        r_squared = (
            true_mod_instance.r_squared()
        )
    qmd.plotExpecValues(
        model_ids = [qmd.TrueOpModelID], # hardcode to see full model for development
        max_time = expec_val_plot_max_time, #in microsec
        t_interval=0.5,
        champ = False,
        save_to_file=str( 
            global_variables.plots_directory +
            'expec_values_' + 
            str(global_variables.long_id)+
            '.png'
        )
    )

    results_file = global_variables.results_file
    pickle.dump(
        qmd.ResultsDict,
        open(results_file, "wb"), 
        protocol=2
    )



elif global_variables.further_qhl == True:

    qmd.runMultipleModelQHL()

    qmd.plotExpecValues(
        model_ids = list(range(qmd.HighestModelID)), # hardcode to see full model for development
        max_time = expec_val_plot_max_time, #in microsec
        champ = False,
        save_to_file=str( 
            global_variables.plots_directory+
            'further_qhl_expec_values.png'
        )
    )

    for mod_id in range(qmd.HighestModelID):
        mod_name = qmd.ModelNameIDs[mod_id]
        qmd.plotParameterEstimates(
            model_id = mod_id, 
            save_to_file= str(global_variables.plots_directory+
            'further_qhl_parameter_estimates_'+ str(mod_name) +
            '.png')
        )

    if global_variables.pickle_qmd_class:
        log_print(["QMD complete. Pickling result to",
            global_variables.class_pickle_file], log_file
        )
        qmd.delete_unpicklable_attributes()
        with open(global_variables.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmd, pkl_file , protocol=2)


else:
    qmd.runRemoteQMD(num_spawns=3) #  Actually run QMD

    """
    Tidy up and analysis. 
    """
    if global_variables.pickle_qmd_class:
        log_print(["QMD complete. Pickling result to",
            global_variables.class_pickle_file], log_file
        )
        qmd.delete_unpicklable_attributes()
        with open(global_variables.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmd, pkl_file , protocol=2)

    # Need to do this so QML reduced class has expectation value
    # dict... should be made unnecessary
    qmd.plotExpecValues(
        model_ids = [11], # hardcode to see full model for development
        max_time = expec_val_plot_max_time, #in microsec
        save_to_file=str( 
        global_variables.plots_directory+
        'expec_values_'+str(global_variables.long_id)+'.png')
    )
    
    if global_variables.save_plots:
        qmd.plotVolumes(
            save_to_file=str(
            global_variables.plots_directory+
            'volumes_all_models_'+ str(global_variables.long_id)+ '.png')
        )
        qmd.plotVolumes(
            branch_champions=True,
            save_to_file=str(global_variables.plots_directory+
            'volumes_branch_champs_'+ str(global_variables.long_id)+
            '.png')
        )
        
        qmd.plotParameterEstimates(
            model_id = qmd.TrueOpModelID, 
            save_to_file= str(global_variables.plots_directory+
            'true_model_parameter_estimates_'+ str(global_variables.long_id) +
            '.png')
        )
        if qmd.ChampID != qmd.TrueOpModelID:
            qmd.plotParameterEstimates(
                model_id = qmd.ChampID, 
                save_to_file= str(global_variables.plots_directory+
                'champ_model_parameter_estimates_'+ str(global_variables.long_id) +
                '.png')
            )
        qmd.plotRadarDiagram(
            save_to_file=str(
            global_variables.plots_directory+
            'radar_'+ str(global_variables.long_id)+ '.png')
        )

        qmd.saveBayesCSV(
            save_to_file=str(
            global_variables.results_directory+ 
            'bayes_factors_'+ str(global_variables.long_id)+'.csv'),
            names_ids='latex'
        )

#        log_print(["Before expec value plot"], log_file)
#        This causes BC to break and nothing after this happens for some reason, so commented out for now (Brian, Aug 16)

    #        qmd.plotHintonAllModels(save_to_file=str(
    #            global_variables.results_directory,'hinton_', 
    #            str(global_variables.long_id), '.png')
    #        )

    #        qmd.plotHintonListModels(model_list=qmd.SurvivingChampions,
    #            save_to_file=str(global_variables.results_directory,
    #            'hinton_champions_', str(global_variables.long_id), 
    #            '.png')
    #        )
        
        qmd.plotTreeDiagram(
            save_to_file = str
            (global_variables.plots_directory+
            'tree_diagram_'+ str(global_variables.long_id)+ '.png')
        )

        qmd.plotRSquaredVsEpoch(
            save_to_file = str(
                global_variables.plots_directory +
                'r_squared_by_epoch_' + str(global_variables.long_id) +
                '.png'
            )
        )


    qmd.writeInterQMDBayesCSV(
        bayes_csv=str(global_variables.cumulative_csv)
    )

    results_file = global_variables.results_file
    pickle.dump(
        qmd.ChampionResultsDict,
        open(results_file, "wb"), 
        protocol=2
    )
            
end = time.time()
log_print(["Time taken:", end-start], log_file)
log_print(["END: QMD id", global_variables.qmd_id, ":",
    global_variables.num_particles, " particles;",
    global_variables.num_experiments, "exp; ", 
    global_variables.num_times_bayes, 
    "bayes. Time:", end-start
    ], 
    log_file
)
    
print("QMD finished - results in:", global_variables.results_directory)

