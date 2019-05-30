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
import Evo
import ExpectationValues
import ModelGeneration 
import UserFunctions
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


# Note this should usually be False, True just for testing/some specific plots. 
store_particles_weights = False

log_file = global_variables.log_file
qle = global_variables.do_qle # True for QLE, False for IQLE
# if global_variables.special_probe == 'plus':
#     num_probes=1
# else:
#     num_probes = 20

# using 40 probes for training - randomly generated
# num_probes = 40

generated_probe_dict = UserFunctions.get_probe_dict(
    experimental_data = global_variables.use_experimental_data, 
    growth_generator = global_variables.growth_generation_rule, 
    special_probe = global_variables.special_probe, 
    noise_level = global_variables.probe_noise_level,
    minimum_tolerable_noise = 0.0,
    # noise_level = 0.0,
    # minimum_tolerable_noise = 1e-7, # to match dec_14/09_55 run # TODO remove!!!
    num_probes = global_variables.num_probes
)

probes_dir = str(
    global_variables.results_directory
    +'training_probes/'
)
if not os.path.exists(probes_dir):
    try:
        os.makedirs(probes_dir)
    except:
        # if already exists (ie created by another QMD since if test ran...)
        pass

training_probes_path = str( 
    probes_dir
    + 'probes_'
    + str(global_variables.long_id)
    +'.p'
)

pickle.dump(
    generated_probe_dict, 
    open(training_probes_path, 'wb')
)

# dataset = UserFunctions.get_experimental_dataset(
#     global_variables.growth_generation_rule
# )

dataset = global_variables.dataset

print("[EXP] For  growth rule {}; use dataset {}".format(
    global_variables.growth_generation_rule, dataset       
    )
)
experimental_measurements_dict = pickle.load(
    open(str('Data/'+dataset), 'rb')
)

num_datapoints_to_plot = 300 # to visualise in expec_val plot for simulated data

if global_variables.use_experimental_data is True:
    expec_val_plot_max_time = max(
        list(experimental_measurements_dict.keys())
    )
    # expec_val_plot_max_time = global_variables.data_max_time
else:
    expec_val_plot_max_time = global_variables.data_max_time   

"""
for t in list(experimental_measurements_dict.keys()):
    # Shift t-values by offset so t=0 corresponds to Pr(0)=1
    # Convert t from ns to ms; remove old records

    new_time = (t - global_variables.data_time_offset)/1000
    msmt = experimental_measurements_dict[t]
    experimental_measurements_dict.pop(t)
    experimental_measurements_dict[new_time] = msmt
"""

plot_lower_time = 0
plot_upper_time = expec_val_plot_max_time
plot_number_times = num_datapoints_to_plot
raw_times = list(np.linspace(
    plot_lower_time, 
    plot_upper_time, 
    plot_number_times+1)
)
plot_times = [ np.round(a, 2 ) for a in raw_times ]
plot_times = sorted(plot_times)
if global_variables.use_experimental_data==True:
    plot_times = sorted(
        list(experimental_measurements_dict.keys())
    )    

initial_op_list  = UserFunctions.get_initial_op_list(
    growth_generator = global_variables.growth_generation_rule,
    log_file = global_variables.log_file
)

if (
    global_variables.use_experimental_data == False
    and 
    global_variables.growth_generation_rule  \
        in UserFunctions.fixed_axis_generators
):
    paulis = ['x', 'y', 'z']
    new_initial_ops = []
    count_paulis = 0 
    for p in paulis:
        if p in global_variables.true_operator:
            core_pauli = p
            for init_op in initial_op_list:
                if core_pauli in init_op:
                    new_initial_ops.append(init_op)

    if len(new_initial_ops) > 1:
        print(
            "For growth rule ", 
            global_variables.growth_generation_rule, 
            "true operator", global_variables.true_opeator, 
            "not valid"
        )
        sys.exit()

    else:
        initial_op_list = new_initial_ops



# true_params_info = pickle.load(
#     open(
#         global_variables.true_params_pickle_file,
#         'rb'
#     )
# )
# true_op = true_params_info['true_op']
# true_params = true_params_info['params_list']
true_op = global_variables.true_operator
true_params = global_variables.true_params
# true_op = global_variables.true_operator

# true_op = UserFunctions.default_true_operators_by_generator[
#     global_variables.growth_generation_rule
# ]
true_num_qubits = DataBase.get_num_qubits(true_op)
true_op_list = DataBase.get_constituent_names_from_name(true_op)
true_op_matrices = [DataBase.compute(t) for t in true_op_list]
num_params = len(true_op_list)


# true_expectation_value_path = str(global_variables.results_directory + 'true_expectation_values.p')
true_expectation_value_path = global_variables.true_expec_path
if os.path.isfile(true_expectation_value_path) == False:
    # true_ham = np.tensordot(
    #     true_params, 
    #     true_op_matrices, 
    #     axes=1
    # )
    # true_ham = None
    # true_params_dict = global_variables.true_params_dict
    # for k in list(true_params_dict.keys()):
    #     param = true_params_dict[k]
    #     mtx = DataBase.compute(k)
    #     if true_ham is not None:
    #         log_print(
    #             [
    #             "[Exp - set true_ham] adding {}*{}:\n{}".format(
    #             np.round(param,2), 
    #             k, 
    #             param*mtx)
    #             ], 
    #             log_file
    #         )
    #         true_ham += param*mtx  
    #     else:
    #         log_print(
    #             [
    #             "[Exp - set true_ham] SETTING {}*{}:\n{}".format(
    #             np.round(param,2), 
    #             k, 
    #             param*mtx)
    #             ], 
    #             log_file
    #         )
    #         true_ham = param*mtx

    true_ham = global_variables.true_hamiltonian
    true_expec_values = {}
    # TODO this probe not always appropriate?
    # probe = np.array([0.5, 0.5, 0.5, 0.5+0j]) # TODO generalise probe - use qmd.PlotProbe
    # probe = ExpectationValues.n_qubit_plus_state(true_num_qubits)
    plot_probe_dict = pickle.load(
        open(global_variables.plot_probe_file, 'rb')
    )
    probe = plot_probe_dict[true_num_qubits]
    # print(
    #     "Generating true expectation values (for plots) with probe", 
    #     probe
    # )

    log_print(
        [
            "for generating true data.",
            "\n\tprobe:\n", repr(probe), 
            "\n\t(with 1-norm:)", np.abs(1-np.linalg.norm(probe)),
            "\n\n\ttrue_ham:\n", repr(true_ham)
        ],
        log_file
    )
    if global_variables.use_experimental_data==True:
        true_expec_values = experimental_measurements_dict
    else:
        log_print(
            [
                "Getting true expectation values (for plotting)"
                "\nTimes computed(len {}): {}\n".format(
                    len(plot_times), plot_times
                ) 
            ],
            log_file
        )
        for t in plot_times:
            try:
                true_expec_values[t] = (
                    # ExpectationValues.expectation_value_wrapper(
                    UserFunctions.expectation_value_wrapper(
                        method=global_variables.measurement_type,
                        ham = true_ham, 
                        t=t, 
                        state=probe,
                        log_file=log_file,
                        log_identifier='[Exp - Getting true expec vals for plotting]'
                    )
                )
            except:
                log_print(
                    [
                        "failure for", 
                        "\ntrue ham:", repr(true_ham), 
                        "\nprobe:", repr(probe),
                        "t=",t
                    ],
                    log_file
                )
                raise


    pickle.dump(
        true_expec_values, 
        open(true_expectation_value_path, 'wb')
    )

else:
    true_expec_values = pickle.load(
        open(true_expectation_value_path, 'rb')   
    )


log_print(
    ["True params:", true_params], 
    log_file
)

if global_variables.custom_prior:
    prior_data = pickle.load(
        open(
            global_variables.prior_pickle_file,
            'rb'
        )
    )
    prior_specific_terms = prior_data['specific_terms']

else:
    prior_specific_terms = {}


log_print(
    [
        "Prior specific terms:", 
        prior_specific_terms
    ], 
    log_file
)


model_priors = None

if global_variables.further_qhl == True:

    qmd_results_model_scores_csv = str(
        global_variables.results_directory 
        + 'average_priors.p'
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
    further_qhl_models = list(model_priors.keys())


num_ops = len(initial_op_list)
# do_qhl_plots = global_variables.qhl_test and False # TODO when to turn this on?
do_qhl_plots = True # testing posterior transition # TODO turn off usually
    
results_directory = global_variables.results_directory
long_id = global_variables.long_id
    
log_print(["\n QMD id", global_variables.qmd_id, 
    " on host ", global_variables.host_name, 
    "and port", global_variables.port_number,
    "has seed", rds.get_seed(global_variables.host_name,
    global_variables.port_number, global_variables.qmd_id,
    print_status=True),
    "\n", global_variables.num_particles,
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


generators = [
    # 'test_changes_to_qmd',
    global_variables.growth_generation_rule,
    # 'non_interacting_ising',
    # 'two_qubit_ising_rotation_hyperfine',
    # 'interacing_nn_ising_fixed_axis'
    # 'deterministic_transverse_ising_nn_fixed_axis'
    # 'heisenberg_nontransverse'
]

generators.extend(
    global_variables.alternative_growth_rules
)

print("All growth rules:", generators)
# generators_from_global_vars = global_variables.

generator_initial_models = {}
for gen in generators:
    try:
        generator_initial_models[gen] = UserFunctions.initial_models[gen]
    except:
        generator_initial_models[gen] = UserFunctions.initial_models[None]


qmd = QMD(
    global_variables = global_variables, 
    initial_op_list=initial_op_list, 
    generator_initial_models = generator_initial_models,
    true_operator=true_op, 
    # true_param_list=true_params, 
    use_time_dep_true_model = False, 
    true_params_time_dep = { 'xTi' : 0.01},
    qle=qle,
    store_particles_weights = store_particles_weights,
    # bayes_time_binning=global_variables.bayes_time_binning, 
    qhl_plots=do_qhl_plots, 
    results_directory = results_directory,
    long_id = long_id, 
    # num_probes=num_probes,
    probe_dict = generated_probe_dict, 
    model_priors = model_priors,
    experimental_measurements = experimental_measurements_dict,
    plot_times = plot_times,
    max_num_branches = 0,
    max_num_qubits = 10, 
    parallel = True,
    use_exp_custom = True, 
    compare_linalg_exp_tol = None,
    prior_specific_terms = prior_specific_terms,    
)

if global_variables.qhl_test:
    qmd.runQHLTest()
    log_print(
        [
            "QHL complete",
        ], 
        log_file
    )
    if global_variables.pickle_qmd_class:
        log_print(
            [
                "QMD complete. Pickling result to",
                global_variables.class_pickle_file
            ], log_file
        )
        qmd.delete_unpicklable_attributes()
        with open(global_variables.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmd, pkl_file , protocol=2)

    if global_variables.save_plots:
        
        print("[Exp.py] Plotting things")

        try:
            log_print(
                [
                    "Plotting parameter estimates",
                ], 
                log_file
            )
            qmd.plotParameterEstimates(
                true_model=True, 
                save_to_file= str(
                    global_variables.plots_directory+
                    'qhl_parameter_estimates_'+ 
                    str(global_variables.long_id) +
                    '.png'
                )
            )
        except: pass

        try:
            log_print(
                [
                    "Plotting volumes",
                ], 
                log_file
            )
            qmd.plotVolumeQHL(
                save_to_file = str( 
                    global_variables.plots_directory+
                    'qhl_volume_'+
                    str(global_variables.long_id)+
                    '.png'
                )
            )
        except: pass
        log_print(
            [
                "Plotting Quadratic Losses",
            ], 
            log_file
        )

        qmd.plotQuadraticLoss(
            save_to_file = str( 
                global_variables.plots_directory+
                'qhl_quadratic_loss_'
                +str(global_variables.long_id)+'.png'
            )
        )

        true_mod_instance = qmd.reducedModelInstanceFromID(
            qmd.TrueOpModelID
        )
        # r_squared = (
        #     true_mod_instance.r_squared()
        # )

    print("plotting expectation values")

    log_print(
        [
            "Plotting Dynamics",
        ], 
        log_file
    )
    qmd.plotDynamics(
        include_bayes_factors_in_dynamics_plots=False, 
        include_param_estimates_in_dynamics_plots=True,
        include_times_learned_in_dynamics_plots=True, 
        save_to_file=str( 
            global_variables.plots_directory +
            'dynamics_' + 
            str(global_variables.long_id)+
            '.png'
        )
    )
    log_print(
        [
            "Finished plotting dynamics",
        ], 
        log_file
    )

    true_mod = qmd.reducedModelInstanceFromID(
        qmd.TrueOpModelID
    )
    extend_dynamics_plot_times = [
        t*2 for t in qmd.PlotTimes  
    ]
    print(
        "[Exp.py - QHL]", 
        "Computing more expectation values"
    )
    true_mod.compute_expectation_values(
        times = extend_dynamics_plot_times
    )  

    qmd.plotDynamics(
        model_ids = [qmd.TrueOpModelID],
        include_bayes_factors_in_dynamics_plots=False, 
        include_param_estimates_in_dynamics_plots=False,
        include_times_learned_in_dynamics_plots=False, 
        save_to_file=str( 
            global_variables.plots_directory +
            'extended_dynamics_' + 
            str(global_variables.long_id)+
            '.png'
        )
    )



#     qmd.plotExpecValues(
#         model_ids = [qmd.TrueOpModelID], # hardcode to see full model for development
#         max_time = expec_val_plot_max_time, #in microsec
#         t_interval=float(expec_val_plot_max_time/num_datapoints_to_plot),
# #        t_interval=0.02,
#         champ = False,
#         save_to_file=str( 
#             global_variables.plots_directory +
#             'expec_values_' + 
#             str(global_variables.long_id)+
#             '.png'
#         )
#     )

    results_file = global_variables.results_file
    pickle.dump(
        qmd.ResultsDict,
        open(results_file, "wb"), 
        protocol=2
    )



elif (
    global_variables.further_qhl == True
    or global_variables.multiQHL == True
):


    if global_variables.multiQHL == True:
        # note models are only for true growth generation rule
        # models to QHL can be declared in 
        # UserFunctions.qhl_models_by_generator dict
        qhl_models = UserFunctions.get_qhl_models(
            global_variables.growth_generation_rule
        )
        
        output_prefix = 'multi_qhl_'

    else:
        qhl_models = further_qhl_models 
        output_prefix = 'further_qhl_'

    print("QHL Models:", qhl_models)    

    qmd.runMultipleModelQHL(
        model_names = qhl_models
    )
    # model_ids = list(range(qmd.HighestModelID))
    model_ids = [
        DataBase.model_id_from_name(
            db=qmd.db, 
            name=mod
        # ) for mod in further_qhl_models
        ) for mod in qhl_models
    ]

    qmd.plotDynamics(
        save_to_file=str( 
            global_variables.plots_directory +
            'dynamics_' + 
            str(global_variables.long_id)+
            '.png'
        )
    )

    # qmd.plotExpecValues(
    #     model_ids = model_ids, # hardcode to see full model for development
    #     times=plot_times,
    #     max_time = expec_val_plot_max_time, #in microsec
    #     t_interval=float(expec_val_plot_max_time/num_datapoints_to_plot),
    #     champ = False,
    #     save_to_file=str( 
    #         global_variables.plots_directory+
    #         output_prefix + 
    #         'expec_values_'+
    #         str(global_variables.long_id) + 
    #         '.png'
    #     )
    # )

    # for mod_id in range(qmd.HighestModelID):
    #     mod_name = qmd.ModelNameIDs[mod_id]
    #     qmd.plotParameterEstimates(
    #         model_id = mod_id, 
    #         save_to_file= str(global_variables.plots_directory+
    #         'further_qhl_parameter_estimates_'+ str(mod_name) +
    #         '.png')
    #     )

    if global_variables.pickle_qmd_class:
        log_print(
            [
            "QMD complete. Pickling result to",
            global_variables.class_pickle_file
            ], 
            log_file
        )
        qmd.delete_unpicklable_attributes()
        with open(global_variables.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmd, pkl_file , protocol=2)

    # results_file = global_variables.results_file
    

    for mid in model_ids:
        mod = qmd.reducedModelInstanceFromID(mid)
        name = mod.Name

        results_file=str(
            global_variables.results_directory + 
            # output_prefix + 
            'results_'+
            str(name)+'_'+
            str(global_variables.long_id)+
            '.p'
        )
        print("[Exp] results file:", results_file)

        pickle.dump(
            mod.results_dict,
            open(results_file, "wb"), 
            protocol=2
        )


else:
    # qmd.runRemoteQMD(num_spawns=3) #  Actually run QMD
    qmd.runRemoteQMD_MULTIPLE_GEN(num_spawns=3) #  Actually run QMD
    print(" \n\n------QMD learned ------\n\n")

    """
    Tidy up and analysis. 
    """

    # Need to do this so QML reduced class has expectation value
    # dict... should be made unnecessary
    # plot_times = np.linspace(
    #     0, 
    #     expec_val_plot_max_time, 
    #     num_datapoints_to_plot
    # )

    expec_value_mods_to_plot = []
    try:    
        expec_value_mods_to_plot = [qmd.TrueOpModelID]
    except:
        pass
    expec_value_mods_to_plot.append(qmd.ChampID)

    print("plotExpecValues")
    # qmd.plotExpecValues(
    #     model_ids = expec_value_mods_to_plot, # hardcode to see full model for development
    #     times=plot_times,
    #     max_time = expec_val_plot_max_time, #in microsec
    #     t_interval=float(expec_val_plot_max_time/num_datapoints_to_plot),
    #     save_to_file=str( 
    #     global_variables.plots_directory+
    #     'expec_values_'+str(global_variables.long_id)+'.png')
    # )
    if global_variables.growth_generation_rule == 'NV_centre_experiment_debug':
        plot_dynamics_all_models = True 
    else:
        plot_dynamics_all_models = False
    qmd.plotDynamics(
        all_models = plot_dynamics_all_models, 
        save_to_file=str( 
            global_variables.plots_directory +
            'dynamics_' + 
            str(global_variables.long_id)+
            '.png'
        )
    )


    champ_mod = qmd.reducedModelInstanceFromID(
        qmd.ChampID
    )
    extend_dynamics_plot_times = [
        t*2 for t in qmd.PlotTimes  
    ]
    print(
        "[Exp.py - QHL]", 
        "Computing more expectation values"
    )
    champ_mod.compute_expectation_values(
        times = extend_dynamics_plot_times
    )  

    qmd.plotDynamics(
        model_ids = [qmd.ChampID],
        include_bayes_factors_in_dynamics_plots=False, 
        include_param_estimates_in_dynamics_plots=False,
        include_times_learned_in_dynamics_plots=False, 
        save_to_file=str( 
            global_variables.plots_directory +
            'extended_dynamics_' + 
            str(global_variables.long_id)+
            '.png'
        )
    )

    if global_variables.save_plots:
        try:
            print("plotVolumes")
            qmd.plotVolumes(
                save_to_file=str(
                global_variables.plots_directory+
                'volumes_all_models_'+ str(global_variables.long_id)+ '.png')
            )
            print("plotExpecValues2")
            qmd.plotVolumes(
                branch_champions=True,
                save_to_file=str(global_variables.plots_directory+
                'volumes_branch_champs_'+ str(global_variables.long_id)+
                '.png')
            )
            print("plotQuadLoss")
            qmd.plotQuadraticLoss(
                save_to_file= str(
                    global_variables.plots_directory+
                    'quadratic_loss_'+ str(global_variables.long_id)+
                    '.png'
                )
            )
        except:
            print("Couldn't plot all individual QMD plots.")

        true_op_known=False
        try:
            if qmd.TrueOpModelID:
                true_op_known = True
        except:
            true_op_known = False

        if true_op_known==True:
        # if QMD has knowledge of the "true" model, then plot params   
            try:
                print("plotParameterEstimates")
                qmd.plotParameterEstimates(
                    model_id = qmd.TrueOpModelID, 
                    save_to_file= str(
                        global_variables.plots_directory+
                        'true_model_parameter_estimates_'+ 
                        str(global_variables.long_id) +
                        '.png'
                    )
                )
            except:
                print(
                    "Failed to plot parameter estimates for true model:", 
                    qmd.TrueOpName
                )

            if qmd.ChampID != qmd.TrueOpModelID:
                try:
                    print("plotParameterEstimates champ id != true id")
                    qmd.plotParameterEstimates(
                        model_id = qmd.ChampID, 
                        save_to_file= str(global_variables.plots_directory+
                        'champ_model_parameter_estimates_'+ str(global_variables.long_id) +
                        '.png')
                    )
                except:
                    print(
                        "Failed to plot parameter estimates for ", 
                        "champ model:", 
                        qmd.ChampID
                    )


        else:
            try:
                    print("else plotParameterEstimates")
                    qmd.plotParameterEstimates(
                        model_id = qmd.ChampID, 
                        save_to_file= str(global_variables.plots_directory+
                        'champ_model_parameter_estimates_'+ str(global_variables.long_id) +
                        '.png')
                    )
            except:
                print(
                    "Failed to plot parameter estimates for champ model:", 
                    qmd.ChampID
                )


        """
        # TODO radar plot not working - when used with RQ ???
        # not finding TrueModelID when using ising_hyperfine generation rule 
        qmd.plotRadarDiagram(
            save_to_file=str(
            global_variables.plots_directory+
            'radar_'+ str(global_variables.long_id)+ '.png')
        )
        """
        print("saveBayesCSV")

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
        # TODO generalise so tree diagram can be used in all cases
        # currently only useful for Ising growth 2 qubits. 
        try:
            print("plotTreeDiagram")
            qmd.plotTreeDiagram(
                only_adjacent_branhces=False, 
                save_to_file = str
                (global_variables.plots_directory+
                'tree_diagram_' + 
                str(global_variables.long_id) + 
                '.png')
            )
        except:
            pass

        try:
            print("plotRSquaredVsEpoch")
            qmd.plotRSquaredVsEpoch(
                save_to_file = str(
                    global_variables.plots_directory +
                    'r_squared_by_epoch_' + str(global_variables.long_id) +
                    '.png'
                )
            )
        except:
            log_print(
                [
                "Failed to plot R squared vs epoch.", 
                "Probably a problem caused by introducing rescaling",
                "resources based on num qubits etc"
                ],
                log_file
            )

    if global_variables.pickle_qmd_class:
        log_print(["QMD complete. Pickling result to",
            global_variables.class_pickle_file], log_file
        )
        qmd.delete_unpicklable_attributes()
        with open(global_variables.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmd, pkl_file , protocol=2)

    # TODO generalise so tree diagram can be used in all cases
    # currently only useful for Ising growth 2 qubits. 
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

