from __future__ import print_function # so print doesn't show brackets
# Libraries
import numpy as np
import itertools as itr
import os as os
import sys as sys 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time as time
import random
import ExperimentalDataFunctions
from psutil import virtual_memory
import json ## possibly worth a different serialization if pickle is very slow
import pickle
pickle.HIGHEST_PROTOCOL=2
import copy

try:
    import RedisSettings as rds
    import redis
    enforce_serial = False  
except:
    enforce_serial = True # shouldn't be needed
      
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Local files
# import Evo as evo
import DataBase 
import QML
# import ModelGeneration
#import BayesF
from qinfer import NormalDistribution
# from Distrib import MultiVariateNormalDistributionNocov

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    
    return time

## Single function call to compute Bayes Factor between models given their IDs

def BayesFactorRemote(
    model_a_id, 
    model_b_id, 
    branchID=None, 
    interbranch=False, 
    num_times_to_use='all', 
    bf_data_folder=None, 
    times_record='BayesFactorsTimes.txt', 
    check_db=False, 
    trueModel=None, 
    bayes_threshold=1, 
    host_name='localhost', 
    port_number=6379, 
    qid=0, 
    log_file='rq_output.log'
):
    """
    This is a standalone function to compute Bayes factors without knowledge 
    of full QMD program. QMD info is unpickled from a redis databse, containing
    learned_model information, i.e. final parameters etc. 
    Given model ids correspond to model names in the database, which are combined 
    with the final learned parameters to reconstruct model classes of 
    complete learned models. 
    From these we extract log likelihoods to compute Bayes factors. 
    
    """

    write_log_file = open(log_file, 'a')
    def log_print(to_print_list):
        identifier = str(str(time_seconds()) + " [RQ Bayes "
            +str(model_a_id)+"/"+str(model_b_id)+"]"
        )
        if type(to_print_list)!=list:
            to_print_list = list(to_print_list)
        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file, flush=True)

    time_start = time.time()
    rds_dbs = rds.databases_from_qmd_id(host_name, port_number, qid)
    qmd_info_db = rds_dbs['qmd_info_db'] 
    learned_models_info = rds_dbs['learned_models_info']
    learned_models_ids = rds_dbs['learned_models_ids']
    bayes_factors_db = rds_dbs['bayes_factors_db']
    bayes_factors_winners_db = rds_dbs['bayes_factors_winners_db']
    active_branches_learning_models = rds_dbs['active_branches_learning_models']
    active_branches_bayes = rds_dbs['active_branches_bayes']
    active_interbranch_bayes = rds_dbs['active_interbranch_bayes']
    
    info_dict = pickle.loads(rds_dbs['qmd_info_db']['QMDInfo'])
    use_experimental_data = info_dict['use_experimental_data']
    experimental_data_times = info_dict['experimental_measurement_times']
    binning = info_dict['bayes_factors_time_binning']
    # use_all_exmodel_id_strp_times_for_bayes_factors = False # TODO make this a QMD input
    use_all_exp_times_for_bayes_factors = info_dict['bayes_factors_time_all_exp_times'] # TODO make this a QMD input
    linspace_times_for_bayes_factor_comparison = False
    use_opponent_learned_times = True
    true_mod_name = info_dict['true_name']
    save_plots_of_posteriors = False
    plot_true_mod_post_bayes_factor_dynamics = False

    if check_db: # built in to only compute once and always return the stored value.
        if pair_id in bayes_factors_db.keys():
            bayes_factor = bayes_factors_db.get(pair_id)
            log_print(["Redis GET bayes_factors_db pair:", pair_id])
            if model_a_id < model_b_id:
                return bayes_factor
            else:
                return (1.0/bayes_factor)
    else:
        
        model_a = QML.modelClassForRemoteBayesFactor(
            modelID=model_a_id,
            host_name=host_name, 
            port_number=port_number, 
            qid=qid, 
            log_file=log_file
        )
        model_b = QML.modelClassForRemoteBayesFactor(
            modelID=model_b_id,
            host_name=host_name, 
            port_number=port_number, 
            qid=qid, 
            log_file=log_file
        )

        # By default, use times the other model trained on, 
        # up to t_idx given. 
        # And inherit renormalization record from QML updater
        # In special cases, eg experimental data, change these defaults below.
        log_print(["Start. Branch", branchID])
        if num_times_to_use == 'all':
            first_t_idx = 0
        else:
            first_t_idx = len(model_a.Times) - num_times_to_use

        if use_opponent_learned_times == True:
            # by default, just use the times learned by the other model
            update_times_model_a = model_b.Times[first_t_idx:]
            update_times_model_b = model_a.Times[first_t_idx:]
            set_renorm_record_to_zero = False
        
        elif (
            use_experimental_data == True
            and
            use_all_exp_times_for_bayes_factors == True
        ):
            experimental_data_times = info_dict[
                'experimental_measurement_times'
            ]
            num_times_learned_over = max(
                len(model_a.Times), 
                len(model_b.Times)
            )
            if len(experimental_data_times) > num_times_learned_over:
                experimental_data_times = experimental_data_times[:num_times_learned_over]


            update_times_model_a = copy.copy(
                experimental_data_times
            )
            update_times_model_b = copy.copy(
                experimental_data_times
            )
        
            log_print(
                [
                    "Using all exp data times for Bayes factors."                
                ]
            )
            set_renorm_record_to_zero = True

        elif (
            binning == True
        ):
            all_times = np.concatenate(
                [
                    model_a.Times, 
                    model_b.Times
                ]
            )

            # num_unique_times = len(np.unique(
            #     all_times
            # ))  

            # binned_times = balance_times_by_binning(
            #     data = all_times, 
            #     all_times_used = True, 
            #     num_bins = num_unique_times
            # )
            # update_times_model_a = list(binned_times)
            # update_times_model_b = list(binned_times)


            # reusing old method (as used in successful Dec_14/09_55 run)
            # where times are linspaced, then mapped to experimental times
            # use 2*num_times_to_use since renormalisation record is set to zero
            max_time = max(all_times)
            min_time = min(all_times)
            linspaced_times = np.linspace(
                min_time, 
                max_time, 
                2*num_times_to_use
            )
            if use_experimental_data==True:
                mapped_to_exp_times = [
                    ExperimentalDataFunctions.nearestAvailableExpTime(
                        times = experimental_data_times,
                        t = t
                    ) for t in linspaced_times
                ]
                update_times_model_a = list(sorted(mapped_to_exp_times))
                update_times_model_b = list(sorted(mapped_to_exp_times))
            elif linspace_times_for_bayes_factor_comparison == True:
                update_times_model_a = linspaced_times
                update_times_model_b = linspaced_times

            set_renorm_record_to_zero = True

        update_times_model_a = sorted(
            update_times_model_a
        )
        update_times_model_b = sorted(
            update_times_model_b
        )


        ### Verbatim code used in december version, though functionally equivalent to elif binning case above
        recreate_dec_13_times_list_method = True
        if (
            use_experimental_data == True
            and 
            recreate_dec_13_times_list_method == True
        ):
            # print("[BF calculation] Generating times as in Dec")
            try:
                min_time = min(min(model_a.Times), min(model_b.Times))
                max_time = max(max(model_a.Times), max(model_b.Times))
            except:
                log_print(
                    [
                    "Can't find min/max of:", 
                    "\ntimes_a:", times_a, 
                    "\ntimes_b:", times_b,
                    "\n Model IDs:", model_a_id, 
                    ";\t", model_b_id,
                    "\nmoda.Times:", model_a.Times,
                    "\nmodb.Times:", model_b.Times,
                    ]
                )
                raise


            times_list = np.linspace(
                min_time,
                max_time, 
                2*num_times_to_use # learning from scratch so need twice the number of times.
            )

            all_times = [
                ExperimentalDataFunctions.nearestAvailableExpTime(
                    times = experimental_data_times,
                    t=t
                ) 
                for t in times_list
            ]
            update_times_model_a = sorted(all_times)
            update_times_model_b = sorted(all_times)


        with open(times_record, 'a') as write_log_file:
            np.set_printoptions(
                precision=2
            )
            print(
                "\n\nModels {}/{}".format(
                    model_a.ModelID,
                    model_b.ModelID
                ), 
                "\n\tID_A {} [len {}]: {}".format(
                    model_a.ModelID, 
                    len(update_times_model_a),
                    [np.round(val, 2) for val in update_times_model_a]
                ),
                "\n\tID_B {} [len {}]: {}".format(
                    model_b.ModelID, 
                    len(update_times_model_b),
                    [np.round(val, 2) for val in update_times_model_b]
                ),
                file = write_log_file
            )

        updater_a_copy = copy.deepcopy(model_a.Updater)
        # updater_b_copy = copy.deepcopy(model_b.updater)

        log_l_a = log_likelihood(
            model_a, 
            update_times_model_a, 
            binning=set_renorm_record_to_zero
        )
        log_l_b = log_likelihood(
            model_b, 
            update_times_model_b, 
            binning=set_renorm_record_to_zero
        )     
        # log_print(["Log likelihoods computed."])

        # after learning, want to see what dynamics are like after further updaters
        bayes_factor = np.exp(log_l_a - log_l_b)


        if (
            save_plots_of_posteriors == True
            and
            DataBase.alph(model_a.Name) == DataBase.alph(true_mod_name)
        ):


            try:
                print("\n\nBF UPDATE Model {}".format(model_a.Name))               
                plot_posterior_marginals(
                    model_a = model_a, 
                    qid = qid,
                    updater_a_copy = updater_a_copy, 
                    bf_data_folder = bf_data_folder
                )
            except:
                raise

        # TODO this is the original position of getting log likelihood + bayes factor; 
        # moving above so we can plot posterior before and after BF updates.             
        # log_l_a = log_likelihood(
        #     model_a, 
        #     update_times_model_a, 
        #     binning=set_renorm_record_to_zero
        # )
        # log_l_b = log_likelihood(
        #     model_b, 
        #     update_times_model_b, 
        #     binning=set_renorm_record_to_zero
        # )     
        # # log_print(["Log likelihoods computed."])

        # # after learning, want to see what dynamics are like after further updaters
        # bayes_factor = np.exp(log_l_a - log_l_b)
        if (
            (
                DataBase.alph(model_a.Name) == DataBase.alph(true_mod_name)
                or 
                DataBase.alph(model_b.Name) == DataBase.alph(true_mod_name)
            )
            and
            plot_true_mod_post_bayes_factor_dynamics == True
        ):
            try:
                plot_path = str(
                    bf_data_folder + 
                    '/BF_{}__{}_{}.png'.format(
                        str(qid), 
                        str(model_a.ModelID), 
                        str(model_b.ModelID)
                    )
                )
                plot_expec_vals_of_models(
                    model_a, 
                    model_b, 
                    bayes_factor = bayes_factor, 
                    bf_times = update_times_model_a,
                    save_to_file = plot_path
                )
            except:
                raise
                # pass




        log_print(
            [
                "BF computed: A:{}; B:{}; BF:{}".format(
                        model_a_id, 
                        model_b_id, 
                        np.round(bayes_factor, 2)
                ),
                "\tReset remormalisation record:", 
                set_renorm_record_to_zero
            ]
        )
        if bayes_factor < 1e-160:
            bayes_factor = 1e-160
        
        pair_id = DataBase.unique_model_pair_identifier(
            model_a_id, model_b_id
        )
        print("Bayes Factor:", pair_id)
        if float(model_a_id) < float(model_b_id):
            # so that BF in db always refers to (low/high), not (high/low). 
            bayes_factors_db.set(pair_id, bayes_factor)
            log_print(
                [
                    "Redis SET bayes_factors_db, pair:", 
                    pair_id,
                    "bayes:",
                    bayes_factor
                ]
            )
        else:
            bayes_factors_db.set(pair_id, (1.0/bayes_factor))
            log_print(
                [
                    "Redis SET bayes_factors_db, pair:", 
                    pair_id, 
                    "bayes:", 
                    (1.0/bayes_factor)
                ]
            )

        if bayes_factor > bayes_threshold: 
            bayes_factors_winners_db.set(pair_id, 'a')
            log_print(
                [
                    "Redis SET bayes_factors_winners_db, pair:", 
                    pair_id, 
                    "winner:", 
                    model_a_id
                ]
            )
        elif bayes_factor < (1.0/bayes_threshold):
            bayes_factors_winners_db.set(pair_id, 'b')
            log_print(
                [
                    "Redis SET bayes_factors_winners_db, pair:", 
                    pair_id, 
                    "winner:", 
                    model_b_id
                ]
            )
        else:
            log_print(["Neither model much better."])

        
        if branchID is not None:    
            # only want to fill these lists when comparing models within branch
            # log_print(["Redis INCR active_branches_bayes branch:", branchID])
            active_branches_bayes.incr(int(branchID), 1)  
        else:
            active_interbranch_bayes.set(pair_id, True)
            # log_print(["Redis SET active_interbranch_bayes pair:", pair_id, 
            #     "; set:True"]
            # )
        time_end = time.time()
        log_print(
            [
            "Finished. rq time: ", 
            str(time_end-time_start)
            ]
        )
    
        return bayes_factor
    
    
def log_likelihood(model, times, binning=False):
    updater = model.Updater


    # print(
    #     "\n[log likel] Log likelihood for model", model.Name
    # )
    # sum_data = 0
    #print("log likelihood function. Model", model.ModelID, "\n Times:", times)


    if binning:
        updater._renormalization_record = []
        updater.log_likelihood = 0 
        # print("BINNING")
    # else:
    #     print("NOT BINNING")    

    for i in range(len(times)):
        exp = get_exp(model, [times[i]])
    #    print("exp:", exp)
        params_array = np.array([[model.TrueParams[:]]]) # TODO this will cause an error for multiple parameters
        
        # print(
        #     "log likelihood", model.Name, 
        #     "\n\ttime:", times[i], 
        #     "\n\tModel.TrueParams:", model.TrueParams,
        #     "\n\tparams array:", params_array, 
        #     "\n\texp:", exp
        # )
        # print("Getting Datum")
        datum = updater.model.simulate_experiment(
            params_array, 
            exp, 
            repeat=1
        )
        # datum = model.GenSimModel.simulate_experiment(
        #     params_array, 
        #     exp, 
        #     repeat=1
        # )
        """
        why updater.model ???
         this is non standard, compare e.g. with QML_lib/QML.py
         >>>> self.Datum = self.GenSimModel.simulate_experiment
        """
        # sum_data += datum   
        # print("Upater")
        updater.update(datum, exp)


    log_likelihood = updater.log_total_likelihood
    return log_likelihood        

def get_exp(model, time):
    gen = model.Updater.model # or gen=model.GenSimModel
    exp = np.empty(len(time), dtype=gen.expparams_dtype)
    exp['t'] = time

    for i in range(1, len(gen.expparams_dtype)):
        col_name = 'w_'+str(i)
        exp[col_name] = model.FinalParams[i-1,0] 
    return exp



######### 
# Functions for rescaling times to be used during Bayes factor calculation
#########

def balance_times_by_binning(
    data, 
    num_bins, 
    all_times_used=False, 
    fraction_times_to_use=0.5, 
    log_base=2
):

    bins = np.linspace(min(data), max(data), num_bins)
    if all_times_used == True:
        bins = np.array(sorted(np.unique(data))) # all exp data points become a bin
    # bins -= 0.00000001
    # print("Bins:", bins, "\nTimes:", sorted(data))
    digitized = np.digitize(data, bins)
    bincount = np.bincount(digitized)

    # scaling by log to remove exponential preference observed for some times
    # remove error cases where log will cause errors (ie log(1) and log(0))
    bincount[np.where( bincount == 1)] = log_base # ratio goes -> 1, bins overrepresented but not critically
    bincount[np.where( bincount == 0)] = 1 # so ratio goes -> 0 and bins don't count
    
    log_bincount = np.log(bincount)/np.log(log_base)
    log_bincount[ np.where( log_bincount == -np.inf ) ] = 0
    log_bincount[ np.where( log_bincount == -np.nan ) ] = 0
    ratio = [ int(np.ceil(i)) for i in log_bincount ]
    
    sum_ratio = np.sum(ratio) 
    median_bincount = np.median( bincount[np.where(bincount!=0)] )
    mean_bincount = np.mean( bincount[ np.where(bincount != 0 ) ] )
    # nonzero_bincounts = bincount[ np.where(bincount != 0 ) ]
    # base_num_elements_per_bin = int(mean_bincount)
    # base_num_elements_per_bin = int(
    #     np.average(bincount, weights=ratio)
    # )
    base_num_elements_per_bin = int(len(data)/sum_ratio)
    # print("sum ratio: ", sum_ratio)    
    # print("base num elements before", base_num_elements_per_bin )
    # print("frac to use:", fraction_times_to_use)

    base_num_elements_per_bin = max(
        1, 
        int(fraction_times_to_use * base_num_elements_per_bin)
    )
    # print("base num elements after", base_num_elements_per_bin )
    newdata = []

    for binid in range(1, len(bincount)):
    # for binid in range(len(bincount)):
        num_elements_in_this_bin = bincount[binid] 
        num_element_to_return_this_bin = base_num_elements_per_bin * ratio[binid]
        if num_elements_in_this_bin > 0:
            multiples = np.where(digitized == binid) # indices of data array which fit in this bin
            for i in range(num_element_to_return_this_bin):
                single = np.random.choice(multiples[0]) # a single random data array index in this bin
                newdata.append(data[single])
    newdata = np.array(newdata)

    return newdata



def plot_expec_vals_of_models(
    model_a, 
    model_b, 
    bayes_factor,
    bf_times, 
    save_to_file=None
):
    import UserFunctions

    exp_msmts = model_a.ExperimentalMeasurements
    times = list(sorted(exp_msmts.keys()))
    experimental_exp_vals = [
        exp_msmts[t] for t in times
    ]
    # plt.clf()
    fig, ax1 = plt.subplots()
    # ax1 = plt.plot()

    ax1.set_ylabel('Exp Val')
    ax1.scatter(
        times, 
        experimental_exp_vals, 
        label='Exp data',
        color='red',
        alpha=0.6,
        s=5
    )
    # plt.legend()
    plot_probes = pickle.load(
        open(model_a.PlotProbePath, 'rb')
    )

    for mod in [model_a, model_b]:
        # TODO get final learned params, 
        # generate hamiltonian from mod.SimOpList
        # get exp vals against times list
        # plot using different colour

        final_params = mod.Updater.est_mean()
        final_ham = np.tensordot(
            final_params, 
            mod.SimOpList, 
            axes=1
        )
        dim = int(np.log2(np.shape(final_ham)[0]))
        plot_probe = plot_probes[dim]

        mod_exp_vals = []
        for t in times:
            # print("Getting exp for t=", t)
            try:
                # todo get this from GenSimModel of mod instead of instantiating class every time
                exp_val = mod.GrowthClass.expectation_value(
                    ham = final_ham, 
                    t = t, 
                    state = plot_probe
                )
            except: 
                exp_val = UserFunctions.expectation_value_wrapper(
                    method = mod.MeasurementType,
                    ham = final_ham, 
                    t = t, 
                    state = plot_probe
                )
            mod_exp_vals.append(exp_val)
            # mod_exp_vals.append(t)
            # print("exp val found for t={}:{}".format(t, exp_val))
        ax1.plot(
            times, 
            mod_exp_vals, 
            label=str(mod.ModelID)
        )
    ax2 = ax1.twinx()

    num_times = int(len(times)) - 1 
    ax2.hist(
        bf_times, 
        bins = num_times, 
        # histtype='bar',
        histtype='stepfilled',
        fill=False,
        color='black',
        label = str( "{} times total".format(len(bf_times))),
        alpha=0.3
    )

    ax2.set_ylabel('Frequency of time updated')
    bf = np.log10(bayes_factor)

    plt.title(
        "[$log_{10}$ Bayes Factor]: " + str(np.round(bf , 2))
    )
    # ax1.legend()
    # ax2.legend()
    plt.figlegend()

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')    


def plot_posterior_marginals(
    model_a,
    updater_a_copy,
    qid, 
    bf_data_folder
):
    old_post_marg = model_a.PosteriorMarginal
    before_bf_updates = []
    new_post_marg = []

    posterior_plot_path = str(
        bf_data_folder + 
        '/posterior_marginal_pickle_{}_{}.p'.format(
            str(qid), 
            str(model_a.ModelID), 
        )
    )

    pickle.dump(
        model_a.Updater,
        open(
            posterior_plot_path, 
            'wb'    
        )
    )

    for i in range(len(old_post_marg)):
        before_bf_updates.append(
            updater_a_copy.posterior_marginal(idx_param=i)
        )                    

        new_post_marg.append(
            model_a.Updater.posterior_marginal(idx_param=i)
        )
        # new_post_marg = model_a.Updater.posterior_marginal(1)
    
    # print("OLD:", old_post_marg)
    # print("NEW:", new_post_marg)
    param_of_interest = 1
    
    for param_of_interest in range(len(old_post_marg)):
        posterior_plot_path = str(
            bf_data_folder + 
            '/posterior_marginal_{}_mod{}_param{}.png'.format(
                str(qid), 
                str(model_a.ModelID), 
                str(param_of_interest)
            )
        )
        if  os.path.exists(posterior_plot_path):
            # ie a previous BF calculation has drawn these for the true model
            break
        plt.clf()
        plt.plot(
            before_bf_updates[param_of_interest][0], 
            before_bf_updates[param_of_interest][1],
            color='blue',
            linestyle = '--',  
            label='Start BF'

        )
        plt.plot(
            new_post_marg[param_of_interest][0], 
            new_post_marg[param_of_interest][1],
            color='y',
            linestyle = '--',  
            label='End BF'

        )
        init_post_marg = model_a.InitialPrior[param_of_interest]
        plt.plot(
            init_post_marg[0], 
            init_post_marg[1],
            color='green', 
            label='Start QML',
            alpha=0.4,
        )
        plt.plot(
            old_post_marg[param_of_interest][0], 
            old_post_marg[param_of_interest][1],
            color='red', 
            linestyle='-.',
            label='End QML'
        )
        plt.legend()

        plt.savefig(posterior_plot_path)
