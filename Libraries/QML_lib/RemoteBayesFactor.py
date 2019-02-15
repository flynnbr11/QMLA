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
    use_all_exp_times_for_bayes_factors = False # TODO make this a QMD input



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

        update_times_model_a = model_b.Times[first_t_idx:]
        update_times_model_b = model_a.Times[first_t_idx:]
        set_renorm_record_to_zero = False


        # if (
        #     num_times_to_use == 'all' 
        #     or 
        #     len(model_a.Times) < num_times_to_use
        # ):
        #     times_a = model_a.Times
        #     times_b = model_b.Times
        # else:
        #     times_a = model_a.Times[first_t_idx:]
        #     times_b = model_b.Times[first_t_idx:]
        
        # # if binning==True and use_experimental_data==True:

        # log_print(
        #     [
        #     "Binning. Before times\n A:", repr(times_a), 
        #     "\nB:", repr(times_b)
        #     ]
        # )

        
        if (
            use_experimental_data is True
            and
            use_all_exp_times_for_bayes_factors is True
        ):
            experimental_data_times = info_dict[
                'experimental_measurement_times'
            ]
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
            time_list = set( model_a.Times + model_b.Times)

            all_times = np.concatenate(
                [
                    model_a.Times, 
                    model_b.Times
                ]
            )

            num_unique_times = len(np.unique(
                all_times
            ))  

            binned_times = balance_times_by_binning(
                data = all_times, 
                all_times_used = True, 
                num_bins = num_unique_times
            )
            update_times_model_a = list(binned_times)
            update_times_model_b = list(binned_times)
            set_renorm_record_to_zero = True

        # elif binning==True:
        #     import ExperimentalDataFunctions

        #     # TODO introduce binning for simulated data. 
        #     try:
        #         min_time = min(min(times_a), min(times_b))
        #         max_time = max(max(times_a), max(times_b))
        #     except:
        #         log_print(
        #             [
        #             "Can't find min/max of:", 
        #             "\ntimes_a:", times_a, 
        #             "\ntimes_b:", times_b,
        #             "\n Model IDs:", model_a_id, 
        #             ";\t", model_b_id,
        #             "\nmoda.Times:", model_a.Times,
        #             "\nmodb.Times:", model_b.Times,
        #             ]
        #         )


        #     times_list = np.linspace(
        #         min_time,
        #         max_time, 
        #         2*num_times_to_use # learning from scratch so need twice the number of times.
        #     )


        #     if use_experimental_data == True:


        #         all_times = [
        #             ExperimentalDataFunctions.nearestAvailableExpTime(
        #                 times = experimental_data_times,
        #                 t=t
        #             ) 
        #             for t in times_list
        #         ]
        #         times_a = all_times
        #         times_b = all_times

        #     else:
        #         times_a = times_list
        #         times_b = times_list            


        # log_print(
        #     [
        #     ".After \n A:", times_a, 
        #     "\nB:", times_b
        #     ]
        # )

        # log_print(["Computing log likelihoods."])
        #print("Num times to use", num_times_to_use)
        #print("Model", model_a.ModelID, " Times:", times_a)
        #print("Model", model_b.ModelID, " Times:", times_b)

        ## shouldn't have to do this -- remove when processing func doesn't return times not available experimentally

        update_times_model_a = sorted(update_times_model_a)
        update_times_model_b = sorted(update_times_model_b)

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

        if use_experimental_data is True:
            experimental_data_times = info_dict[
                'experimental_measurement_times'
            ]

            all_times_in_exp_data = np.all(
                [
                    d in experimental_data_times
                    for d in binned_times
                ]
            )
            all_avail = np.all(
                [
                    d in experimental_data_times
                    for d in binned_times
                ]
            )
            if all_avail is False:
                log_print(
                    [
                        "all NOT experimentally available."
                    ]
                )

            num_times_to_print = min(len(update_times_model_a), 5)
            # log_print(
            #     [
            #         "Binning:", binning,
            #         "\n\t", model_a.Name, 
            #         "\n\tInitial \n\t", repr(model_a.Times[1:num_times_to_print]),
            #         "\n\tUpdate (len ", len(update_times_model_a), ")", 
            #         "\n\t", repr(update_times_model_a[1:num_times_to_print]),
            #         "\n\t", model_b.Name, 
            #         "\n\tInitial \n\t", repr(model_b.Times[1:num_times_to_print]), 
            #         "\n\tUpdate (len ", len(update_times_model_b), ")", 
            #         "\n\t", repr(update_times_model_b[1:num_times_to_print])
            #     ]
            # )
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

        bayes_factor = np.exp(log_l_a - log_l_b)

        log_print(
            [
                "BF computed: A:{}; B:{}; BF:{}".format(
                        model_a_id, 
                        model_b_id, 
                        np.round(bayes_factor, 2)
                    )
            ]
        )
        if bayes_factor < 1e-160:
            bayes_factor = 1e-160
        
        pair_id = DataBase.unique_model_pair_identifier(
            model_a_id, model_b_id
        )
        print("Bayes Factor:", pair_id)
        if float(model_a_id) < float(model_b_id):
            # so that BF in db always refers to (a/b), not (b/a). 
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
        # print("Datum")
        datum = updater.model.simulate_experiment(
            params_array, 
            exp, 
            repeat=1
        )
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