from __future__ import print_function # so print doesn't show brackets
# Libraries
import numpy as np
import itertools as itr
import os as os
import sys as sys 
import pandas as pd
import warnings
import time as time
from time import sleep
import random
from psutil import virtual_memory
 # only want to do this once at the start!
import pickle 
pickle.HIGHEST_PROTOCOL=2 # TODO if >python3, can use higher protocol
import json
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math

import redis
# import from RedisSettings only when env variables set.
import RedisSettings as rds

# Local files
import Evo as evo
import DataBase 
import QML
import ModelGeneration
import BayesF
import PlotQMD 
from RemoteModelLearning import *
from RemoteBayesFactor import * 
# Class definition
#from RQ_config import *

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time




class QMD():
    #TODO: rename as ModelsDevelopmentClass when finished
    def __init__(self,
                 initial_op_list=['x'],
                 true_operator='x',
                 true_param_list = None,
                 num_particles= 300,
                 num_experiments = 50,
                 max_num_models=30, 
                 max_num_qubits=7, #TODO change -- this may cause crashes somewhere
                 gaussian=True,
                 resample_threshold = 0.5,
                 resampler_a = 0.95,
                 pgh_prefactor = 1.0,
                 num_probes = 20, 
                 num_times_for_bayes_updates = 'all',
                 max_num_layers = 10,
                 max_num_branches = 20, 
                 use_exp_custom = True,
                 enable_sparse = True,
                 compare_linalg_exp_tol = None,
                 sigma_threshold = 1e-13, 
                 debug_directory = None,
                 qle = True, # Set to False for IQLE
                 parallel = False,
                 q_id = 0, # id for QMD instance to keep concurrent QMDs distinct on cluster
                 host_name='localhost',
                 port_number = 6379,
                 use_rq=True, 
                 rq_timeout=3600,
                 growth_generator='simple_ising',
                 log_file = None
                ):
        self.StartingTime = time.time()
        self.QLE = qle # Set to False for IQLE
        trueOp = DataBase.operator(true_operator)
        self.TrueOpName = true_operator
        self.TrueOpDim = trueOp.num_qubits
        self.InitialOpList = initial_op_list
        self.TrueOpList = trueOp.constituents_operators
        self.TrueOpNumParams = trueOp.num_constituents
        if true_param_list is not None: 
            self.TrueParamsList = true_param_list
        else:
            print("No parameters passed, randomising")
            self.TrueParamsList = [random.random() for i in self.TrueOpList] # TODO: actual true params?
        # todo set true parmams properly
        #self.TrueParamsList = [0.75 for i in self.TrueOpList] # TODO: actual true params?
        self.MaxModNum = max_num_models #TODO: necessary?
        self.gaussian = gaussian
        self.NumModels = len(initial_op_list)
        self.NumParticles = num_particles
        self.NumExperiments = num_experiments
        self.MaxQubitNumber = max_num_qubits
        self.NumTimesForBayesUpdates = num_times_for_bayes_updates
        self.ResampleThreshold = resample_threshold
        self.ResamplerA = resampler_a
        self.PGHPrefactor = pgh_prefactor
        self.NumProbes = num_probes
        self.ProbeDict = separable_probe_dict(max_num_qubits=self.MaxQubitNumber, num_probes=self.NumProbes)
        self.HighestQubitNumber = int(0)
        self.MaxBranchID = max_num_branches
        self.HighestBranchID = 0
        self.HighestModelID = len(initial_op_list)
        self.MaxLayerNumber = max_num_layers
        self.BranchChampions = {}
        self.LayerChampions = {}
        self.BayesPointsByBranch ={}
        self.BranchRankings = {}
        self.BranchBayesComputed = {}
        self.InterBranchChampions = {}
        self.GlobalEpoch = 0 
        self.UseExpCustom = use_exp_custom
        self.EnableSparse = enable_sparse
        self.ExpComparisonTol = compare_linalg_exp_tol
        self.SigmaThreshold = sigma_threshold
        self.DebugDirectory = debug_directory
        self.ModelPointsDict = {}
        self.AllBayesFactors = {}
        self.BranchBayesComputed[0] = False
        self.BayesFactorsComputed = []
#        for i in range(self.MaxBranchID+1):
#            self.BranchChampions[i] = 0
#        for i in range(self.MaxLayerNumber+1):
#            self.LayerChampions[i] = 0
        self.ModelNameIDs = {}
        self.GrowthGenerator = growth_generator
        self.SpawnDepth = 0
        self.NumModelsPerBranch = {0:len(self.InitialOpList)}
        self.NumModelPairsPerBranch = {0 : num_pairs_in_list(len(self.InitialOpList))}
        self.BranchAllModelsLearned = { 0 : False}
        self.BranchComparisonsComplete = {0 : False}
        self.Q_id = q_id
        self.HostName = host_name
        self.PortNumber = port_number
        self.use_rq = use_rq
        self.rq_timeout = rq_timeout
        if log_file is not None:
            self.log_file = log_file
        else:
            self.log_file = str('QMD_'+str(q_id)+'.log')
        self.rq_log_file = self.log_file


        self.write_log_file = open(self.log_file, 'a')
        self.MaxSpawnDepth = ModelGeneration.max_spawn_depth(self.GrowthGenerator, log_file=self.log_file)
            
        try:
            from rq import Connection, Queue, Worker
            self.redis_conn = redis.Redis(host=self.HostName, port=self.PortNumber)
            test_workers=self.use_rq
            self.rq_queue = Queue(self.Q_id, connection=self.redis_conn, async=test_workers, default_timeout=self.rq_timeout) # TODO is this timeout sufficient for ALL QMD jobs?

            parallel_enabled = True
        except:
            print("importing rq failed")
            parallel_enabled = False    

        self.RunParallel = parallel and parallel_enabled

        self.log_print(["Retrieving databases from redis"])
        self.RedisDataBases = rds.databases_from_qmd_id(self.HostName, self.PortNumber, self.Q_id)
        
        rds.flush_dbs_from_id(self.HostName, self.PortNumber, self.Q_id) # fresh redis databases for this instance of QMD.
 
        
        if self.QLE:
            self.QLE_Type = 'QLE'
        else: 
            self.QLE_Type = 'IQLE'
    
        self.QMDInfo = {
         # may need to take copies of these in case pointers accross nodes break
          'num_probes' : self.NumProbes,
#          'probe_dict' : self.ProbeDict, # possibly include here?
          'true_oplist' : self.TrueOpList,
          'true_params' : self.TrueParamsList,  
          'num_particles' : self.NumParticles,
          'num_experiments' : self.NumExperiments, 
          'resampler_thresh' : self.ResampleThreshold,
          'resampler_a' : self.ResamplerA,
          'pgh_prefactor' : self.PGHPrefactor,
          'debug_directory' : self.DebugDirectory,
          'qle' : self.QLE,
          'sigma_threshold' : self.SigmaThreshold,
          'true_name' : self.TrueOpName,
          'use_exp_custom' : self.UseExpCustom,
          'compare_linalg_exp_tol' : self.ExpComparisonTol,
          'gaussian' : self.gaussian,
          'q_id' : self.Q_id
        }
        
        self.log_print(["RunParallel=", self.RunParallel])
        compressed_qmd_info = pickle.dumps(self.QMDInfo, protocol=2)
        compressed_probe_dict = pickle.dumps(self.ProbeDict, protocol=2)
        qmd_info_db = self.RedisDataBases['qmd_info_db']
        self.log_print(["Saving qmd info db to ", qmd_info_db])
        qmd_info_db.set('QMDInfo', compressed_qmd_info)
        qmd_info_db.set('ProbeDict', compressed_probe_dict)

        # Initialise database and lists.
        self.log_print(["Running ", self.QLE_Type, " for true operator ", true_operator, " with parameters : ", self.TrueParamsList])
        self.initiateDB()
        

    def log_print(self, to_print_list):
        identifier = str(str(time_seconds()) +" [QMD "+ str(self.Q_id) +"]")
        if type(to_print_list)!=list:
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file, flush=True)
        

    def initiateDB(self):
        self.db, self.legacy_db, self.model_lists = \
            DataBase.launch_db(
                true_op_name = self.TrueOpName,
                log_file = self.log_file, 
                gen_list = self.InitialOpList,
                qle = self.QLE,
                true_ops = self.TrueOpList,
                true_params = self.TrueParamsList,
                num_particles = self.NumParticles,
                redimensionalise = False,
                resample_threshold = self.ResampleThreshold,
                resampler_a = self.ResamplerA,
                pgh_prefactor = self.PGHPrefactor,
                num_probes = self.NumProbes,
                probe_dict = self.ProbeDict,
                use_exp_custom = self.UseExpCustom,
                enable_sparse = self.EnableSparse,
                debug_directory = self.DebugDirectory,
                qid=self.Q_id, 
                host_name=self.HostName,
                port_number=self.PortNumber
            )
            
        for i in range(len(self.InitialOpList)):
            model = self.InitialOpList[i]
            self.ModelNameIDs[i] = model
            
    def addModel(self, model, branchID=0):
        #self.NumModels += 1
        tryAddModel = DataBase.add_model(
            model_name = model,
            running_database = self.db,
            num_particles = self.NumParticles, 
            true_op_name = self.TrueOpName,
            model_lists = self.model_lists,
            true_ops = self.TrueOpList,
            true_params = self.TrueParamsList,
            branchID = branchID,
            resample_threshold = self.ResampleThreshold,
            resampler_a = self.ResamplerA,
            pgh_prefactor = self.PGHPrefactor,
            num_probes = self.NumProbes,
            probe_dict = self.ProbeDict,
            use_exp_custom = self.UseExpCustom,
            enable_sparse = self.EnableSparse,
            debug_directory = self.DebugDirectory,
            modelID = self.NumModels,
            redimensionalise = False,
            qle = self.QLE,
            host_name = self.HostName, 
            port_number = self.PortNumber, 
            qid = self.Q_id,
            log_file = self.log_file
        )
        if tryAddModel == True: ## keep track of how many models/branches in play
            self.HighestModelID += 1 
            self.ModelNameIDs[self.NumModels] = model
            self.NumModels += 1
            if DataBase.get_num_qubits(model) > self.HighestQubitNumber:
                self.HighestQubitNumber = DataBase.get_num_qubits(model)

    def delete_unpicklable_attributes(self):
        del self.redis_conn
        del self.rq_queue
        del self.RedisDataBases
        del self.write_log_file

    def newBranch(self, model_list):
        self.HighestBranchID += 1
        branchID = self.HighestBranchID
        self.BranchBayesComputed[branchID] = False
        num_models = len(model_list)
        self.NumModelsPerBranch[branchID] = num_models
        self.NumModelPairsPerBranch[branchID] = num_pairs_in_list(num_models)
        self.BranchAllModelsLearned[branchID] = False
        self.BranchComparisonsComplete[branchID] = False
        
        for model in model_list:
            self.addModel(model, branchID=branchID)
        return branchID
    
    def printState(self):
        self.log_print(["Branch champions:", self.BranchChampions])
        self.log_print(["InterBranch champions:", self.InterBranchChampions])
        self.log_print(["Branch Rankings: \n", self.BranchRankings])
        
    def getModelInstance(self, name):
        try: 
            instance = DataBase.get_qml_instance(self.db, name)
            return instance
        except: 
            if name in list(self.legacy_db['<Name>']):
                self.log_print(["Operator in legacy databse - retired. "])
            else: 
                self.log_print(["Model not found."])
    def getOperatorInstance(self, name):
        try: 
            return DataBase.get_operator_instance(self.db, name)
        except:
            if name in list(self.legacy_db['<Name>']):
                self.log_print(["Operator in legacy databse - retired. "])
            else: 
               self.log_print(["Operator not found."])

    def getModelDBIndex(self, name):
        return DataBase.get_location(self.db, name)

    def getModelInstanceFromID(self, model_id):
        return DataBase.model_instance_from_id(self.db, model_id)    

    def reducedModelInstanceFromID(self, model_id):
        return DataBase.reduced_model_instance_from_id(self.db, model_id)    

    
    def killModel(self, name):
        if name not in list(self.db['<Name>']):
            print("Cannot remove ", name, "; not in ", list(self.db["<Name>"]))
        else:
            print("Killing model", name)
            # Add to legacy_db
            DataBase.move_to_legacy(self.db, self.legacy_db, name)
            model_instance = self.getModelInstance(name)
            operator_instance = self.getOperatorInstance(name)
            # Remove from self.db
            self.db = DataBase.remove_model(self.db, name)
            del model_instance
            del operator_instance
    
    def runIQLE(self, model, num_exp=50):
        model_exists=False
        if model in list(self.db['<Name>']):
            model_exists = True
        elif model in list(self.legacy_db['<Name>']):
            print("Model ", model, " previously considered and retired.")
        
        has_model_finished = self.pullField(name=model, field='Completed')
        
        if model_exists==True and has_model_finished==False : 
            model_instance = self.getModelInstance(model)
            print("Running ", self.QLE_Type, " for model: ", model)
            model_instance.UpdateModel(num_exp, 
                sigma_threshold = self.SigmaThreshold
            )
            self.updateModelRecord(name=model, field='Completed', new_value=True)
        else: 
            print("Model ", model ,"does not exist")

    
    def learnUnfinishedModels(self, use_rq=True, blocking=False):
        unfinished_model_names = DataBase.all_unfinished_model_names(self.db)
        for model_name in unfinished_model_names:
            print("Model ", model_name, "being learned")
            self.learnModel(model_name=model_name, use_rq=use_rq, blocking=blocking)
            self.updateModelRecord(field='Completed', 
                name=model_name,  new_value=True
            )
        

    def learnModelFromBranchID(self, branchID, use_rq=True, blocking=False):
        model_list = DataBase.model_names_on_branch(self.db, branchID)
        active_branches_learning_models = \
            self.RedisDataBases['active_branches_learning_models']
        active_branches_learning_models.set(int(branchID), 0)
        active_branches_learning_models.set('LOCKED', 0)

        for model_name in model_list:
            self.log_print(["Model ", model_name, "being learned"])
            self.learnModel(model_name=model_name, use_rq=use_rq, blocking=blocking)
            self.updateModelRecord(field='Completed', name=model_name, 
                new_value=True
            )

        
    
    
    def learnModelNameList(self, model_name_list, use_rq=True, blocking=False):
        for model_name in model_name_list:
            self.learnModel(model_name=model_name, use_rq=use_rq, blocking=blocking)
            self.updateModelRecord(field='Completed', name=model_name,
                new_value=True
            )
            
    
    def learnModel(self, model_name, use_rq = True, blocking=False): 
        exists = DataBase.check_model_exists(model_name=model_name,
            model_lists = self.model_lists, db = self.db
        )
        if exists:
            modelID = DataBase.model_id_from_name(self.db, name = model_name)
            branchID = DataBase.model_branch_from_model_id(self.db, 
                model_id=modelID
            )
            if self.RunParallel and use_rq:
            # i.e. use a job queue rather than sequentially doing it. 
                from rq import Connection, Queue, Worker
                queue = Queue(self.Q_id, connection=self.redis_conn,
                async=self.use_rq, default_timeout=self.rq_timeout
            ) # TODO is this timeout sufficient for ALL QMD jobs?

                # add function call to RQ queue
                queued_model = queue.enqueue(learnModelRemote, model_name,
                    modelID, branchID=branchID, remote=True, 
                    host_name=self.HostName, port_number=self.PortNumber,
                    qid=self.Q_id, log_file=self.rq_log_file, result_ttl=-1,
                    timeout=self.rq_timeout) 
                
                self.log_print(["Model", model_name, "added to queue."])
                if blocking: # i.e. wait for result when called. 
                    while not queued_model.is_finished:
                        if queued_model.is_failed:
                            self.log_print(["Model", model_name, "has failed on remote worker"])
                            break
                        time.sleep(0.1)
                    del updated_model_info

            else:
                self.QMDInfo['probe_dict'] = self.ProbeDict
                updated_model_info = learnModelRemote(model_name,
                    modelID, branchID=branchID, qmd_info=self.QMDInfo, 
                    remote=True, host_name=self.HostName, 
                    port_number=self.PortNumber,
                    qid=self.Q_id,log_file=self.rq_log_file
                )

                del updated_model_info
        else:
            self.log_print(["Model", model_name, "does not yet exist."])

    def remoteBayes(self, model_a_id, model_b_id, return_job=False, branchID=None, interbranch=False, remote=True, bayes_threshold=100):
        # only do this when both models have learned. TODO add check for this. 
        
        if branchID is None:
            interbranch=True            
        
        if self.use_rq:
            from rq import Connection, Queue, Worker
            queue = Queue(self.Q_id, connection=self.redis_conn, async=self.use_rq, default_timeout=self.rq_timeout) # TODO is this timeout sufficient for ALL QMD jobs?

            job = queue.enqueue(BayesFactorRemote, model_a_id=model_a_id,
                model_b_id=model_b_id, branchID=branchID, 
                interbranch=interbranch, 
                num_times_to_use = self.NumTimesForBayesUpdates, 
                trueModel=self.TrueOpName, bayes_threshold=bayes_threshold,
                host_name=self.HostName, port_number=self.PortNumber, 
                qid=self.Q_id, log_file=self.rq_log_file, result_ttl=-1,
                timeout=self.rq_timeout
            ) 
            self.log_print(["Bayes factor calculation queued. Model IDs",
                model_a_id, model_b_id]
            )
            if return_job:
                return job
        else:

            BayesFactorRemote(model_a_id=model_a_id, model_b_id=model_b_id,
                trueModel=self.TrueOpName, branchID=branchID,
                interbranch=interbranch, bayes_threshold=bayes_threshold,
                host_name=self.HostName, port_number=self.PortNumber, 
                qid=self.Q_id, log_file=self.rq_log_file
            )
            
    def remoteBayesFromIDList(self, model_id_list, remote=True,
        wait_on_result=False, recompute=False, bayes_threshold=1
    ): 
        remote_jobs=[]
        num_models = len(model_id_list)
        for i in range(num_models):
            a = model_id_list[i]
            for j in range(i,num_models):
                b=model_id_list[j]
                if a!=b:
                    unique_id = DataBase.unique_model_pair_identifier(a,b)
                    if (unique_id not in self.BayesFactorsComputed 
                        or recompute==True
                    ): #ie not yet considered
                        self.BayesFactorsComputed.append(unique_id)
                        remote_jobs.append(self.remoteBayes(a,b, remote=remote,
                            return_job=wait_on_result,
                            bayes_threshold=bayes_threshold)
                        )

        if wait_on_result and self.use_rq: # test_workers from RedisSettings
            self.log_print(["Waiting on result of \
                Bayes comparisons from given list:", model_id_list]
            )
            for job in remote_jobs:
                while job.is_finished == False:
                    time.sleep(0.01)
        else:
            self.log_print(["Not waiting on results \
                since not using RQ workers."]
            )
    
    def remoteBayesFromBranchID(self, branchID, remote=True, 
        bayes_threshold=50
    ):

        active_branches_bayes = self.RedisDataBases['active_branches_bayes']
        model_id_list = DataBase.active_model_ids_by_branch_id(self.db, branchID)
        active_branches_bayes.set(int(branchID), 0) # set up branch 0
        num_models = len(model_id_list)
        for i in range(num_models):
            a = model_id_list[i]
            for j in range(i,num_models):
                b=model_id_list[j]
                if a!=b:
                    unique_id = DataBase.unique_model_pair_identifier(a,b)
                    if (unique_id not in self.BayesFactorsComputed or recompute==True): #ie not yet considered
                        self.BayesFactorsComputed.append(unique_id)
                        self.remoteBayes(a,b, remote=remote, 
                            branchID=branchID, bayes_threshold=bayes_threshold
                        )

    def blockingQMD(self):
        self.learnModelNameList(model_name_list=self.InitialOpList, 
            blocking=False, use_rq=True
        )
        ids=DataBase.active_model_ids_by_branch_id(self.db, 0)
        self.remoteBayesFromIDList(ids, remote=True)
        self.remoteBranchBayesComparison(branchID=0)
    
    def remoteBranchBayesComparison(self, branchID):
        active_models_in_branch = \
            DataBase.active_model_ids_by_branch_id(self.db,
            branchID
        )
        
        num_models = len(active_models_in_branch)
        models_points = {}
        for i in range(num_models):
            a = active_models_in_branch[i]
            for j in range(i+1, num_models):
                b = active_models_in_branch[j]
                if a!=b:
                    self.processRemoteBayesPair(a=a, b=b)            
    

    def processAllRemoteBayesFactors(self):
        bayes_factors_db = self.RedisDataBases['bayes_factors_db']
        computed_pairs = bayes_factors_db.keys()
        #TODO check whether pair computed before using bayes dict of QMD, or something more efficient
        # TODO take list, or branch argument and only process those.
        for pair in computed_pairs:
            self.processRemoteBayesPair(pair=pair)


    def processRemoteBayesPair(self, a=None, b=None, pair=None, bayes_threshold=1):
        bayes_factors_db = self.RedisDataBases['bayes_factors_db']
        if pair is not None:        
            model_ids = pair.split(',')
            a=(float(model_ids[0]))
            b=(float(model_ids[1]))
        elif a is not None and b is not None:
            a=float(a)
            b=float(b)
            pair = DataBase.unique_model_pair_identifier(a,b)
        else:
            self.log_print(["Must pass either two model ids, or a \
                pair name string, to process Bayes factors."]
            )
            


        try:
            bayes_factor = float(bayes_factors_db.get(pair))
        except TypeError:
            self.log_print(["On bayes_factors_db for pair id", 
                pair, "value=", bayes_factors_db.get(pair)]
            )
            
        mod_a = self.reducedModelInstanceFromID(a)
        mod_b = self.reducedModelInstanceFromID(b)
        if b in mod_a.BayesFactors:
            mod_a.BayesFactors[b].append(bayes_factor)
        else:
            mod_a.BayesFactors[b] = [bayes_factor]
        
        if a in mod_b.BayesFactors:
            mod_b.BayesFactors[a].append((1.0/bayes_factor))
        else:
            mod_b.BayesFactors[a] = [(1.0/bayes_factor)]
            
        if bayes_factor > bayes_threshold: 
            return "a"
        elif bayes_factor <  (1.0/bayes_threshold):
            return "b"
                        

    def runAllActiveModelsIQLE(self, num_exp):
        active_models = self.db.loc[self.db['Status']=='Active']['<Name>']
        for model in active_models:
            self.runIQLE(model=model, num_exp=num_exp)
        self.GlobalEpoch += num_exp
            
        
    def updateModelRecord(self, field, name=None, model_id=None, 
        new_value=None, increment=None
    ):
        DataBase.update_field(
            db=self.db, 
            name=name,
            model_id=model_id,
            field=field,
            new_value=new_value,
            increment=increment
        )
    def pullField(self, name, field):
        return DataBase.pull_field(self.db, name, field)

    def statusChangeBranch(self, branchID, new_status='Saturated'):
        self.db.loc[ self.db['branchID']==branchID , 'Status'] = new_status

    def statusChangeModel(self, model_name, new_status='Saturated'):
        self.db.loc[ self.db['<Name>']==model_name , 'Status'] = new_status
        
    def getListTrueOpByDimension(self):
        self.TrueOpListByDim = {}
        self.TrueParamByDim = {}
        for dim in range(1, 1+self.MaxDimension):
            new_op = ModelGeneration.identity_interact(
                subsystem=self.TrueOpName, num_qubits=dim, return_operator=True
            )
            self.TrueOpListByDim[dim] = new_op.constituents_operators
        for i in range(1, self.TrueOpDim+1):
            self.TrueParamByDim[i] = self.TrueParamsList
        for i in range(self.TrueOpDim+1, self.MaxDimension):
            self.TrueParamByDim[i] = [self.TrueParamsList[0]]

    def compareModels(self, log_comparison_high=50.0, num_times_to_use='all',
        model_a_id = None, model_b_id =None, model_a = None, model_b = None,
        name_a=None, name_b=None, print_result = True
    ):
        # Either pass in name_a and name_b OR model_a and model_b
        if model_a is None and model_b is None:
            if model_a_id is not None and model_b_id is not None: 
                model_a = self.getModelInstanceFromID(model_a_id)
                model_b = self.getModelInstanceFromID(model_b_id)
            else: # if only names were passed 
                model_a = self.getModelInstance(name_a)
                model_b = self.getModelInstance(name_b)
        if model_a ==  model_b:
            return "Same Models"
        else: 
            log_comparison_low = 1.0/log_comparison_high
            if model_a_id is None and model_b is None:
                model_a_id = model_a.ModelID
                model_b_id = model_b.ModelID

            if num_times_to_use == 'all':
                times_a = model_a.TrackTime
            elif len(model_a.TrackTime) < num_times_to_use:
                times_a = model_a.TrackTime
            else: 
                times_a = model_a.TrackTime[num_times_to_use:]

            if num_times_to_use=='all':
                times_b = model_b.TrackTime
            elif len(model_b.TrackTime) < num_times_to_use:
                times_b = model_b.TrackTime
            else: 
                times_b = model_b.TrackTime[num_times_to_use:]
            
            times = []
            times.extend(times_a)
            times.extend(times_b)

            bayes_factor = compute_bayes_factor(model_a, model_b, times_a, times_b)

            model_a.addBayesFactor(compared_with=model_b_id,
                bayes_factor=bayes_factor
            )
            model_b.addBayesFactor(compared_with=model_a_id, 
                bayes_factor=1.0/bayes_factor
            )
            if print_result:
                self.log_print(["Bayes factor b/w ", model_a.Name, "&", 
                    model_b.Name," = ", bayes_factor]
                )
            if bayes_factor >= log_comparison_high: 
                if print_result: print("Point to ", model_a.Name)
                return "a"
            elif bayes_factor < log_comparison_low: 
                if print_result: print("Point to ", model_b.Name)
                return "b"

    def compareModelsWithinBranch(self, branchID, bayes_threshold=1):
        active_models_in_branch = DataBase.active_model_ids_by_branch_id(self.db,
            branchID
        )
        
        models_points = {}
        for model_id in active_models_in_branch:
            models_points[model_id] = 0
        
        for i in range(len(active_models_in_branch)):
            mod1 = active_models_in_branch[i]
            for j in range(i, len(active_models_in_branch)): 
                mod2 = active_models_in_branch[j]
                if mod1!=mod2:
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)
                    
                    if res == "a":
                        models_points[mod1] += 1
                    elif res == "b":
                        models_points[mod2] += 1
                        # todo if more than one model has max points
        max_points = max(models_points.values())
        max_points_branches = [key for key, val in models_points.items() 
            if val==max_points]
        
        if len(max_points_branches) > 1: 
            # todo: recompare. Fnc: compareListOfModels (rather than branch based)
            self.log_print(["Multiple models have same number of points within \
                branch.\n", models_points]
            )
            self.remoteBayesFromIDList(model_id_list=max_points_branches, 
                remote=True, recompute=True, bayes_threshold=bayes_threshold,
                wait_on_result=True
            )

            champ_id = self.compareModelList(max_points_branches,
                bayes_threshold=bayes_threshold, 
                models_points_dict=models_points
            )
        else: 
            champ_id = max(models_points, key=models_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        
        self.BranchChampions[int(branchID)] = champ_id
        for model_id in active_models_in_branch:
            self.updateModelRecord(model_id=model_id, field='Status',
                new_value='Deactivated'
            )
        self.updateModelRecord(name=DataBase.model_name_from_id(self.db, champ_id),
            field='Status', new_value='Active'
        )
        ranked_model_list = sorted(models_points, key=models_points.get, reverse=True)

        if self.BranchBayesComputed[int(float(branchID))] == False: 
        # only update self.BranchRankings the first time branch is considered
            self.BranchRankings[int(float(branchID))] = ranked_model_list
            self.BranchBayesComputed[int(float(branchID))] = True
            
        self.log_print(["Champion of branch ", branchID, " is ", champ_name])
        self.BayesPointsByBranch[branchID] = models_points
        return models_points, champ_id

    
    def compareModelList(self, model_list, bayes_threshold = 1,
        models_points_dict=None, num_times_to_use = 'all'
    ):
        models_points = {}
        for mod in model_list:
            models_points[mod] = 0
        
        for i in range(len(model_list)):
            mod1 = model_list[i]
            for j in range(i, len(model_list)):
                mod2 = model_list[j]
                if mod1 != mod2:
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)
                    if res == "a":
                        models_points[mod1] += 1
                        if models_points_dict is not None: 
                            models_points_dict[mod1]+=1
                    elif res == "b":
                        models_points[mod2]+=1
                        if models_points_dict is not None: 
                            models_points_dict[mod2]+=1

        max_points = max(models_points.values())
        max_points_branches = [key for key, val in models_points.items() 
            if val==max_points]
        if len(max_points_branches) > 1: 
            # todo: recompare. Fnc: compareListOfModels (rather than branch based)
            self.log_print(["Multiple models \
                have same number of points in compareModelList:",
                max_points_branches]
            )
            self.log_print(["Recompute Bayes bw:"])
            for i in max_points_branches:
                self.log_print([DataBase.model_name_from_id(self.db, i)])
            self.log_print(["Points:\n", models_points])
            self.remoteBayesFromIDList(model_id_list=max_points_branches,
                remote=True, recompute=True, bayes_threshold=1, wait_on_result=True
            )
            champ_id = self.compareModelList(max_points_branches, bayes_threshold=1)
        else: 
            self.log_print(["After comparing list:", models_points])
            champ_id = max(models_points, key=models_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        
        return champ_id
    
    
    
    def finalBayesComparisons(self, bayes_threshold=100):
        bayes_factors_db = self.RedisDataBases['bayes_factors_db']
        branch_champions = list(self.BranchChampions.values())
        job_list = []
        job_finished_count = 0
        interbranch_collapse_threshold = 1e5 ## if a spawned model is this much better than its parent, parent is deactivated
        num_champs = len(branch_champions)
        
        for k in range( num_champs -1 ):
            mod1 = branch_champions[k]
            mod2 = branch_champions[k+1]
            
            job_list.append(self.remoteBayes(model_a_id=mod1, model_b_id=mod2,
                return_job=True, remote=self.use_rq)
            )
            
        self.log_print(["Entering while loop in final bayes fnc."])
        self.log_print(["num champs = ", num_champs])
        
        if self.use_rq:
            self.log_print(["Waiting on parent/child Bayes factors."])
            for k in range(len(job_list)):
                self.log_print(["Waiting on parent/child Bayes factors."])
                while job_list[k].is_finished == False:
                    sleep(0.01)
        else:
            self.log_print(["Jobs all finished because not on RQ"])

        self.log_print(["Bayes calculated between branch champions"])
        for k in range(num_champs - 1):
            mod1 = branch_champions[k]
            mod2 = branch_champions[k+1]
            pair_id = DataBase.unique_model_pair_identifier(mod1, mod2)
            bf_from_db = bayes_factors_db.get(pair_id)
            bayes_factor = float(bf_from_db)
        
            if bayes_factor > interbranch_collapse_threshold:
                # bayes_factor heavily favours mod1, so deactive mod2
                self.log_print(["Parent model, ", mod1, 
                    "stronger than spawned; deactivating model", mod2]
                )
                self.updateModelRecord(model_id=mod2, field='Status',
                    new_value='Deactivated'
                )
            elif bayes_factor < (1.0/interbranch_collapse_threshold):
                self.log_print(["Spawned model", mod2, 
                    "stronger than parent; deactivating model", mod1]
                )
                self.updateModelRecord(model_id=mod1,
                    field='Status', new_value='Deactivated'
                )
                
            # Add bayes factors to BayesFactor dict for each model        
            mod_a = self.reducedModelInstanceFromID(mod1)
            mod_b = self.reducedModelInstanceFromID(mod2)
            if mod2 in mod_a.BayesFactors:
                mod_a.BayesFactors[mod2].append(bayes_factor)
            else:
                mod_a.BayesFactors[mod2] = [bayes_factor]
            
            if mod1 in mod_b.BayesFactors:
                mod_b.BayesFactors[mod1].append((1.0/bayes_factor))
            else:
                mod_b.BayesFactors[mod1] = [(1.0/bayes_factor)]
        
        
        
        active_models = DataBase.all_active_model_ids(self.db)
        self.SurvivingChampions = DataBase.all_active_model_ids(self.db)
        self.log_print(["After initial interbranch comparisons, \
            remaining active branch champions:", active_models]
        )
        num_active_models = len(active_models)
        
        self.remoteBayesFromIDList(model_id_list=active_models, remote=True,
            recompute=True, wait_on_result=True, bayes_threshold=bayes_threshold
        )

        branch_champions_points = {}
        for c in active_models: 
            branch_champions_points[c] = 0

        for i in range(num_active_models):
            mod1 = active_models[i]
            for j in range(i, num_active_models):
                mod2 = active_models[j]
                if mod1!=mod2:
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)
                    
                    if res == "a":
                        branch_champions_points[mod1] += 1
                    elif res == "b":
                        branch_champions_points[mod2] += 1
        self.ranked_champions = sorted(branch_champions_points, reverse=True)
        
        max_points = max(branch_champions_points.values())
        max_points_branches = [key for key, val in branch_champions_points.items() 
            if val==max_points
        ]
        if len(max_points_branches) > 1: 
            # todo: recompare. Fnc: compareListOfModels (rather than branch based)
            self.log_print(["No distinct champion, recomputing bayes \
                factors between : ", max_points_branches]
            )
            champ_id = self.compareModelList(max_points_branches, bayes_threshold=1, models_points_dict=branch_champions_points)
        else: 
            champ_id = max(branch_champions_points, key=branch_champions_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        
        branch_champ_names = [DataBase.model_name_from_id(self.db, mod_id) 
            for mod_id in active_models
        ]
        self.statusChangeModel(champ_name, new_status = 'Active')
        
        return champ_name, branch_champ_names
    
    
    def interBranchChampion(self, branch_list=[], just_active_models=False,
        global_champion=False, bayes_threshold=1
    ):
        
        all_branches = self.db['branchID'].unique()
        if global_champion == True: 
            branches = all_branches
        elif just_active_models:
            # some models turned off intermediately
            branches = DataBase.all_active_model_ids(self.db)
        else: 
            branches = branch_list
        self.log_print(["Branches : ", branches])
        
        
        num_branches = len(branches)
        points_by_branches = [None] * num_branches
        champions_of_branches = [None] * num_branches

        for i in range(num_branches):
            branchID = branches[i]
            if branchID not in all_branches:
                self.log_print(["branch ID : ", branchID])
                warnings.warn("branch not in database.")
                return False
            points_by_branches[i], champions_of_branches[i] = self.compareModelsWithinBranch(branchID)

        self.remoteBayesFromIDList(model_id_list=champions_of_branches, 
            remote=True, recompute=True, wait_on_result=True,
            bayes_threshold=bayes_threshold
        )

        self.log_print(["All jobs have finished while computing interbranch \
            champion"]
        )
        branch_champions_points = {}
        for c in champions_of_branches: 
            branch_champions_points[c] = 0

        for i in range(num_branches):
            mod1 = champions_of_branches[i]
            for j in range(i, num_branches):
                mod2 = champions_of_branches[j]
                if mod1!=mod2:
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)
                    
                    if res == "a":
                        branch_champions_points[mod1] += 1
                    elif res == "b":
                        branch_champions_points[mod2] += 1
        self.ranked_champions = sorted(branch_champions_points, reverse=True)
        
        max_points = max(branch_champions_points.values())
        max_points_branches = [key for key, val in branch_champions_points.items() 
            if val==max_points
        ]
        if len(max_points_branches) > 1: 
            # todo: recompare. Fnc: compareListOfModels (rather than branch based)
            self.log_print(["No distinct champion, recomputing bayes factors \
                between : ", max_points_branches]
            )
            champ_id = self.compareModelList(max_points_branches, 
                bayes_threshold=1, models_points_dict=branch_champions_points
            )
        else: 
            champ_id = max(branch_champions_points, 
                key=branch_champions_points.get
            )
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        
        branch_champ_names = [DataBase.model_name_from_id(self.db, mod_id) for 
            mod_id in champions_of_branches
        ]
        self.statusChangeModel(champ_name, new_status = 'Active')
        
        interBranchChampListID = len(self.InterBranchChampions)
        self.InterBranchChampions[interBranchChampListID] = [branches, champ_id]
        return champ_name, branch_champ_names
    
    def globalChampionCalculation(self):
        branches = self.db['branchID'].unique()
        
        num_branches = len(branches)
        self.points_by_branches = [None] * num_branches
        self.champions_of_branches = [None] * num_branches

        for i in range(num_branches):
            branchID = branches[i]
            self.points_by_branches[i], self.champions_of_branches[i] = (
                self.compareModelsWithinBranch(branchID)
            )

        self.champions_points = {}
        for c in self.champions_of_branches: 
            self.champions_points[c] = 0

        for i in range(num_branches):
            mod1 = self.champions_of_branches[i]
            for j in range(i, num_branches):
                mod2 = self.champions_of_branches[j]
                if mod1!=mod2:
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)
                
                    if res == "a":
                        self.champions_points[mod1] += 1
                    elif res == "b":
                        self.champions_points[mod2]+=1
        self.ranked_champions = sorted(self.champions_points, reverse=True)
        champ_id = max(self.champions_points, key=self.champions_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        self.log_print(["Champion of Champions is",  champ_name])
        
        
    def spawn(self, 
              branch_list = None, 
              num_models_to_consider=1, 
              single_champion=True, 
              all_branches=False,
              spawn_new = True
             ):
        if all_branches or branch_list is None: 
            global_champion = True
            
        overall_champ, branch_champions = \
            self.interBranchChampion(branch_list=branch_list,
            global_champion=global_champion
        )
        self.log_print(["Overall champion within spawn function:",
            overall_champ]
        )
        options=['x', 'y', 'z'] # append best model with these options
        
        if single_champion:
            new_models = ModelGeneration.new_model_list(model_list=[overall_champ],
                generator='simple_ising',model_dict=self.model_lists,
                log_file=self.log_file, options=options
            )
        else: 
            new_models = ModelGeneration.new_model_list(model_list=branch_champions,
            generator='simple_ising', model_dict=self.model_lists,
            log_file=self.log_file, options=options
        )
        
        self.log_print(["New models to add to new branch : ", new_models])
        self.newBranch(model_list=new_models) 


    def spawnFromBranch(self, branchID, num_models=1):
        self.SpawnDepth+=1
        self.log_print(["Spawning, spawn depth:", self.SpawnDepth])
        best_models = self.BranchRankings[branchID][:num_models]
        best_model_names = [DataBase.model_name_from_id(self.db, mod_id) for
            mod_id in best_models 
        ]
        new_models = ModelGeneration.new_model_list(model_list=best_model_names,
            spawn_depth=self.SpawnDepth, model_dict=self.model_lists,
            log_file=self.log_file, generator=self.GrowthGenerator
        )
        
        self.log_print(["New models to add to new branch : ", new_models])
        new_branch_id = self.newBranch(model_list=new_models) 
        self.learnModelFromBranchID(new_branch_id, blocking=False, use_rq=True)
        
        if self.SpawnDepth == self.MaxSpawnDepth:
            return True
        else:
            return False
        


    def runRemoteQMD(self, num_exp=40, num_spawns=1, max_branches= None,
        max_num_qubits = None, max_num_models=None, spawn=True,
        just_given_models=False
    ):

        active_branches_learning_models = (
            self.RedisDataBases['active_branches_learning_models']
        )
        active_branches_bayes = self.RedisDataBases['active_branches_bayes']
        self.learnModelFromBranchID(0, blocking=False, use_rq=True)
        max_spawn_depth_reached=False
        all_comparisons_complete=False

        while max_spawn_depth_reached==False:
            model_ids_on_db = list(active_branches_learning_models.keys())
            model_ids_on_db.remove(b'LOCKED')
            for branchID_bytes in model_ids_on_db:
                branchID = int(branchID_bytes)
                if (int(active_branches_learning_models.get(branchID)) == \
                    self.NumModelsPerBranch[branchID] 
                    and self.BranchAllModelsLearned[branchID]==False
                ):
                    
                    self.log_print(["All models on branch", branchID, 
                        "have finished learning."]
                    )
                    self.BranchAllModelsLearned[branchID] = True
                    self.remoteBayesFromBranchID(branchID)

            for branchID_bytes in active_branches_bayes.keys():
                
                branchID = int(branchID_bytes)
                bayes_calculated = active_branches_bayes.get(branchID_bytes)
                if (int(bayes_calculated) ==  
                    self.NumModelPairsPerBranch[branchID] and
                    self.BranchComparisonsComplete[branchID]==False
                ):
                    self.BranchComparisonsComplete[branchID] = True
                    self.compareModelsWithinBranch(branchID)
                    max_spawn_depth_reached = self.spawnFromBranch(branchID,
                        num_models=1
                    )

            if max_spawn_depth_reached:
                self.log_print(["Max spawn depth reached; determining winner. \
                    Entering while loop until all models/Bayes factors remaining \
                    have finished."]
                )
                still_learning = True

                while still_learning:
                    branch_ids_on_db = list(active_branches_learning_models.keys())
                    branch_ids_on_db.remove(b'LOCKED')
                    for branchID_bytes in branch_ids_on_db:
                        branchID = int(branchID_bytes)
                        if ( int(active_branches_learning_models.get(branchID)) == \
                            self.NumModelsPerBranch[branchID] and
                            self.BranchAllModelsLearned[branchID]==False
                        ):
                            self.BranchAllModelsLearned[branchID] = True
                            self.remoteBayesFromBranchID(branchID)
                            
                        if branchID_bytes in active_branches_bayes:
                            num_bayes_done_on_branch = (
                                active_branches_bayes.get(branchID_bytes)
                            )
                            if ( int(num_bayes_done_on_branch) == 
                                self.NumModelPairsPerBranch[branchID] and
                                self.BranchComparisonsComplete[branchID]==False
                            ):
                                self.BranchComparisonsComplete[branchID] = True
                                self.compareModelsWithinBranch(branchID)
                    
                    if (np.all(
                        np.array(list(self.BranchAllModelsLearned.values()))==True)
                        and
                        np.all(np.array(list(
                        self.BranchComparisonsComplete.values()))==True)
                    ):    
                            still_learning = False # i.e. break out of this while loop


        ### Final functions at end of QMD
        final_winner, final_branch_winners = self.finalBayesComparisons()        
        

        self.ChampionName = final_winner
        self.ChampID = self.pullField(name=final_winner, field='ModelID')
        self.log_print(["Final winner = ", final_winner])
        self.updateDataBaseModelValues()
        for i in range(self.HighestModelID):
            # Dict of all Bayes factors for each model considered. 
            self.AllBayesFactors[i] = (
                self.reducedModelInstanceFromID(i).BayesFactors
            )

        self.ChampionFinalParams = self.reducedModelInstanceFromID(self.ChampID).FinalParams

        champ_op = DataBase.operator(self.ChampionName)        
        num_params_champ_model = champ_op.num_constituents
        
        correct_model = misfit = underfit = overfit = 0
        self.log_print(["Num params - champ:", num_params_champ_model,"; \t true:", self.TrueOpNumParams])

        if DataBase.alph(self.ChampionName) == DataBase.alph(self.TrueOpName):
            correct_model = 1
        elif num_params_champ_model ==  self.TrueOpNumParams and DataBase.alph(self.ChampionName) != DataBase.alph(self.TrueOpName):
            misfit = 1
        elif num_params_champ_model > self.TrueOpNumParams: 
            overfit = 1
        elif num_params_champ_model < self.TrueOpNumParams: 
            underfit=1
        
        num_exp_ham = self.NumParticles * (self.NumExperiments + self.NumTimesForBayesUpdates)

        config = str( 'config' + 
            '_p'+str(self.NumParticles) +
            '_e' + str(self.NumExperiments) +
            '_b' + str( self.NumTimesForBayesUpdates)  +
            '_ra' + str(self.ResamplerA) +
            '_rt' + str(self.ResampleThreshold) +
            '_rp' + str(self.PGHPrefactor)
            )

        latex_config = str( 
            '$P_{'+str(self.NumParticles) +
            '}E_{' + str(self.NumExperiments) +
            '}B_{' + str( self.NumTimesForBayesUpdates)  +
            '}RT_{' + str(self.ResampleThreshold) +
            '}RA_{' + str(self.ResamplerA) +
            '}RP_{' + str(self.PGHPrefactor) +
            '}H_{' + str(num_exp_ham) + 
            '}$'
            )


        time_now = time.time()
        time_taken = time_now - self.StartingTime
        
        self.ChampionResultsDict = {
            'NameAlphabetical' : DataBase.alph(self.ChampionName),
            'NameNonAlph' : self.ChampionName,
            'FinalParams' : self.ChampionFinalParams,
            'LatexName' : DataBase.latex_name_ising(self.ChampionName),
            'NumParticles' : self.NumParticles,
            'NumExperiments' : self.NumExperiments,
            'NumBayesTimes' : self.NumTimesForBayesUpdates,
            'ResampleThreshold' : self.ResampleThreshold,
            'ResamplerA' : self.ResamplerA,
            'PHGPrefactor' : self.PGHPrefactor,
            'LogFile' : self.log_file,
            'ParamConfiguration' : config,
            'ConfigLatex' : latex_config,       
            'Time': time_taken,
            'QID' : self.Q_id,
            'CorrectModel' : correct_model,
            'Underfit' : underfit,
            'Overfit' : overfit, 
            'Misfit' : misfit
        }
               

    def updateDataBaseModelValues(self):
        for mod_id in range(self.HighestModelID):
            mod = self.reducedModelInstanceFromID(mod_id)
            mod.updateLearnedValues()
        
            
    def runQMD(self, num_exp = 20, max_branches= None, max_num_qubits = None,
        max_num_models=None, spawn=True, just_given_models=False
    ):
        if just_given_models:
            self.runAllActiveModelsIQLE(num_exp=num_exp)
            final_winner, final_branch_winners = (
                self.interBranchChampion(global_champion=True)
            )
            self.ChampionName = final_winner

            print("Final winner = ", final_winner)
            self.log_print([])

        else:
            if max_branches is None:
                max_branches = self.MaxBranchID

            if max_num_qubits is None:
                max_num_qubits = self.MaxQubitNumber
                
            if max_num_models is None: 
                max_num_models = self.MaxModNum
                
            while self.HighestQubitNumber < max_num_qubits: 
                self.runAllActiveModelsIQLE(num_exp=num_exp)
                self.spawn()
                if ( self.HighestBranchID > max_branches or \
                    self.NumModels > max_num_models
                ):
                    break
            
            self.runAllActiveModelsIQLE(num_exp=num_exp)
            self.log_print(["\n\n\n\nBayes Updates\n\n\n\n"])
            final_winner, final_branch_winners = (
                self.interBranchChampion(global_champion=True)
            )
            self.ChampionName = final_winner
            self.log_print(["Final winner = ", final_winner])

    def majorityVoteQMD(self, num_runs=1, num_exp=20, max_branches= None,
        max_num_qubits = None, max_num_models=None, spawn=True,
        just_given_models=False
    ):

        model_id_list = DataBase.active_model_ids_by_branch_id(self.db, branchID=0) 
        for i in range(num_runs):
            for j in model_id_list:
                mod = self.getModelInstanceFromID(j)
                mod.resetPrior()
                mod.UpdateModel(n_experiments=num_exp)
            self.compareModelList(model_list=model_id_list, bayes_threshold=1,
                num_times_to_use=num_exp
            )
        self.MajorityVotingScores = self.majorityVotingTally()

    def plotVolumes(self, model_id_list=None, branch_champions=False, 
        branch_id=None, save_to_file=None
    ):

        plt.clf()
        plot_descriptor = '\n('+str(self.NumParticles)+'particles; '+ \
            str(self.NumExperiments)+'experiments).'

        if branch_champions:
            # only plot for branch champions
            model_id_list = list(self.BranchChampions.values())
            plot_descriptor+='[Branch champions]'

        
        elif branch_id is not None:
            model_id_list = DataBase.list_model_id_in_branch(self.db, branch_id)
            plot_descriptor+='[Branch'+str(branch_id)+']'
        
        elif model_id_list is None:
            self.log_print(["Plotting volumes for all models by default."])
            
            model_id_list = range(self.HighestModelID)
            plot_descriptor+='[All models]'

        plt.title('Volume evolution through QMD '+plot_descriptor)
        plt.xlabel('Epoch')
        plt.ylabel('Volume')


        for i in model_id_list:
            vols = self.reducedModelInstanceFromID(i).VolumeList
            plt.semilogy(vols, label=str('ID:'+str(i)))
#            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        ax = plt.subplot(111)

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        lgd=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=4)


        if save_to_file is None:
            plt.show()
        else:
            plt.savefig(save_to_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
    

    def saveBayesCSV(self, save_to_file, names_ids='latex'):
        PlotQMD.BayesFactorsCSV(self, save_to_file, names_ids=names_ids)

    def writeInterQMDBayesCSV(self, bayes_csv):
        PlotQMD.updateAllBayesCSV(self, bayes_csv)

    def plotHintonAllModels(self, save_to_file=None):
        PlotQMD.plotHinton(model_names=self.ModelNameIDs, bayes_factors=self.AllBayesFactors, save_to_file=save_to_file)


    def plotHintonListModels(self, model_list, save_to_file=None):
        bayes_factors = {}
        for a in model_list:
            bayes_factors[a] = {}
            key_empty = True
            for b in model_list:
                if a!=b:
                    try:
                        bayes_factors[a][b] = self.AllBayesFactors[a][b]
                        key_empty = False
                    except:
                        pass
            if key_empty:
                bayes_factors.pop(a)
            
        model_name_dict = {}
        for m in model_list:
            model_name_dict[m] = DataBase.model_name_from_id(self.db, m)
        
        
        PlotQMD.plotHinton(model_names=model_name_dict, 
            bayes_factors=bayes_factors, save_to_file=save_to_file
        )
        
    def plotExpecValues(self, model_ids=None, champ=True, max_time=10,
        t_interval=0.1, save_to_file=None
    ):
        PlotQMD.ExpectationValuesTrueSim(qmd=self, model_ids=model_ids, 
            champ=champ, max_time=max_time, t_interval=t_interval,
            save_to_file=save_to_file
        )

    def plotTreeDiagram(self, modlist=None, save_to_file=None):
        PlotQMD.plotQMDTree(self, modlist=modlist, save_to_file=save_to_file)

    def plotRadarDiagram(self, modlist=None, save_to_file=None):
        plot_title=str("Radar Plot QMD "+ str(self.Q_id))
        if modlist is None:
            modlist = list(self.BranchChampions.values())
        PlotQMD.plotRadar(self, modlist, save_to_file=save_to_file,
            plot_title=plot_title
        )

    def majorityVotingTally(self):
        mod_ids = DataBase.list_model_id_in_branch(self.db, 0)
        tally = {}

        for i in mod_ids:
            mod = self.getModelInstanceFromID(i)
            tally[mod.Name] = 0
            scores = mod.BayesFactors
            for j in mod_ids:
                if i != j:
                    comparison = np.array(scores[j]) > 1
                    points = np.sum(comparison)
                    tally[mod.Name] += points
        return tally    


    def inspectModel(self, name):
        print("\nmodel name: ", name)
        mod = self.getModelInstance(name)
        
        print("experiments done ", mod.NumExperimentsToDate)
        print("times: ",  mod.TrackTime)
        print("final params : ", mod.FinalParams)
        print("bayes factors: ", mod.BayesFactors)
        

    def one_qubit_probes_bloch_sphere(self):
        print("In jupyter, include the following to view sphere: %matplotlib inline")
        import qutip as qt
        bloch = qt.Bloch()
        for i in range(self.NumProbes):
            state = self.ProbeDict[i,1]
            a = state[0]
            b = state[1]
            A=a*qt.basis(2,0)
            B=b*qt.basis(2,1)
            vec = (A + B)
            print(vec)
            bloch.add_states(vec)
        bloch.show()


def num_pairs_in_list(num_models):
    if num_models <= 1:
        return 0

    n = num_models
    k = 2 # ie. nCk where k=2 since we want pairs
    
    try:    
        a= math.factorial(n) / math.factorial(k)
        b= math.factorial(n-k)
    except:
        print("n=",n,"\t k=",k)
    
    return a/b

def separable_probe_dict(max_num_qubits, num_probes):
    seperable_probes = {}
    for i in range(num_probes):
        seperable_probes[i,0] = random_probe(1)
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = np.tensordot(seperable_probes[i,j-1], random_probe(1), axes=0).flatten(order='c')
            while np.isclose(1.0, np.linalg.norm(seperable_probes[i,j]), atol=1e-14) is  False:
                print("non-unit norm: ", np.linalg.norm(seperable_probes[i,j]))
                # keep replacing until a unit-norm 
                seperable_probes[i,j] = np.tensordot(seperable_probes[i,j-1], random_probe(1), axes=0).flatten(order='c')
    return seperable_probes

def random_probe(num_qubits):
    dim = 2**num_qubits
    real = []
    imaginary = []
    complex_vectors = []
    for i in range(dim):
        real.append(np.random.uniform(low=-1, high=1))
        imaginary.append(np.random.uniform(low=-1, high=1))
        complex_vectors.append(real[i] + 1j*imaginary[i])

    a=np.array(complex_vectors)
    norm_factor = np.linalg.norm(a)
    probe = complex_vectors/norm_factor
    if np.isclose(1.0, np.linalg.norm(probe), atol=1e-14) is False:
        print("Probe not normalised. Norm factor=", np.linalg.norm(probe)-1)
        return random_probe(num_qubits)

    return probe
