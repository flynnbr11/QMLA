from __future__ import print_function # so print doesn't show brackets
# Libraries
import numpy as np
import itertools as itr
import os as os
import sys as sys 
import pandas as pd
import warnings
import time as time
import random
from psutil import virtual_memory
from RedisSettings import *
flushdatabases() # only want to do this once at the start!
import pickle 
pickle.HIGHEST_PROTOCOL=2 # TODO if >python3, can use higher protocol
import json
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math

# Local files
import Evo as evo
import DataBase 
import QML
import ModelGeneration
import BayesF
from RemoteModelLearning import *
from RemoteBayesFactor import * 
# Class definition


class QMD():
    #TODO: rename as ModelsDevelopmentClass when finished
    def __init__(self,
                 initial_op_list=['x'],
                 true_operator='x',
                 true_param_list = None,
                 num_particles= 300,
                 num_experiments = 50,
                 max_num_models=10, 
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
                 sigma_threshold = 1e-13, 
                 debug_directory = None,
                 qle = True, # Set to False for IQLE
                 parallel = False,
                 growth_generator='simple_ising'
                ):
#    def __init__(self, initial_op_list, true_op_list, true_param_list):
        self.QLE = qle # Set to False for IQLE
        trueOp = DataBase.operator(true_operator)
        self.TrueOpName = true_operator
        self.TrueOpDim = trueOp.num_qubits
        self.InitialOpList = initial_op_list
        self.TrueOpList = trueOp.constituents_operators
        if true_param_list is not None: 
            self.TrueParamsList = true_param_list
        else:
            print("No parameters passed, randomising")
            self.TrueParamsList = [random.random() for i in self.TrueOpList] # TODO: actual true params?
        # todo set true parmams properly
        #self.TrueParamsList = [0.75 for i in self.TrueOpList] # TODO: actual true params?
        #self.TrueHam = evo.getH(self.TrueParamsList, self.TrueOpList)
        #self.TrueHam = np.tensordot(self.TrueParamsList, self.TrueOpList, axes=1)
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
        self.SigmaThreshold = sigma_threshold
        self.DebugDirectory = debug_directory
        self.ModelPointsDict = {}
        self.BranchBayesComputed[0] = False
        self.BayesFactorsComputed = []
#        for i in range(self.MaxBranchID+1):
#            self.BranchChampions[i] = 0
#        for i in range(self.MaxLayerNumber+1):
#            self.LayerChampions[i] = 0
        self.ModelNameIDs = {}
        self.RunParallel = parallel and parallel_enabled # only true if both. 
        self.GrowthGenerator = growth_generator
        self.SpawnDepth = 0
        self.MaxSpawnDepth = ModelGeneration.max_spawn_depth(self.GrowthGenerator)
        self.NumModelsPerBranch = {0:len(self.InitialOpList)}
        self.NumModelPairsPerBranch = {0 : num_pairs_in_list(len(self.InitialOpList))}
        self.BranchAllModelsLearned = { 0 : False}
        self.BranchComparisonsComplete = {0 : False}
#        active_branches_bayes.set(int(0), 0)


        
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
          'true_name' : self.TrueOpName
        }
        if self.RunParallel:
            compressed_qmd_info = pickle.dumps(self.QMDInfo)
            compressed_probe_dict = pickle.dumps(self.ProbeDict)
            qmd_info_db.set('QMDInfo', compressed_qmd_info)
            qmd_info_db.set('ProbeDict', compressed_probe_dict)

        print("\nRunning ", self.QLE_Type, " for true operator ", true_operator, " with parameters : ", self.TrueParamsList)
        # Initialise database and lists.
        self.initiateDB()
        
    def initiateDB(self):
        ## TODO: Models should be initialised with appropriate TrueOp dimension -- see getListTrueOpByDimension function
        self.db, self.legacy_db, self.model_lists = \
            DataBase.launch_db(
                true_op_name = self.TrueOpName,
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
                debug_directory = self.DebugDirectory
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
            qle = self.QLE
        )
        if tryAddModel == True: ## keep track of how many models/branches in play
            self.HighestModelID += 1 
            # this_model_id = model_id_from_name(self.db, name = model) 
            self.ModelNameIDs[self.NumModels] = model
            self.NumModels += 1
            if DataBase.get_num_qubits(model) > self.HighestQubitNumber:
                self.HighestQubitNumber = DataBase.get_num_qubits(model)
        #else: 
        #    self.NumModels-=int(1)

    def newBranch(self, model_list):
        self.HighestBranchID += 1
        branchID = self.HighestBranchID
        self.BranchBayesComputed[branchID] = False
        num_models = len(model_list)
        self.NumModelsPerBranch[branchID] = num_models
        self.NumModelPairsPerBranch[branchID] = num_pairs_in_list(num_models)
        self.BranchAllModelsLearned[branchID] = False
        self.BranchComparisonsComplete[branchID] = False
#        active_branches_bayes.set(int(branchID), 0)
        
        for model in model_list:
            self.addModel(model, branchID=branchID)
        return branchID
    
    def printState(self):
        print("Branch champions: \n", self.BranchChampions)
        print("InterBranch champions: \n", self.InterBranchChampions)
        print("Branch Rankings: \n", self.BranchRankings)
        #print("Layer Champions: \n", self.LayerChampions)
            
            
    def getModelInstance(self, name):
        try: 
            instance = DataBase.get_qml_instance(self.db, name)
            return instance
        except: 
            if name in list(self.legacy_db['<Name>']):
                print("Operator in legacy databse - retired. ")
            else: 
                print("Model not found.")
    def getOperatorInstance(self, name):
        try: 
            return DataBase.get_operator_instance(self.db, name)
        except:
            if name in list(self.legacy_db['<Name>']):
                print("Operator in legacy databse - retired. ")
            else: 
                print("Operator not found.")

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
            model_instance.UpdateModel(num_exp, sigma_threshold = self.SigmaThreshold)
            self.updateModelRecord(name=model, field='Completed', new_value=True)
            #model_instance.BayesOnModelsWithinbranches
        else: 
            print("Model ", model ,"does not exist")

    
    def learnUnfinishedModels(self, use_rq=True, blocking=False):
        unfinished_model_names = DataBase.all_unfinished_model_names(self.db)
        for model_name in unfinished_model_names:
            print("Model ", model_name, "being learned")
            self.learnModel(model_name=model_name, use_rq=use_rq, blocking=blocking)
            self.updateModelRecord(field='Completed', name=model_name,  new_value=True)
        

    def learnModelFromBranchID(self, branchID, use_rq=True, blocking=False):
        model_list = DataBase.model_names_on_branch(self.db, branchID)
        active_branches_learning_models.set(int(branchID), 0)
        for model_name in model_list:
            print("Model ", model_name, "being learned")
            self.learnModel(model_name=model_name, use_rq=use_rq, blocking=blocking)
            self.updateModelRecord(field='Completed', name=model_name,  new_value=True)

        
    
    
    def learnModelNameList(self, model_name_list, use_rq=True, blocking=False):
        for model_name in model_name_list:
            self.learnModel(model_name=model_name, use_rq=use_rq, blocking=blocking)
            self.updateModelRecord(field='Completed', name=model_name,  new_value=True)
            
    
    def learnModel(self, model_name, use_rq = True, blocking=False): 
        exists = DataBase.check_model_exists(model_name=model_name, model_lists = self.model_lists, db = self.db)
        if exists:
            modelID = DataBase.model_id_from_name(self.db, name = model_name)
            branchID = DataBase.model_branch_from_model_id(self.db, model_id=modelID)
            if self.RunParallel and use_rq: # i.e. use a job queue rather than sequentially doing it. 
                # add function call to RQ queue
                queued_model = q.enqueue(learnModelRemote, model_name, modelID, branchID=branchID, remote=True) # add result_ttl=-1 to keep result indefinitely on redis server
                
                print("Model", model_name, "added to queue.")
                if blocking: # i.e. wait for result when called. 
                    while not queued_model.is_finished:
                        if queued_model.is_failed:
                            print("Model", model_name, "has failed on remote worker")
                            break
                        time.sleep(0.1)
                    updated_model_info = pickle.loads(learned_models_info[modelID])
                    reduced_model = DataBase.pull_field(self.db, name=model_name, field='Reduced_Model_Class_Instance')
                    reduced_model.updateLearnedValues(learned_info = updated_model_info)
                    del updated_model_info


            else:
                self.QMDInfo['probe_dict'] = self.ProbeDict
                updated_model_info = learnModelRemote(model_name, modelID, branchID=branchID, qmd_info=self.QMDInfo, remote=False)

                reduced_model = DataBase.pull_field(self.db, name=model_name, field='Reduced_Model_Class_Instance')
                reduced_model.updateLearnedValues(learned_info = updated_model_info)
                del updated_model_info
        else:
            print("Model", model_name, "does not yet exist.")        


    def remoteBayes(self, model_a_id, model_b_id, branchID=None, interbranch=False, remote=True, bayes_threshold=100):
        # only do this when both models have learned. TODO add check for this. 
        
        if branchID is None:
            interbranch=True            
        
        if remote:
            q.enqueue(BayesFactorRemote, model_a_id=model_a_id, model_b_id=model_b_id, branchID=branchID, interbranch=interbranch, num_times_to_use = self.NumTimesForBayesUpdates,  trueModel=self.TrueOpName, bayes_threshold=bayes_threshold) 
            print("Bayes factor calculation queued.")
        else:
            BayesFactorRemote(model_a_id=model_a_id, model_b_id=model_b_id, trueModel=self.TrueOpName, branchID=branchID, interbranch=interbranch, bayes_threshold=bayes_threshold)
        

    def remoteBayesFromIDList(self, model_id_list, remote=True, recompute=False, bayes_threshold=1): 
    #TODO need a remote blocking argument for when this is called in case where numerous models have same points   
        num_models = len(model_id_list)
        for i in range(num_models):
            a = model_id_list[i]
            for j in range(i,num_models):
                b=model_id_list[j]
                if a!=b:
                    unique_id = DataBase.unique_model_pair_identifier(a,b)
                    if (unique_id not in self.BayesFactorsComputed or recompute==True): #ie not yet considered
                        self.BayesFactorsComputed.append(unique_id)
                        self.remoteBayes(a,b, remote=remote, bayes_threshold=bayes_threshold)
#                    else:
#                       print("Bayes already computed bw", a,b)
    
    
    
    def remoteBayesFromBranchID(self, branchID, remote=False, bayes_threshold=50):
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
                        self.remoteBayes(a,b, remote=remote, branchID=branchID, bayes_threshold=bayes_threshold)
#                    else:
#                       print("Bayes already computed bw", a,b)
        



    def blockingQMD(self):
        # primarily for use during development. 
        self.learnModelNameList(model_name_list=self.InitialOpList, blocking=True, use_rq=False)
        ids=DataBase.active_model_ids_by_branch_id(self.db, 0)
        self.remoteBayesFromIDList(ids, remote=False)
        self.remoteBranchBayesComparison(branchID=0)
    
    def remoteBranchBayesComparison(self, branchID):
        active_models_in_branch = DataBase.active_model_ids_by_branch_id(self.db, branchID)
        
        num_models = len(active_models_in_branch)
        models_points = {}
        for i in range(num_models):
            a = active_models_in_branch[i]
            for j in range(i+1, num_models):
                b = active_models_in_branch[j]
                if a!=b:
                    self.processRemoteBayesPair(a=a, b=b)            
    

    def processAllRemoteBayesFactors(self):
        computed_pairs = bayes_factors_db.keys()
        #TODO check whether pair computed before using bayes dict of QMD, or something more efficient
        # TODO take list, or branch argument and only process those.
        for pair in computed_pairs:
            self.processRemoteBayesPair(pair=pair)


    def processRemoteBayesPair(self, a=None, b=None, pair=None, bayes_threshold=1):
        if pair is not None:        
            model_ids = pair.split(',')
            a=(float(model_ids[0]))
            b=(float(model_ids[1]))
        elif a is not None and b is not None:
            a=float(a)
            b=float(b)
            pair = DataBase.unique_model_pair_identifier(a,b)
        else:
            print("Must pass either two model ids, or a pair name string, to process Bayes factors.")


        bayes_factor = float(bayes_factors_db.get(pair))
            
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
            
        
    def updateModelRecord(self, field, name=None, model_id=None, new_value=None, increment=None):
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
            new_op = ModelGeneration.identity_interact(subsystem=self.TrueOpName, num_qubits=dim, return_operator=True)
            self.TrueOpListByDim[dim] = new_op.constituents_operators
        for i in range(1, self.TrueOpDim+1):
            self.TrueParamByDim[i] = self.TrueParamsList
        for i in range(self.TrueOpDim+1, self.MaxDimension):
            self.TrueParamByDim[i] = [self.TrueParamsList[0]]

    def compareModels(self, log_comparison_high=50.0, num_times_to_use = 'all', model_a_id = None, model_b_id =None, model_a = None, model_b = None, name_a=None, name_b=None, print_result = True):
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
#            print("Computing Bayes Factor b/w ", model_a.Name, " & ", model_b.Name)
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

 #           num_times_to_use = min(len(times_a), len(times_b))
#            times = times_a[:num_times_to_use] + times_b[:num_times_to_use]
            
            times = []
            times.extend(times_a)
            times.extend(times_b)

            bayes_factor = compute_bayes_factor(model_a, model_b, times_a, times_b)

            model_a.addBayesFactor(compared_with=model_b_id, bayes_factor=bayes_factor)
            model_b.addBayesFactor(compared_with=model_a_id, bayes_factor=1.0/bayes_factor)
            if print_result:
              print("Bayes factor b/w ", model_a.Name, "&", model_b.Name," = ", bayes_factor)

            if bayes_factor >= log_comparison_high: 
                if print_result: print("Point to ", model_a.Name)
                return "a"
            elif bayes_factor < log_comparison_low: 
                if print_result: print("Point to ", model_b.Name)
                return "b"
            


        


    def compareModelsWithinBranch(self, branchID, bayes_threshold=1):
        active_models_in_branch = DataBase.active_model_ids_by_branch_id(self.db, branchID)
        
        models_points = {}
        for model_id in active_models_in_branch:
            models_points[model_id] = 0
        
        for i in range(len(active_models_in_branch)):
            mod1 = active_models_in_branch[i]
            for j in range(i, len(active_models_in_branch)): 
                mod2 = active_models_in_branch[j]
                if mod1!=mod2:
                    #res = self.compareModels(model_a_id = mod1, model_b_id=mod2)
#                    pair = DataBase.unique_model_pair_identifier(mod1, mod2)
#                    res = bayes_factors_winners_db.get(pair)
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)

                    
                    if res == "a":
                        models_points[mod1] += 1
                    elif res == "b":
                        models_points[mod2] += 1
                        # todo if more than one model has max points
        max_points = max(models_points.values())
        max_points_branches = [key for key, val in models_points.items() if val==max_points]
        
        if len(max_points_branches) > 1: 
            # todo: recompare. Fnc: compareListOfModels (rather than branch based)
            print("Multiple models have same number of points within branch.\n", models_points)
            self.remoteBayesFromIDList(model_id_list=max_points_branches, remote=False, recompute=True, bayes_threshold=bayes_threshold)

            champ_id = self.compareModelList(max_points_branches, bayes_threshold=bayes_threshold, models_points_dict=models_points)
        else: 
            champ_id = max(models_points, key=models_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        
        #todo list of ranked models by branch
        
        self.BranchChampions[int(branchID)] = champ_id
        for model_id in active_models_in_branch:
            self.updateModelRecord(model_id=model_id, field='Status', new_value='Deactivated')
        self.updateModelRecord(name=DataBase.model_name_from_id(self.db, champ_id), field='Status', new_value='Active')

        ranked_model_list = sorted(models_points, key=models_points.get, reverse=True)

        if self.BranchBayesComputed[int(float(branchID))] == False: # only update self.BranchRankings the first time branch is considered
            self.BranchRankings[int(float(branchID))] = ranked_model_list
            self.BranchBayesComputed[int(float(branchID))] = True
            
        print("Champion of branch ", branchID, " is ", champ_name)
        self.BayesPointsByBranch[branchID] = models_points
        return models_points, champ_id

    
    def compareModelList(self, model_list, bayes_threshold = 1, models_points_dict=None, num_times_to_use = 'all'):
        #TODO not sure this function is suitable to call recursively by pulling bayes factors from redis db, doesn't seem to work
        # i.e. when 3 models need to be reconsidered, announcing the wrong winner.    
        models_points = {}
        for mod in model_list:
            models_points[mod] = 0
        
        for i in range(len(model_list)):
            mod1 = model_list[i]
            for j in range(i, len(model_list)):
                mod2 = model_list[j]
                if mod1 != mod2:
    #                res = self.compareModels(model_a_id=mod1, model_b_id=mod2, log_comparison_high=bayes_threshold, num_times_to_use=num_times_to_use)
#                    pair = DataBase.unique_model_pair_identifier(mod1, mod2)                    
 #                   res = bayes_factors_winners_db.get(pair)
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
        max_points_branches = [key for key, val in models_points.items() if val==max_points]
        if len(max_points_branches) > 1: 
            # todo: recompare. Fnc: compareListOfModels (rather than branch based)
            #print("No distinct champion, recomputing bayes factors between : ", max_points_branches)
            print("Multiple models have same number of points in compareModelList:", max_points_branches)
            print("Recompute Bayes bw:")
            for i in max_points_branches:
                print(DataBase.model_name_from_id(self.db, i))
            print("Points:\n", models_points)
            self.remoteBayesFromIDList(model_id_list=max_points_branches, remote=False, recompute=True, bayes_threshold=1)
            champ_id = self.compareModelList(max_points_branches, bayes_threshold=1)
        else: 
            print("After comparing list:", models_points)
            champ_id = max(models_points, key=models_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        
        return champ_id
    
    def interBranchChampion(self, branch_list=[], global_champion=False, bayes_threshold=1):
        all_branches = self.db['branchID'].unique()
        if global_champion == True: 
            branches = all_branches
        else: 
            branches = branch_list
        print("Branches : ", branches)
        
        num_branches = len(branches)
        points_by_branches = [None] * num_branches
        champions_of_branches = [None] * num_branches

        for i in range(num_branches):
            branchID = branches[i]
            if branchID not in all_branches:
                print("branch ID : ", branchID)
                warnings.warn("branch not in database.")
                return False
            points_by_branches[i], champions_of_branches[i] = self.compareModelsWithinBranch(branchID)

        self.remoteBayesFromIDList(model_id_list=champions_of_branches, remote=False, recompute=True, bayes_threshold=bayes_threshold)

        branch_champions_points = {}
        for c in champions_of_branches: 
            branch_champions_points[c] = 0

        for i in range(num_branches):
            mod1 = champions_of_branches[i]
            for j in range(i, num_branches):
                mod2 = champions_of_branches[j]
                if mod1!=mod2:
    #                res = self.compareModels(model_a_id=mod1, model_b_id=mod2, log_comparison_high=20.0)
#                    pair = DataBase.unique_model_pair_identifier(mod1, mod2)
#                    res = bayes_factors_winners_db.get(pair)
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)
                    
                    if res == "a":
                        branch_champions_points[mod1] += 1
                    elif res == "b":
                        branch_champions_points[mod2] += 1
        self.ranked_champions = sorted(branch_champions_points, reverse=True)
        
        max_points = max(branch_champions_points.values())
        max_points_branches = [key for key, val in branch_champions_points.items() if val==max_points]
        if len(max_points_branches) > 1: 
            # todo: recompare. Fnc: compareListOfModels (rather than branch based)
            print("No distinct champion, recomputing bayes factors between : ", max_points_branches)
            champ_id = self.compareModelList(max_points_branches, bayes_threshold=1, models_points_dict=branch_champions_points)
        else: 
            champ_id = max(branch_champions_points, key=branch_champions_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        
        branch_champ_names = [DataBase.model_name_from_id(self.db, mod_id) for mod_id in champions_of_branches]
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
            self.points_by_branches[i], self.champions_of_branches[i] = self.compareModelsWithinBranch(branchID)

        self.champions_points = {}
        for c in self.champions_of_branches: 
            self.champions_points[c] = 0

        for i in range(num_branches):
            mod1 = self.champions_of_branches[i]
            for j in range(i, num_branches):
                mod2 = self.champions_of_branches[j]
                if mod1!=mod2:
#                    pair = DataBase.unique_model_pair_identifier(mod1, mod2)
#                    res = bayes_factors_winners_db.get(pair)
                    res = self.processRemoteBayesPair(a=mod1, b=mod2)
                
#                    res = self.compareModels(model_a_id=mod1, model_b_id=mod2, log_comparison_high=10.0)
                    if res == "a":
                        self.champions_points[mod1] += 1
                    elif res == "b":
                        self.champions_points[mod2]+=1
        self.ranked_champions = sorted(self.champions_points, reverse=True)
        champ_id = max(self.champions_points, key=self.champions_points.get)
        champ_name = DataBase.model_name_from_id(self.db, champ_id)
        print("Champion of Champions is",  champ_name)
        
        
    def spawn(self, 
              branch_list = None, 
              num_models_to_consider=1, 
              single_champion=True, 
              all_branches=False,
              spawn_new = True
             ):
        if all_branches or branch_list is None: 
            global_champion = True
            
        overall_champ, branch_champions = self.interBranchChampion(branch_list=branch_list, global_champion=global_champion)
        print("Overall champion within spawn function:", overall_champ)
        options=['x', 'y', 'z'] # append best model with these options
        
        # for ising model of all x components
#        options = ['x']

        if single_champion:
            new_models = ModelGeneration.new_model_list(model_list=[overall_champ], generator='simple_ising',options=options)
        else: 
            new_models = ModelGeneration.new_model_list(model_list=branch_champions, generator='simple_ising', options=options)
        
        print("New models to add to new branch : ", new_models)
        self.newBranch(model_list=new_models) 
      #todo probailistically append model_list with suboptimal model in any of the branches in branch_list


    def spawnFromBranch(self, branchID, num_models=1):
        self.SpawnDepth+=1
        print("Spawning, spawn depth:", self.SpawnDepth)
        best_models = self.BranchRankings[branchID][:num_models]
        best_model_names = [DataBase.model_name_from_id(self.db, mod_id) for mod_id in best_models ]
        new_models = ModelGeneration.new_model_list(model_list=best_model_names, spawn_depth=self.SpawnDepth, generator=self.GrowthGenerator)
        
        print("New models to add to new branch : ", new_models)
        new_branch_id = self.newBranch(model_list=new_models) 

        self.learnModelFromBranchID(new_branch_id, blocking=True, use_rq=False)

        
        if self.SpawnDepth == self.MaxSpawnDepth:
            return True
        else:
            return False
        


    def runRemoteQMD(self, num_exp=40, num_spawns=1, max_branches= None, max_num_qubits = None, max_num_models=None, spawn=True, just_given_models=False):

#        self.learnModelNameList(model_name_list=self.InitialOpList, blocking=True, use_rq=False)


        self.learnModelFromBranchID(0, blocking=True, use_rq=False)
        max_spawn_depth_reached=False
        all_comparisons_complete=False
#        for i in range(num_spawns):
#        while max_spawn_depth_reached==False and all_comparisons_complete==False: 
        while max_spawn_depth_reached==False:
    #            print("While loop: before active learning for loop")
            for branchID_bytes in active_branches_learning_models.keys():
                branchID = int(branchID_bytes)
                if int(active_branches_learning_models.get(branchID)) == self.NumModelsPerBranch[branchID] and self.BranchAllModelsLearned[branchID]==False:
                    
                    print("All models on branch", branchID, "have finished learning.")

                    #print("Deleting branch", branchID, "from active_branches_learning_models")    

                    # del active_branches_learning_models[branchID]
                    self.BranchAllModelsLearned[branchID] = True
                    self.remoteBayesFromBranchID(branchID)

            for branchID_bytes in active_branches_bayes.keys():
                
                branchID = int(branchID_bytes)
                from_db = active_branches_bayes.get(branchID_bytes)
                if int(from_db) == self.NumModelPairsPerBranch[branchID] and self.BranchComparisonsComplete[branchID]==False:
                    self.BranchComparisonsComplete[branchID] = True
                    
                    try: self.compareModelsWithinBranch(branchID)
                    except ValueError:
                        print("New models have already been considered")
                        final_winner, final_branch_winners = self.interBranchChampion(global_champion=True)
                        self.ChampionName = final_winner

                        print("Final winner = ", final_winner)
                        max_spawn_depth_reached=True
                    try:
                        max_spawn_depth_reached = self.spawnFromBranch(branchID, num_models=2)
                        
                    except ValueError:
                        print("New models have already been considered")
                        max_spawn_depth_reached=True
                        #TODO tidy up function
                    #print("Deleting branch", branchID, "from active_branches_bayes")    
                    #del active_branches_learning_models[branchID]
                        
#            print("While loop: before checking if spawn depth reached")

            if max_spawn_depth_reached:
                print("Max spawn depth reached; determining winner. Entering while loop until all models/Bayes factors remaining have finished.")


                still_learning = True
                while still_learning:
                    for branchID_bytes in active_branches_learning_models.keys():
                        branchID = int(branchID_bytes)
                        if int(active_branches_learning_models.get(branchID)) == self.NumModelsPerBranch[branchID] and self.BranchAllModelsLearned[branchID]==False:
                            self.BranchAllModelsLearned[branchID] = True
                            self.remoteBayesFromBranchID(branchID)

                            
                        from_db = active_branches_bayes.get(branchID_bytes)
                        if int(from_db) == self.NumModelPairsPerBranch[branchID] and self.BranchComparisonsComplete[branchID]==False:
                            self.BranchComparisonsComplete[branchID] = True

                    
                    if np.all(np.array(list(self.BranchAllModelsLearned.values()))==True) and np.all(np.array(list(self.BranchComparisonsComplete.values()))==True):
                            
                            still_learning = False # i.e. break out of this while loop
                            final_winner, final_branch_winners = self.interBranchChampion(global_champion=True)
                            self.ChampionName = final_winner

                            print("Final winner = ", final_winner)
                    
               
        



    def runQMD(self, num_exp = 20, max_branches= None, max_num_qubits = None, max_num_models=None, spawn=True, just_given_models=False):
        if just_given_models:
              self.runAllActiveModelsIQLE(num_exp=num_exp)
              final_winner, final_branch_winners = self.interBranchChampion(global_champion=True)
              self.ChampionName = final_winner
              
              print("Final winner = ", final_winner)

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
                if self.HighestBranchID > max_branches or self.NumModels > max_num_models:
                    break

            
            self.runAllActiveModelsIQLE(num_exp=num_exp)
            print("\n\n\n\nBayes Updates\n\n\n\n")
            final_winner, final_branch_winners = self.interBranchChampion(global_champion=True)
            self.ChampionName = final_winner
            
            print("Final winner = ", final_winner)

    def majorityVoteQMD(self, num_runs=1, num_exp=20, max_branches= None, max_num_qubits = None, max_num_models=None, spawn=True, just_given_models=False):

        model_id_list = DataBase.active_model_ids_by_branch_id(self.db, branchID=0) 
        #print("model list : ", model_id_list)
        for i in range(num_runs):
            print("i=", i)
            for j in model_id_list:
                mod = self.getModelInstanceFromID(j)
                mod.resetPrior()
                mod.UpdateModel(n_experiments=num_exp)
            self.compareModelList(model_list=model_id_list, bayes_threshold=1, num_times_to_use=num_exp)
        self.MajorityVotingScores = self.majorityVotingTally()

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
    n = num_models
    k = 2 # ie. nCk where k=2 since we want pairs
    
    a= math.factorial(n) / math.factorial(k)
    b= math.factorial(n-k)
    
    return a/b

        


        
def get_exps(model, gen, times):

    exps = np.empty(len(times), dtype=gen.expparams_dtype)
    exps['t'] = times

    for i in range(1, len(gen.expparams_dtype)):
        col_name = 'w_'+str(i)
        exps[col_name] = model.FinalParams[i-1,0] ## TODO: should be model.NewEval[i-1]???
    return exps

def get_exp(model, gen, time):
    # same as get_exps but for a single experiment. 
    # For use in alternative to batch_update during Bayes update.
    exp = np.empty(len(time), dtype=gen.expparams_dtype)
    exp['t'] = time

    for i in range(1, len(gen.expparams_dtype)):
        col_name = 'w_'+str(i)
        exp[col_name] = model.FinalParams[i-1,0] ## TODO: should be model.NewEval[i-1]???
    return exp


def get_log_likelihood(model, gen, exps, times):
    import copy
    #print("Creating Copy of updater for", model.Name)
    #print("Before copy/update, gen._a=", gen._a)
    updater = copy.deepcopy(model.Updater) #this could be what Updater.hypotheticalUpdate is for?
    
    #print("Generating data")
    #data = updater.model.simulate_experiment(model.SimParams, exps)[0][0]
    #print("Data generated")
    
    #TODO: do update in loop instead of batch update
    #updater.model._a -= 1
    #print("Taken one from _a :" , updater.model._a)

    batch=False
    hypothetical_update = False
    if hypothetical_update:
      model.Updater.hypothetical_update(data, exps)
      likelihood = model.Updater.log_total_likelihood
      # print("Likelihood = ", likelihood)
      return likelihood

    else:
      if batch:
        updater.batch_update(data, exps, resample_interval=100)
    
      else:
        for i in range(len(times)):
          #print("i=", i)
          exp = get_exp(model, gen, [times[i]])
          params_array = np.array([[model.FinalParams[0][0]]])
          
          datum = updater.model.simulate_experiment(params_array, exp)
          #updater.model._a+=1
#          datum = data[i]
          #print("Updating copy")
          updater.update(datum, exp)

      log_likelihood = updater.log_total_likelihood
      del updater
      return log_likelihood        


def alternative_log_total_likelihood(model, gen, times):
    import copy 
    exps= get_exps(model, gen, times)
    data = gen.simulate_experiment(model.FinalParams, exps)[0][0]

    norms = copy.deepcopy(model.Updater.normalization_record)
    #print("Norms before : ", norms)
    for idx in range(len(times)):
        datum = np.array([data[idx]])
        exp = np.array([exps[idx]])
        #print("For time ", times[idx], " datum=", datum, " exp= ", exp)
        normalization = model.Updater.hypothetical_update(datum, exp, return_normalization=True)[1]
        norms.append(normalization)
    
    #print("norms after : ", norms)
    return np.sum(np.log(np.array(norms)))

def compute_bayes_factor(model_a, model_b, times_a, times_b, only_ideal_probes=False):
    if model_a.Dimension == model_b.Dimension and only_ideal_probes:
        # models have same dimension so may be locally equivalent (??)
        if model_a.Updater.log_total_likelihood > model_b.Updater.log_total_likelihood:
            ideal = model_a.Operator.ideal_probe
            #if ideal in model_b.Operator.eigenvectors:
             #   print("Chosen ideal probe (model", model_a.Name, ") is an eigenvector of model", model_b.Name)
        else: 
            ideal = model_b.Operator.ideal_probe
            #if ideal in model_a.Operator.eigenvectors:
             #   print("Chosen ideal probe (model", model_b.Name, ") is an eigenvector of model", model_a.Name)
        
        ideal_probes_list = [model_a.Operator.ideal_probe, model_b.Operator.ideal_probe]
        
        print("Updating ", model_a.Name, " with times of ", model_b.Name)
#        log_l_a = log_likelihood_given_probelist(model_a, times_b, ideal_probes_list)
        log_l_a = log_likelihood_given_probe(model_a, times_b, ideal)

        print("Updating ", model_b.Name, " with times of ", model_a.Name)
#        log_l_b = log_likelihood_given_probelist(model_b, times_a, ideal_probes_list)
        log_l_b = log_likelihood_given_probe(model_b, times_a, ideal)

    else:
        log_l_a = log_likelihood_general(model_a, times_b)
        log_l_b = log_likelihood_general(model_b, times_a)

    bayes_factor = np.exp(log_l_a - log_l_b)
    return bayes_factor
    

def log_likelihood_given_probe(model, times, ideal_probe):
    import copy
    updater = copy.deepcopy(model.Updater)
    updater.model.inBayesUpdates = True # updater.model is our gen
    updater.model.ideal_probe = ideal_probe
    for i in range(len(times)):
    
        exp = get_exp(model, updater.model, [times[i]])
        params_array = np.array([[model.FinalParams[0][0]]])
        datum = updater.model.simulate_experiment(params_array, exp)
        updater.update(datum, exp)

    log_likelihood = updater.log_total_likelihood
    del updater
    return log_likelihood        

def log_likelihood_given_probelist(model, times, ideal_probes):
    import copy
    updater = copy.deepcopy(model.Updater)
    updater.model.inBayesUpdates = True # updater.model is our gen
    updater.model.ideal_probelist = ideal_probes
    for i in range(len(times)):
    
        exp = get_exp(model, updater.model, [times[i]])
        params_array = np.array([[model.FinalParams[0][0]]])
        datum = updater.model.simulate_experiment(params_array, exp)
        updater.update(datum, exp)

    log_likelihood = updater.log_total_likelihood
    del updater
    return log_likelihood        

def log_likelihood_general(model, times):
    import copy
    updater = copy.deepcopy(model.Updater) #this could be what Updater.hypotheticalUpdate is for?
    
    for i in range(len(times)):
    
        exp = get_exp(model, updater.model, [times[i]])
        params_array = np.array([[model.FinalParams[0][0]]])
        datum = updater.model.simulate_experiment(params_array, exp)
        updater.update(datum, exp)

    log_likelihood = updater.log_total_likelihood
    del updater
    return log_likelihood        
    




def separable_probe_dict(max_num_qubits, num_probes):
    seperable_probes = {}
    for i in range(num_probes):
        seperable_probes[i,0] = random_probe(1)
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = np.tensordot(seperable_probes[i,j-1], random_probe(1), axes=0).flatten(order='c')
            if np.linalg.norm(seperable_probes[i,j]) < 0.999999999 or np.linalg.norm(seperable_probes[i,j]) > 1.0000000000001:
                print("non-unit norm: ", np.linalg.norm(seperable_probes[i,j]))
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
    if np.linalg.norm(probe) -1  > 1e-10:
        print("Probe not normalised. Norm factor=", np.linalg.norm(probe)-1)
    return probe
