from __future__ import print_function # so print doesn't show brackets
import numpy as np
import scipy as sp
import os 
import time
import qinfer as qi
import Evo as evo
import Distrib as Distributions
import GenSimQMD_IQLE as gsi
import ExperimentalDataFunctions as expdt
import multiPGH as mpgh
import DataBase as DB
from MemoryTest import print_loc
from psutil import virtual_memory
import RedisSettings as rds
import PlotQMD 
import redis
import pickle
import matplotlib.pyplot as plt
pickle.HIGHEST_PROTOCOL=2

global debug_print
global print_mem_status
global debug_log_print
debug_log_print = True
debug_print = False
print_mem_status = True
global_print_loc = False


"""
In this file are class definitions:
    - ModelLearningClass
    - reducedModel
    - modelClassForRemoteBayesFactor

"""

def time_seconds():
    # return time in h:m:s format for logging. 
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time



class ModelLearningClass():
    """
    Class to learn individual model. Model name is given when initialised. 
    A host_name and port_number are given to InitialiseNewModel. 
    The qmd_info dict from Redis is pulled and pickled to find
    the true model and other QMD parameters needed. 
    A GenSimModel is set which details the SMCUpdater
    used to update the posterior distribution. 
    UpdateModel calls the updater in a loop of n_experiments. 
    The final parameter estimates are set as the mean of the
    posterior distribution after n_experiments wherein n_particles 
    are sampled per experiment (set in qmd_info). 
    
    """

    def __init__(self, name, num_probes=20, probe_dict=None, qid=0,
        log_file='QMD_log.log', modelID=0
    ):
        self.VolumeList = np.array([])  
        self.Name = name
        self.LatexTerm = DB.latex_name_ising(self.Name)
        self.Dimension = DB.get_num_qubits(name)
        self.NumExperimentsToDate = 0
        self.BayesFactors = {}
        self.log_file = log_file
        self.Q_id = qid
        self.ModelID = int(modelID)
 
    
    def log_print(self, to_print_list):
        identifier = str(str(time_seconds()) +" [QML "+ str(self.ModelID) +"]")
        if type(to_print_list)!=list:
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file)


    
    def InitialiseNewModel(self, 
        trueoplist,
        modeltrueparams,
        simoplist, 
        simparams, simopnames, numparticles, modelID, 
        use_time_dep_true_params=False, 
        time_dep_true_params=None,
        resample_thresh=0.5, resampler_a = 0.95, pgh_prefactor = 1.0,
        store_partices_weights=False, checkloss=True, gaussian=True,
        use_exp_custom=True, enable_sparse=True,
        debug_directory=None, qle=True, 
        host_name='localhost', 
        port_number=6379, qid=0, log_file='QMD_log.log'
    ):
       
        self.log_print(["QID=", qid])
        rds_dbs = rds.databases_from_qmd_id(host_name, port_number, qid)
        qmd_info_db = rds_dbs['qmd_info_db'] 
        init_model_print_loc = False
        qmd_info = pickle.loads(qmd_info_db.get('QMDInfo'))
        self.ProbeDict = pickle.loads(qmd_info_db['ProbeDict'])
        self.NumParticles = qmd_info['num_particles']
        self.NumProbes = qmd_info['num_probes']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.StoreParticlesWeights = qmd_info['store_particles_weights']
        self.QHL_plots = qmd_info['qhl_plots']
        self.ResultsDirectory = qmd_info['results_directory']
        self.TrueOpList = qmd_info['true_oplist']
        self.TrueParams = qmd_info['true_params']
        self.TrueOpName  = qmd_info['true_name']
        self.UseTimeDepTrueModel = qmd_info['use_time_dep_true_params']
        self.TimeDepTrueParams = qmd_info['time_dep_true_params']
        self.NumTimeDepTrueParams = qmd_info['num_time_dependent_true_params']
        self.QLE = qmd_info['qle']
        self.UseExpCustom = qmd_info['use_exp_custom']
        self.ExpComparisonTol = qmd_info['compare_linalg_exp_tol']
        self.UseExperimentalData = qmd_info['use_experimental_data']
        self.ExperimentalMeasurements = qmd_info['experimental_measurements']
        self.ExperimentalMeasurementTimes = qmd_info['experimental_measurement_times']
        self.SimOpsNames = simopnames
        print_loc(print_location=init_model_print_loc)
        self.log_print(["learning true params:", self.TrueParams])
        
        self.SimOpList  = np.asarray(simoplist)
        self.SimParams = np.asarray([simparams[0]])
        self.EnableSparse = enable_sparse
        self.checkQLoss = True
        print_loc(print_location=init_model_print_loc)
        """        
        self.log_print(['True params', self.TrueParams, '\n true op list:',
            self.TrueOpList, 'true op name:', self.TrueOpName]
        )
        self.log_print(['SimOpsNames:', self.SimOpsNames, 
            '\n\tSimOpList:\n', self.SimOpList, 
            '\n\t SimParams:', self.SimParams]
        )
        """    
        if debug_directory is not None: 
            self.debugSave = True
            self.debugDirectory = debug_directory 
        else:            
            self.debugSave = False
        num_params = len(self.SimOpList)
        if gaussian:
            # Use a normal distribution
            self.log_print(["Normal distribution generated"])
            means = self.TrueParams[0:num_params]
            if num_params > len(self.TrueParams):
                for i in range(len(self.TrueParams), num_params):
                    means.append(self.TrueParams[i%len(self.TrueParams)])
#            self.Prior = Distributions.MultiVariateNormalDistributionNocov(num_params)
            
            self.PriorSpecificTerms = qmd_info['prior_specific_terms']
            
            if (
                qmd_info['model_priors'] is not None
                and 
                DB.alph(self.Name) in list(qmd_info['model_priors'].keys()) 
            ):
                self.PriorSpecificTerms = (
                    qmd_info['model_priors'][DB.alph(self.Name)]
                )

            self.Prior = Distributions.normal_distribution_ising(
                term = self.Name,
                specific_terms = self.PriorSpecificTerms
            )
        else:
            self.log_print(["Uniform distribution generated"])
 
            self.Prior = Distributions.uniform_distribution_ising(
                term = self.Name
            )
    
#            self.Prior = Distributions.MultiVariateUniformDistribution(num_params) #the prior distribution is on the model we want to test i.e. the one implemented in the simulator
	  
        log_identifier=str("QML "+str(self.ModelID))
        self.GenSimModel = gsi.GenSimQMD_IQLE(
            oplist=self.SimOpList, modelparams=self.SimParams, 
            true_oplist=self.TrueOpList, trueparams=self.TrueParams,
            truename=self.TrueOpName, 
            use_time_dep_true_model = self.UseTimeDepTrueModel,
            time_dep_true_params = self.TimeDepTrueParams,
            num_time_dep_true_params = self.NumTimeDepTrueParams,
            num_probes=self.NumProbes,
            use_experimental_data = self.UseExperimentalData,
            experimental_measurements = self.ExperimentalMeasurements,
            experimental_measurement_times=self.ExperimentalMeasurementTimes, 
            probe_dict=self.ProbeDict, probecounter=0, solver='scipy',
            trotter=True, qle=self.QLE, use_exp_custom=self.UseExpCustom,
            exp_comparison_tol = self.ExpComparisonTol,
            enable_sparse=self.EnableSparse, model_name=self.Name,
            log_file=self.log_file, log_identifier=log_identifier
        ) 

        self.Updater = qi.SMCUpdater(self.GenSimModel, self.NumParticles,
            self.Prior, resample_thresh=self.ResamplerThresh , 
            resampler=qi.LiuWestResampler(a=self.ResamplerA),
            debug_resampling=False
        )

        self.Inv_Field = [
            item[0] for item in self.GenSimModel.expparams_dtype[1:] 
        ]
        self.Heuristic = mpgh.multiPGH(self.Updater, inv_field=self.Inv_Field)
        
        if checkloss == True or self.checkQLoss==True:     
            self.QLosses = np.array([])
        self.TrackLogTotLikelihood = np.array([])
        self.TrackTime = np.array([]) #only for debugging
        self.Particles = np.array([])
        self.Weights = np.array([])
        self.ResampleEpochs = []
        self.Experiment = self.Heuristic()   
        self.ExperimentsHistory = np.array([])
        self.FinalParams = np.empty([len(self.SimOpList),2]) #average and standard deviation at the final step of the parameters inferred distributions
        print_loc(print_location=init_model_print_loc)
        self.log_print(['Initialization Ready'])

    def UpdateModel(self, n_experiments, sigma_threshold=10**-13,
        checkloss=True
    ):
        self.NumExperiments = n_experiments
        if self.checkQLoss == True: 
            self.QLosses = np.empty(n_experiments)
        self.Covars= np.empty(n_experiments)
        self.TrackEval = []
        self.TrackCovMatrices = []
        self.TrackTime =np.empty(n_experiments)#only for debugging
    
        self.Particles = np.empty([self.NumParticles, 
            len(self.SimParams[0]), self.NumExperiments]
        )
        self.Weights = np.empty([self.NumParticles, self.NumExperiments])
        self.DistributionMeans = np.empty([self.NumExperiments])
        self.DistributionStdDevs = np.empty([self.NumExperiments])
        
        self.Experiment = self.Heuristic()    
        self.SigmaThresh = sigma_threshold   #This is the value of the Norm of the COvariance matrix which stops the IQLE 
        self.LogTotLikelihood=[] #log_total_likelihood

        self.datum_gather_cumulative_time = 0
        self.update_cumulative_time = 0
        
        for istep in range(self.NumExperiments):
            self.Experiment =  self.Heuristic()
            print_loc(global_print_loc)
            self.Experiment[0][0] = self.Experiment[0][0] * self.PGHPrefactor
            if self.UseExperimentalData:
                t = self.Experiment[0][0]
                nearest = expdt.nearestAvailableExpTime(
                    times=self.ExperimentalMeasurementTimes,
                    t=t
                )
                self.Experiment[0][0] = nearest
            
            self.NumExperimentsToDate += 1
            print_loc(global_print_loc)
            if istep == 0:
                print_loc(global_print_loc)
                self.log_print(['Initial time selected > ',
                    str(self.Experiment[0][0])]
                )
            
            self.TrackTime[istep] = self.Experiment[0][0]
            true_params = np.array([[self.TrueParams[0]]])
            
            before_datum = time.time()

#            self.log_print(
#                [
#                'Getting Datum'
#                ]
#            )

            self.Datum = self.GenSimModel.simulate_experiment(
                self.SimParams,
                self.Experiment,
                repeat=1
            ) # TODO reconsider repeat number
            after_datum = time.time()
            self.datum_gather_cumulative_time+=after_datum-before_datum
            
            exp_t = self.Experiment[0][0]
            before_upd = time.time()
            ## Call updater to update distribution based on datum
            self.Updater.update(self.Datum, self.Experiment)
            after_upd = time.time()
            self.update_cumulative_time+=after_upd-before_upd
            
            if self.Updater.just_resampled:
                self.ResampleEpochs.append(istep)
            
            print_loc(global_print_loc)

            if len(self.Experiment[0]) < 3:
                print_loc(global_print_loc)
                self.covmat = self.Updater.est_covariance_mtx()
                self.VolumeList = np.append(self.VolumeList, self.covmat)
                print_loc(global_print_loc)

            else:
                print_loc(global_print_loc)
                self.covmat = self.Updater.est_covariance_mtx()
                self.VolumeList = np.append(self.VolumeList,  
                    np.linalg.det( sp.linalg.sqrtm(self.covmat) )
                )
                print_loc(global_print_loc)
            
#            if istep%50==0:
#                self.log_print(['Step', istep, '\t Mean:', self.Updater.est_mean()])
            self.TrackEval.append(self.Updater.est_mean())
            self.TrackCovMatrices.append(self.Updater.est_covariance_mtx())

            print_loc(global_print_loc)
            self.Covars[istep] = np.linalg.norm(
                self.Updater.est_covariance_mtx()
            )
            print_loc(global_print_loc)
            self.Particles[:, :, istep] = self.Updater.particle_locations
            #self.Weights[:, istep] = self.Updater.particle_weights

            self.NewEval = self.Updater.est_mean()
            print_loc(global_print_loc)


            #TODO this won't work -- what does iterator mean??
#            self.DistributionMeans[istep] = self.Updater.est_mean()
#            self.DistributionStdDevs[istep] = 
                
            if checkloss == True: 
                if False: # can be reinstated to stop learning when volume converges
                    if self.debugSave: 
                        self.debug_store()
                    self.log_print(['Final time selected > ',
                        str(self.Experiment[0][0])]
                    )
                    print('Exiting learning for Reaching Num. Prec. \
                         -  Iteration Number ' + str(istep)
                    )
                    for iterator in range(len(self.FinalParams)):
                        self.FinalParams[iterator]= [
                            #np.mean(self.Particles[:,iterator,istep]), 
                            self.Updater.est_mean(),
                            np.std(self.Particles[:,iterator,istep])
                        ]
                        print('Final Parameters mean and stdev:'+
                            str(self.FinalParams[iterator])
                        ) 
                    self.LogTotLikelihood=(
                        self.Updater.log_total_likelihood                
                    )
                    self.QLosses=(np.resize(self.QLosses, (1,istep)))[0]
                    self.Covars=(np.resize(self.Covars, (1,istep)))[0]
                    self.Particles = self.Particles[:, :, 0:istep]
                    self.Weights = self.Weights[:, 0:istep]
                    self.TrackTime = self.TrackTime[0:istep] 
                    break 
            
            if self.Covars[istep]<self.SigmaThresh and False: 
                # can be reinstated to stop learning when volume converges
                if self.debugSave: 
                    self.debug_store()
                self.log_print(['Final time selected > ',
                    str(self.Experiment[0][0])]
                )
                self.log_print(['Exiting learning for Reaching Cov. \
                    Norm. Thrshold of ', str(self.Covars[istep])]
                )
                self.log_print([' at Iteration Number ' , str(istep)]) 
                for iterator in range(len(self.FinalParams)):
                    self.FinalParams[iterator]= [
#                        np.mean(self.Particles[:,iterator,istep]), 
                        self.Updater.est_mean(),
                        np.std(self.Particles[:,iterator,istep])
                    ]
                    self.log_print(['Final Parameters mean and stdev:',
                        str(self.FinalParams[iterator])]
                    )
                self.LogTotLikelihood=self.Updater.log_total_likelihood
                if checkloss == True: 
                    self.QLosses=(np.resize(self.QLosses, (1,istep)))[0]
                
                self.Covars=(np.resize(self.Covars, (1,istep)))[0]
                self.Particles = self.Particles[:, :, 0:istep]
                self.Weights = self.Weights[:, 0:istep]
                self.TrackTime = self.TrackTime[0:istep]
                
                break 
            
            if istep == self.NumExperiments-1:
                self.log_print(["Results for QHL on ", self.Name])
                self.log_print(['Final time selected >',
                    str(self.Experiment[0][0])]
                )
                self.LogTotLikelihood=self.Updater.log_total_likelihood
                #from pympler import asizeof
                self.log_print(['Cumulative time.\t Datum:',
                    self.datum_gather_cumulative_time, '\t Update:',
                    self.update_cumulative_time]
                )
        
                #self.log_print(['Sizes:\t updater:', asizeof.asizeof(self.Updater), '\t GenSim:', asizeof.asizeof(self.GenSimModel) ])
                if self.debugSave: 
                    self.debug_store()
                
                self.LearnedParameters = {}
                for iterator in range(len(self.FinalParams)):
                    self.FinalParams[iterator]= [
#                        np.mean(self.Particles[:,iterator,istep-1]), 
                        self.Updater.est_mean()[iterator],
                        np.std(self.Particles[:,iterator,istep-1])
                    ]
                    self.log_print([
                        'Final Parameters mean and stdev (term ',
                        self.SimOpsNames[iterator] , '):',
                        str(self.FinalParams[iterator])]
                    )
                    self.LearnedParameters[self.SimOpsNames[iterator]] = (
                        self.FinalParams[iterator][0]
                    )
#                plt.savefig(posterior_plot,'posterior.png')
            

            if debug_print:
                self.log_print(["step ", istep])
                self.log_print( ["has params: ", self.NewEval])
                self.log_print(["log total likelihood:",
                    self.TrackLogTotLikelihood[-1]]
                )


    def resetPrior(self):
        self.Updater.prior = self.Prior
        self.Updater = qi.SMCUpdater(self.GenSimModel,
            self.NumParticles, self.Prior, resample_thresh=self.ResamplerThresh,
            resampler = qi.LiuWestResampler(a=self.ResamplerA),
            debug_resampling=False
        )
        self.Heuristic = mpgh.multiPGH(self.Updater, 
            self.SimOpList, inv_field=self.Inv_Field
        )
        return 1
        
        
    def learned_info_dict(self):
        """
        Place essential information after learning has occured into a dict. 
        This can be used to recreate the model on another node. 
        
        """
        learned_info = {}
        learned_info['times'] = self.TrackTime
        learned_info['final_params'] = self.FinalParams
        learned_info['normalization_record'] = self.Updater.normalization_record
        learned_info['log_total_likelihood'] = self.Updater.log_total_likelihood
        learned_info['data_record'] = self.Updater.data_record
        learned_info['name'] = self.Name
        learned_info['model_id'] = self.ModelID
        learned_info['final_prior'] = self.Updater.prior # TODO regenerate this from mean and std_dev instead of saving it
        learned_info['initial_params'] = self.SimParams
        learned_info['volume_list'] = self.VolumeList
        learned_info['track_eval'] = self.TrackEval
        learned_info['track_cov_matrices'] = self.TrackCovMatrices
        learned_info['resample_epochs'] = self.ResampleEpochs
        learned_info['quadratic_losses'] = self.QLosses
        learned_info['learned_parameters'] = self.LearnedParameters
        learned_info['cov_matrix'] = self.covmat
        if self.StoreParticlesWeights or self.QHL_plots:
            learned_info ['particles'] = self.Particles
            learned_info['weights'] = self.Weights

        return learned_info
        
        
    
    def UpdateKLogTotLikelihood(self, epoch, tpool, stepnum):
        # Calcalate total log likelihood when the model finishes, compared with all previously completed but still active models. 
        
        mytpool = np.setdiff1d(tpool, self.TrackTime[-stepnum-1:-1])
        
        self.TrackLogTotLikelihood = np.append(
            self.TrackLogTotLikelihood, LogL_UpdateCalc(self, tpool)
        )


    def addBayesFactor(self, compared_with, bayes_factor):
        if compared_with in self.BayesFactors: 
            self.BayesFactors[compared_with].append(bayes_factor)
        else: 
            self.BayesFactors[compared_with] = [bayes_factor]
            
            
    def store_particles(self, debug_dir=None):
        if debug_dir is not None: 
            save_dir = debug_dir
        elif self.debugDirectory is not None: 
            save_dir = self.debugDirectory
        else: 
            self.log_print([
                "Need to pass debug_dir to QML.debug_save function"]
            )
            return False            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)            
        
        save_file =  save_dir+'/particles_mod_' +str(self.ModelID)+'.dat'
        
        particle_file = open(save_file, 'w')
        particle_file.write("\n".join(str(elem) for elem in self.Particles.T))
        particle_file.close()
        
    def store_covariances(self, debug_dir=None):
        if debug_dir is not None: 
            save_dir = debug_dir
        elif self.debugDirectory is not None: 
            save_dir = self.debugDirectory
        else: 
            self.log_print(["Need to pass debug_dir to QML.debug_save function"])
            return False            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)            
        
        save_file =  save_dir+'/covariances_mod_' +str(self.ModelID)+'.dat'
        particle_file = open(save_file, 'w')
        particle_file.write("\n".join(str(elem) for elem in self.Covars))
        particle_file.close()
        
    def debug_store(self, debug_dir=None): ## Adjust what gets stored here
        self.store_particles(debug_dir=debug_dir)
        self.store_covariances(debug_dir=debug_dir)
        
    def plotDistributionProgression(self, 
        renormalise=False, 
        save_to_file=None
    ):
        PlotQMD.plotDistributionProgressionQML(
            mod = self,
            num_steps_to_show = 2, 
            show_means = True,
            renormalise = renormalise,
            save_to_file = save_to_file
        )           
        
        
        
### Reduced class with only essential information saved ###
class reducedModel():
    """
    Class holds what is required for updates only. 
    i.e. 
        - times learned over
        - final parameters
        - oplist
        - true_oplist (?) needed to regenerate GenSimModel identically (necessary?)
        - true_params (?)
        - resample_thresh
        - resample_a [are resampling params needed only for updates?]
        - Prior (specified by mean and std_dev?)
    
    Then initialises an updater and GenSimModel which are used for updates. 
    """
    def __init__(
        self, 
        model_name, sim_oplist, 
        true_oplist, true_params,
        numparticles, modelID, 
        resample_thresh=0.5, resample_a=0.9, qle=True,
        probe_dict= None, qid=0,
        host_name='localhost', port_number=6379,
        log_file='QMD_log.log'
    ):

        rds_dbs = rds.databases_from_qmd_id(host_name, port_number, qid)
        qmd_info_db = rds_dbs['qmd_info_db'] 
        #print("In reduced model. rds_dbs:", rds_dbs)
      #  print("QMD INFO DB has type", type(qmd_info_db), "\n", qmd_info_db)
        
        self.Name = model_name
        self.ModelID = modelID
        self.SimOpList = sim_oplist
        self.ModelID = modelID
        qmd_info = pickle.loads(qmd_info_db.get('QMDInfo'))
        self.ProbeDict = pickle.loads(qmd_info_db['ProbeDict'])
        self.ExperimentalMeasurements = qmd_info['experimental_measurements']
        self.NumParticles = qmd_info['num_particles']
        self.NumExperiments = qmd_info['num_experiments']
        self.NumProbes = qmd_info['num_probes']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.TrueOpList = qmd_info['true_oplist']
        self.TrueParams = qmd_info['true_params']
        self.TrueOpName  = qmd_info['true_name']
        self.QLE = qmd_info['qle']
        self.UseExpCustom = qmd_info['use_exp_custom']
        self.StoreParticlesWeights = qmd_info['store_particles_weights']
        self.BayesFactors = {}
        self.LatexTerm = DB.latex_name_ising(self.Name)
        self.HostName = host_name
        self.PortNumber = port_number
        self.Q_id = qid
        self.log_file = log_file
        self.expectation_values = {}
        
        
    def log_print(self, to_print_list):
        identifier = str(str(time_seconds()) +
            "[QML:Reduced "+ str(self.ModelID) +"; QMD "+str(self.Q_id)+"]"
        )
        if type(to_print_list)!=list:
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file)
    
        
        
    def updateLearnedValues(self, learned_info=None):
        """
        Pass a dict, learned_info, with essential info on 
        reconstructing the state of the model, updater and GenSimModel
        
        """

        rds_dbs = rds.databases_from_qmd_id(
            self.HostName, self.PortNumber, self.Q_id
        )
        learned_models_info = rds_dbs['learned_models_info']

        if learned_info is None:
            model_id_float = float(self.ModelID)
            model_id_str = str(model_id_float)
            try:
                learned_info = pickle.loads(
                    learned_models_info.get(model_id_str), 
                    encoding='latin1'
                ) # TODO telling pickle which encoding was used, though I'm not sure why/where that encoding was given...        
            except:
                print("model_id_str: ", model_id_str)
                print("model id: ", self.ModelID)
                print("learned info keys:, ", learned_models_info.keys())
                print("learned info:, ", learned_models_info.get(model_id_str))

        self.Times = learned_info['times']
        self.FinalParams = learned_info['final_params'] # should be final params from learning process
        self.SimParams_Final = np.array([[self.FinalParams[0,0]]]) # TODO this won't work for multiple parameters
        self.Prior = learned_info['final_prior'] # TODO this can be recreated from finalparams, but how for multiple params?
        self._normalization_record = learned_info['normalization_record']
        self.log_total_likelihod = learned_info['log_total_likelihood']
        self.VolumeList = learned_info['volume_list'] 
        self.TrackEval = np.array(learned_info['track_eval'])
        self.TrackCovMatrices = np.array(learned_info['track_cov_matrices'])
        self.ResampleEpochs = learned_info['resample_epochs']
        self.QuadraticLosses = learned_info['quadratic_losses']
        self.LearnedParameters = learned_info['learned_parameters']
        self.cov_matrix = learned_info['cov_matrix']


        self.TrackParameterEstimates = {}
        num_params = np.shape(self.TrackEval)[1]
        max_exp = np.shape(self.TrackEval)[0] -1
        for i in range(num_params):
            for term in self.LearnedParameters.keys():
                if self.LearnedParameters[term] == self.TrackEval[max_exp][i]:
                    self.TrackParameterEstimates[term] = self.TrackEval[:,i]


        try:
            self.Particles = np.array(learned_info['particles'])
            self.Weights = np.array(learned_info['weights'])
        except:
            self.Particles = 'Particles not stored.'
            self.Weights = 'Weights not stored.'
        
        sim_params = list(self.FinalParams[:,0])
        self.LearnedHamiltonian = np.tensordot(
            sim_params, 
            self.SimOpList, 
            axes=1
        )


    def compute_expectation_values(
        self, 
        times = [],
        probe = np.array([0.5, 0.5, 0.5, 0.5+0j])
    ):
        for t in times:
            self.expectation_values[t] = evo.hahn_evolution(
                ham = self.LearnedHamiltonian, 
                t = t,
                state = probe
            )


    def r_squared(
        self, 
        min_time = 0,
        max_time = None 
    ):
        exp_times = sorted(list(self.ExperimentalMeasurements.keys()))
        if max_time is None:
            max_time = max(exp_times)

        min_time = expdt.nearestAvailableExpTime(exp_times, min_time)
        max_time = expdt.nearestAvailableExpTime(exp_times, max_time)
        min_data_idx = exp_times.index(min_time)
        max_data_idx = exp_times.index(max_time)
        exp_times = exp_times[min_data_idx:max_data_idx]
        exp_data = [
            self.ExperimentalMeasurements[t] for t in exp_times
        ]
        probe = np.array([0.5, 0.5, 0.5, 0.5+0j]) # TODO generalise

        datamean = np.mean(exp_data[0:max_data_idx])
        datavar = np.sum( (exp_data[0:max_data_idx] - datamean)**2  )

        ham = self.LearnedHamiltonian

        #print(exp_times)
        sum_of_residuals = 0
        
        available_expectation_values = sorted(list(self.expectation_values.keys()))
        self.r_squared_of_t = {}
        for t in exp_times:
            # TODO if use_experimental_data is False, call full expectatino value function isntead
            if t in available_expectation_values:
                sim = self.expectation_values[t]
            else:
                sim = evo.hahn_evolution(ham, t, probe)
#                print("[r^2 ", self.Name, "] t=",t,":\t", sim)
                self.expectation_values[t] = sim

            true = self.ExperimentalMeasurements[t]
            diff_squared = (sim - true)**2
            sum_of_residuals += diff_squared
            self.r_squared_of_t[t] = 1 - sum_of_residuals/datavar

        Rsq = 1 - sum_of_residuals/datavar
        
        return Rsq


    def r_squared_by_epoch(
        self, 
        min_time=0,
        max_time=None,
        num_points=10 # maximum number of epochs to take R^2 at
    ):
        exp_times = sorted(list(self.ExperimentalMeasurements.keys()))
        if max_time is None:
            max_time = max(exp_times)

        min_time = expdt.nearestAvailableExpTime(exp_times, min_time)
        max_time = expdt.nearestAvailableExpTime(exp_times, max_time)
        min_data_idx = exp_times.index(min_time)
        max_data_idx = exp_times.index(max_time)
        exp_times = exp_times[min_data_idx:max_data_idx]
        exp_data = [
            self.ExperimentalMeasurements[t] for t in exp_times
        ]
        probe = np.array([0.5, 0.5, 0.5, 0.5+0j]) # TODO generalise

        datamean = np.mean(exp_data[0:max_data_idx])
        datavar = np.sum( (exp_data[0:max_data_idx] - datamean)**2  )
        r_squared_by_epoch =  {}
        spaced_epochs = np.round(
            np.linspace(
                0, 
                self.NumExperiments-1, 
                min(self.NumExperiments, num_points))
        )
        
        for e in spaced_epochs:

            ham = np.tensordot(
                self.TrackEval[int(e)], 
                self.SimOpList, 
                axes=1
            ) # the Hamiltonian this model held at epoch e
            sum_of_residuals = 0
            available_expectation_values = sorted(
                list(self.expectation_values.keys())
            )
            for t in exp_times:
                sim = evo.hahn_evolution(ham, t, probe)
                true = self.ExperimentalMeasurements[t]
                diff_squared = (sim - true)**2
                sum_of_residuals += diff_squared

            Rsq = 1 - sum_of_residuals/datavar
            r_squared_by_epoch[e] = Rsq

        return r_squared_by_epoch


    
#        self.GenSimModel = gsi.GenSimQMD_IQLE(oplist=self.SimOpList, modelparams=self.SimParams_Final, true_oplist = self.TrueOpList, trueparams = self.TrueParams, truename=self.TrueOpName,             use_experimental_data = self.UseExperimentalData,
#            experimental_measurements = self.ExperimentalMeasurements,
#            experimental_measurement_times=(
#                self.ExperimentalMeasurementTimes
#            ),             
#model_name=self.Name, probe_dict = self.ProbeDict)    # probelist=self.TrueOpList,
#        self.Updater = qi.SMCUpdater(self.GenSimModel, self.NumParticles, self.Prior, resample_thresh=self.ResamplerThresh , resampler = qi.LiuWestResampler(a=self.ResamplerA), debug_resampling=False) ## TODO does the reduced model instance need an updater or GenSimModel?
#        self.Updater._normalization_record = self._normalization_record
 
        
        
        
        
        
class modelClassForRemoteBayesFactor():
    """
    When Bayes factors are calculated remotely (ie on RQ workers), 
    they require SMCUpdaters etc to do calculations. 
    This class captures the minimum required to enable these calculations. 
    These are pickled by the ModelLearningClass to a redis database: 
    this class unpickles the useful information and generates new instances 
    of GenSimModel etc. to use in those calculations. 
    
    """


    def __init__(
            self,
            modelID,
            host_name='localhost',
            port_number=6379,
            qid=0,
            log_file='QMD_log.log'
        ):

        rds_dbs = rds.databases_from_qmd_id(host_name, port_number, qid)
        qmd_info_db = rds_dbs['qmd_info_db'] 
        learned_models_info = rds_dbs['learned_models_info']
    
        model_id_float = float(modelID)
        model_id_str = str(model_id_float)
        try:
            learned_model_info = pickle.loads(
                learned_models_info.get(model_id_str), encoding='latin1'
            )        
        except:
            learned_model_info = pickle.loads(
                learned_models_info.get(model_id_str)
            )        

        qmd_info = pickle.loads(qmd_info_db.get('QMDInfo'))

        self.ProbeDict = pickle.loads(qmd_info_db['ProbeDict'])
        self.ModelID = modelID
        self.NumParticles = qmd_info['num_particles']
        self.NumProbes = qmd_info['num_probes']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.TrueOpList = qmd_info['true_oplist']
        self.TrueParams = qmd_info['true_params']
        self.TrueOpName  = qmd_info['true_name']
        self.UseExpCustom = qmd_info['use_exp_custom']
        self.UseExperimentalData = qmd_info['use_experimental_data']
        self.ExperimentalMeasurements = qmd_info['experimental_measurements']
        self.ExperimentalMeasurementTimes = qmd_info['experimental_measurement_times']

        self.log_file = log_file
        self.Q_id = qid

        self.Name = learned_model_info['name']
        op = DB.operator(self.Name)
        self.SimOpList = op.constituents_operators # todo, put this in a lighter function
        self.Times = learned_model_info['times']
        self.FinalParams = learned_model_info['final_params'] 
        self.SimParams_Final = np.array([[self.FinalParams[0,0]]]) # TODO this won't work for multiple parameters
        self.InitialParams = learned_model_info['initial_params']
        
        self.Prior = learned_model_info['final_prior'] # TODO this can be recreated from finalparams, but how for multiple params?
        self._normalization_record = learned_model_info['normalization_record']
        self.log_likelihood = learned_model_info['log_total_likelihood']
        
        log_identifier = str("Bayes "+str(self.ModelID)) 
        
                
        self.GenSimModel = gsi.GenSimQMD_IQLE(oplist=self.SimOpList,
            modelparams=self.SimParams_Final, 
            true_oplist = self.TrueOpList,
            trueparams = self.TrueParams, truename=self.TrueOpName,
            use_experimental_data = self.UseExperimentalData,
            experimental_measurements = self.ExperimentalMeasurements,
            experimental_measurement_times=(
                self.ExperimentalMeasurementTimes
            ),             
            model_name=self.Name, num_probes = self.NumProbes, 
            probe_dict=self.ProbeDict, log_file=self.log_file,
            log_identifier=log_identifier
        )    

        self.Updater = qi.SMCUpdater(self.GenSimModel, self.NumParticles,
            self.Prior, resample_thresh=self.ResamplerThresh , 
            resampler=qi.LiuWestResampler(a=self.ResamplerA), 
            debug_resampling=False
        )
        self.Updater._normalization_record = self._normalization_record
        self.Updater.log_likelihood = self.log_likelihood

        #self.GenSimModel = pickle.loads(learned_model_info['gen_sim_model'])
        #self.Updater = pickle.loads(learned_model_info['updater'])
        # TODO not clear which is quicker: generating new instance of classes/updater or unpickling every time.
        del qmd_info, learned_model_info
        
        # could pickle updaters to a redis db for updaters, but first construct these model classes each time a BF is to be computed. 

    def log_print(self, to_print_list):
        identifier = str(str(time_seconds()) +
            "[QML:Bayes "+ str(self.ModelID) +"; QMD "+str(self.Q_id)+"]"
        )
        if type(to_print_list)!=list:
            to_print_list = list(to_print_list)

        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(self.log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file)
        


