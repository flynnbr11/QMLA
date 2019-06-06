from __future__ import print_function # so print doesn't show brackets
import numpy as np
import scipy as sp
import os 
import time
import copy
import qinfer as qi
# import Evo
import ExpectationValues
import GrowthRules
import UserFunctions
import Distrib as Distributions
import GenSimQMD_IQLE as gsi
import ExperimentalDataFunctions as expdt
# import multiPGH as mpgh
import DataBase 
import ModelNames
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


global test_growth_class_implementation
test_growth_class_implementation = True


"""
In this file are class definitions:
    - ModelLearningClass
    - reducedModel
    - modelClassForRemoteBayesFactor

"""


def resource_allocation(
    base_qubits, 
    base_terms, 
    max_num_params, 
    this_model_qubits, 
    this_model_terms, 
    num_experiments, 
    num_particles,
    given_resource_as_cap=True
):
    new_resources = {}
    if given_resource_as_cap == True:
        # i.e. reduce number particles for models with fewer params    
        proportion_of_particles_to_receive = (
            this_model_terms/max_num_params
        )
        print(
            "Model gets proportion of particles:", 
            proportion_of_particles_to_receive
        )

        if proportion_of_particles_to_receive < 1: 
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = max(
                int(
                    proportion_of_particles_to_receive
                    * num_particles
                ),
                10
            )
        else:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = num_particles

    else:
        # increase proportional to number params/qubits 
        qubit_factor = float(this_model_qubits/base_qubits)
        terms_factor = float(this_model_terms/base_terms)
        

        overall_factor = int(qubit_factor * terms_factor)

        if overall_factor > 1: 
            new_resources['num_experiments'] = overall_factor * num_experiments
            new_resources['num_particles'] = overall_factor * num_particles
        else:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = num_particles
        
    print("New resources:", new_resources)
    return new_resources


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

    def __init__(
        self, 
        name, 
        num_probes=20, 
        probe_dict=None, 
        qid=0,
        log_file='QMD_log.log', 
        modelID=0
    ):
        self.VolumeList = np.array([])  
        self.Name = name
        # self.LatexTerm = DataBase.latex_name_ising(self.Name)
        self.Dimension = DataBase.get_num_qubits(name)
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


    
    def InitialiseNewModel(
        self, 
        trueoplist,
        modeltrueparams,
        simoplist, 
        simparams, 
        simopnames, 
        numparticles, 
        modelID, 
        growth_generator,
        use_time_dep_true_params=False, 
        time_dep_true_params=None,
        resample_thresh=0.5, 
        resampler_a = 0.95, 
        pgh_prefactor = 1.0,
        store_partices_weights=False, 
        checkloss=True, 
        gaussian=True,
        use_exp_custom=True, 
        enable_sparse=True,
        debug_directory=None, 
        qle=True, 
        host_name='localhost', 
        port_number=6379, 
        qid=0, 
        log_file='QMD_log.log'
    ):
       
        # self.log_print(["QID=", qid])
        self.log_print(["QML for ", self.Name])
        rds_dbs = rds.databases_from_qmd_id(host_name, port_number, qid)
        qmd_info_db = rds_dbs['qmd_info_db'] 
        init_model_print_loc = False
        qmd_info = pickle.loads(qmd_info_db.get('QMDInfo'))
        self.UseExperimentalData = qmd_info['use_experimental_data']
        self.ProbeDict = pickle.loads(qmd_info_db['ProbeDict'])
        self.NumParticles = qmd_info['num_particles']
        self.NumExperiments = qmd_info['num_experiments']
        self.GrowthGenerator = growth_generator

        try:
            self.GrowthClass = GrowthRules.get_growth_generator_class(
                growth_generation_rule = self.GrowthGenerator,
                use_experimental_data = self.UseExperimentalData
            )
        except:
            # raise
            self.GrowthClass = None


        base_resources = qmd_info['base_resources']
        base_num_qubits = base_resources['num_qubits']
        base_num_terms = base_resources['num_terms']
        this_model_num_qubits = DataBase.get_num_qubits(self.Name)
        this_model_num_terms = len(
            DataBase.get_constituent_names_from_name(self.Name)
        )

        try:
            max_num_params = self.GrowthClass.max_num_parameter_estimate
        except:
            if test_growth_class_implementation == True: raise
            max_num_params = UserFunctions.max_num_parameter_estimate[
                self.GrowthGenerator
            ]

        if qmd_info['reallocate_resources']==True:
            new_resources = resource_allocation(
                base_qubits = base_num_qubits, 
                base_terms = base_num_terms, 
                max_num_params = max_num_params, 
                this_model_qubits = this_model_num_qubits, 
                this_model_terms = this_model_num_terms, 
                num_experiments = self.NumExperiments, 
                num_particles = self.NumParticles
            )

            self.NumExperiments = new_resources['num_experiments']
            self.NumParticles = new_resources['num_particles']
            self.log_print(
                [
                'After resource reallocation, QML on', self.Name, 
                '\n\tParticles:', self.NumParticles, 
                '\n\tExperiments:', self.NumExperiments, 
                ]
            )
        self.NumProbes = qmd_info['num_probes']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.PGHExponent = qmd_info['pgh_exponent']
        self.IncreasePGHTime = qmd_info['increase_pgh_time']
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
        self.PlotTimes = qmd_info['plot_times']
        self.UseExpCustom = qmd_info['use_exp_custom']
        self.ExpComparisonTol = qmd_info['compare_linalg_exp_tol']
        self.MeasurementType = qmd_info['measurement_type']
        self.ExperimentalMeasurements = qmd_info['experimental_measurements']
        self.ExperimentalMeasurementTimes = qmd_info['experimental_measurement_times']
        self.SimOpsNames = simopnames

        try:
            self.LatexTerm = self.GrowthClass.latex_name(
                name = self.Name
            )
        except:
            if test_growth_class_implementation == True: raise
            self.LatexTerm = UserFunctions.get_latex_name(
                name = self.Name,
                growth_generator = self.GrowthGenerator
            )
        
        print_loc(print_location=init_model_print_loc)
        

        self.SimOpList  = np.asarray(simoplist)
        self.SimParams = np.asarray([simparams[0]])

        individual_terms_in_name = DataBase.get_constituent_names_from_name(
            self.Name
        )

        for i in range(len(individual_terms_in_name)):
            term = individual_terms_in_name[i]
            term_mtx = DataBase.compute(term)
            if np.all(term_mtx==self.SimOpList[i]) is False:
                print("[QML] UNEQUAL SIM OP LIST / TERM MATRICES.")
                print("==> INSPECT PRIORS ORDERING.")
                self.log_print(
                    [
                    "Term", term, 
                    "\ncalculated mtx:", term_mtx, 
                    "\nSimOpList:", self.SimOpList[i]
                    ]
                )
            elif term != self.SimOpsNames[i]:
                print("!!! Check log -- QML", self.Name)

                self.log_print(
                    [
                        "term {} != SimOpsNames[i] {}".format(
                            term, self.SimOpsNames[i]
                        )
                    ]
                )
            # else:
            #     print("term:", term)
            #     print("calculated mtx:", term_mtx )
            #     print("sim op list:", self.SimOpList[i] )



        # self.log_print(["True oplist:", self.TrueOpList ])
        # self.log_print(["Sim oplist:", self.SimOpList])
        # self.log_print(["learning true params:", self.TrueParams])
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

        self.PriorSpecificTerms = qmd_info['prior_specific_terms']
#         if gaussian:
#             # Use a normal distribution
#             self.log_print(["Normal distribution generated"])
#             means = self.TrueParams[0:num_params]
#             if num_params > len(self.TrueParams):
#                 for i in range(len(self.TrueParams), num_params):
#                     means.append(self.TrueParams[i%len(self.TrueParams)])
# #            self.Prior = Distributions.MultiVariateNormalDistributionNocov(num_params)
            
#             self.PriorSpecificTerms = qmd_info['prior_specific_terms']
            
#             if (
#                 qmd_info['model_priors'] is not None
#                 and 
#                 DataBase.alph(self.Name) in list(qmd_info['model_priors'].keys()) 
#             ):
#                 self.PriorSpecificTerms = (
#                     qmd_info['model_priors'][DataBase.alph(self.Name)]
#                 )

#             self.Prior = Distributions.normal_distribution_ising(
#                 term = self.Name,
#                 specific_terms = self.PriorSpecificTerms
#             )
#         else:
#             self.log_print(["Uniform distribution generated"])
 
#             self.Prior = Distributions.uniform_distribution_ising(
#                 term = self.Name
#             )
        log_identifier=str("QML "+str(self.ModelID))

        self.Prior = Distributions.get_prior(
            model_name = self.Name, 
            gaussian = gaussian, 
            param_minimum = qmd_info['param_min'],
            param_maximum = qmd_info['param_max'], 
            param_normal_mean = qmd_info['param_mean'], 
            param_normal_sigma = qmd_info['param_sigma'], 
            random_mean=False, 
            specific_terms = self.PriorSpecificTerms, 
            log_file = self.log_file, 
            log_identifier = log_identifier
        )

        prior_dir = str(
            self.ResultsDirectory + 
            'priors/QMD_{}/'.format(self.Q_id)
        )

        if not os.path.exists(prior_dir):
            try:
                os.makedirs(prior_dir)
            except:
                # if already exists (ie created by another QMD since if test ran...)
                pass
        prior_file = str(
            prior_dir + 
            'prior_' + 
            str(self.ModelID) + 
            '.png'
        )

        latex_terms = []
        for term in individual_terms_in_name:
            try:
                lt = self.GrowthClass.latex_name(
                    name = term
                )
            except:
                if test_growth_class_implementation == True: raise
                lt = UserFunctions.get_latex_name(
                    name = term, 
                    growth_generator = self.GrowthGenerator
                )
            latex_terms.append(lt)

        plot_all_priors = False
        if plot_all_priors == True:
            Distributions.plot_prior(
                model_name = self.LatexTerm, 
                model_name_individual_terms = latex_terms, 
                prior = self.Prior, 
                plot_file = prior_file, 
            )

        # # pickle.dump(
        # #     self.Prior, 
        # #     open(
        # #         prior_file,
        # #         'wb'
        # #     )
        # # )

        self.GenSimModel = gsi.GenSimQMD_IQLE(
            oplist=self.SimOpList, 
            modelparams=self.SimParams, 
            true_oplist=self.TrueOpList, 
            trueparams=self.TrueParams,
            truename=self.TrueOpName, 
            use_time_dep_true_model = self.UseTimeDepTrueModel,
            time_dep_true_params = self.TimeDepTrueParams,
            num_time_dep_true_params = self.NumTimeDepTrueParams,
            num_probes=self.NumProbes,
            measurement_type = self.MeasurementType,
            growth_generation_rule = self.GrowthGenerator, 
            use_experimental_data = self.UseExperimentalData,
            experimental_measurements = self.ExperimentalMeasurements,
            experimental_measurement_times=self.ExperimentalMeasurementTimes, 
            probe_dict=self.ProbeDict, probecounter=0, solver='scipy',
            trotter=True, qle=self.QLE, use_exp_custom=self.UseExpCustom,
            exp_comparison_tol = self.ExpComparisonTol,
            enable_sparse=self.EnableSparse, model_name=self.Name,
            log_file=self.log_file, log_identifier=log_identifier
        ) 

        self.Updater = qi.SMCUpdater(
            self.GenSimModel, 
            self.NumParticles,
            self.Prior, 
            resample_thresh=self.ResamplerThresh , 
            resampler=qi.LiuWestResampler(a=self.ResamplerA),
            debug_resampling=False
        )

        self.InitialPrior = []
        for i in range(len(self.SimParams[0])):
            self.InitialPrior.append(
                self.Updater.posterior_marginal(idx_param=i)
            )

        self.Inv_Field = [
            item[0] 
            for item 
            in self.GenSimModel.expparams_dtype[1:] 
        ]
        # self.Heuristic = mpgh.multiPGH(
        #     growth_generator = self.GrowthGenerator, 
        #     self.Updater, 
        #     inv_field=self.Inv_Field, 
        #     increase_time = self.IncreasePGHTime,
        #     pgh_exponent = self.PGHExponent
        # )
        try:
            self.Heuristic = self.GrowthClass.heuristic(
                updater = self.Updater, 
                oplist = self.SimOpList, 
                inv_field = self.Inv_Field, 
                increase_time = self.IncreasePGHTime, 
                pgh_exponent = self.PGHExponent, 
                time_list = self.PlotTimes,
                num_experiments = self.NumExperiments,
            )
        except:
            if test_growth_class_implementation == True: raise
            self.Heuristic = UserFunctions.get_heuristic(
                growth_generator = self.GrowthGenerator,
                updater = self.Updater, 
                oplist = self.SimOpList, 
                inv_field = self.Inv_Field, 
                increase_time = self.IncreasePGHTime, 
                pgh_exponent = self.PGHExponent, 
                time_list = self.PlotTimes,
                num_experiments = self.NumExperiments,
            )
        self.HeuristicType = self.Heuristic.__class__.__name__


        # if checkloss == True or self.checkQLoss==True:     
        #     self.QLosses = np.array([])
        self.QLosses = []
        self.TrackLogTotLikelihood = np.array([])
        self.TrackTime = np.array([]) #only for debugging
        self.Particles = np.array([])
        self.Weights = np.array([])
        self.ResampleEpochs = []
        # self.Experiment = self.Heuristic()   
        self.ExperimentsHistory = np.array([])
        self.FinalParams = np.empty([len(self.SimOpList),2]) #average and standard deviation at the final step of the parameters inferred distributions
        print_loc(print_location=init_model_print_loc)
        self.log_print(['Initialization Ready'])

    def UpdateModel(
        self, 
        n_experiments=None, 
        sigma_threshold=10**-13,
        checkloss=True
    ):
        # self.NumExperiments = n_experiments

        # if self.checkQLoss == True: 
        #     self.QLosses = np.empty(self.NumExperiments)
        self.Covars= np.empty(self.NumExperiments)
        self.TrackEval = []
        self.TrackCovMatrices = []
        self.TrackTime =np.empty(self.NumExperiments)#only for debugging
    
        self.Particles = np.empty([self.NumParticles, 
            len(self.SimParams[0]), self.NumExperiments]
        )
        self.Weights = np.empty([self.NumParticles, self.NumExperiments])
        self.DistributionMeans = np.empty([self.NumExperiments])
        self.DistributionStdDevs = np.empty([self.NumExperiments])
        
        # self.Experiment = self.Heuristic()    
        self.SigmaThresh = sigma_threshold   #This is the value of the Norm of the COvariance matrix which stops the IQLE 
        self.LogTotLikelihood=[] #log_total_likelihood

        self.datum_gather_cumulative_time = 0
        self.update_cumulative_time = 0
        self.learned_est_means = {}
        
        self.TrueParamsDict = {}
        

        true_params_names = DataBase.get_constituent_names_from_name(
            self.TrueOpName
        )
        if self.UseExperimentalData == False:
            for i in range(len(true_params_names)):
                term = true_params_names[i]
                true_param_val = self.TrueParams[i]
                self.TrueParamsDict[term] = true_param_val

        all_params_for_q_loss = list(
            set(true_params_names).union(self.SimOpsNames)
        )
        param_indices = {}
        for op_name in self.SimOpsNames:
            param_indices[op_name] = self.SimOpsNames.index(op_name)

        print_frequency = max(
            int(self.NumExperiments/10), 
            5
        )
        # print("[QML] STARTING QHL UPDATES")
        # true_params = np.array([[self.TrueParams[0]]])
        for istep in range(self.NumExperiments):
            if (istep%print_frequency == 0):
                # print so we can see how far along algorithm is. 
                self.log_print(
                    [
                    "Experiment", istep
                    ]
                )

            # print("[QML] Calling heuristic")
            if istep==0: 
                param_estimates = self.Updater.est_mean()
            else:
                param_estimates = self.TrackEval[-1]
            self.Experiment =  self.Heuristic(
                test_param = "from QML",
                num_params = len(self.SimOpsNames), 
                epoch_id = istep, 
                current_params = param_estimates
            )
            print_loc(global_print_loc)
            self.Experiment[0][0] = self.Experiment[0][0] * self.PGHPrefactor # TODO prefactor, if used, should be inside specific heuristic
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
            
            # exp_t = self.Experiment[0][0]
            before_upd = time.time()
            ## Call updater to update distribution based on datum
            try:
                # print("[QML] calling updater")
                self.Updater.update(
                    self.Datum, 
                    self.Experiment
                )
            except RuntimeError:
                import sys
                self.log_print(
                    [
                        "RuntimeError from updater on model ID ", 
                        self.ModelID, 
                        ":",
                        self.Name
                    ]
                )
                print("\n\nEXITING; Inspect log\n\n")
                sys.exit()

            after_upd = time.time()
            self.update_cumulative_time+=after_upd-before_upd
            
            if self.Updater.just_resampled is True:
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
                self.VolumeList = np.append(
                    self.VolumeList,  
                    np.linalg.det( sp.linalg.sqrtm(self.covmat) )
                )
                print_loc(global_print_loc)
            
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


               
            if (
                checkloss == True 
                and
                self.UseExperimentalData == False
                # and istep%10 == 0
            ): 
                quadratic_loss = 0
                for param in all_params_for_q_loss:
                    if param in self.SimOpsNames:
                        learned_param = self.NewEval[param_indices[param]]
                    else:
                        learned_param = 0

                    if param in true_params_names:
                        true_param = self.TrueParamsDict[param]
                    else:
                        true_param = 0
                    # print("[QML] param:", param, "learned param:", learned_param, "\t true param:", true_param)
                    quadratic_loss += (learned_param - true_param)**2
                self.QLosses.append(quadratic_loss)

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
                    # self.QLosses=(np.resize(self.QLosses, (1,istep)))[0]
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
                # if checkloss == True: 
                #     self.QLosses=(
                #         (np.resize(self.QLosses, (1,istep)))[0]
                #     )
                
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
                self.FinalSigmas  = {}
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
                    self.FinalSigmas[self.SimOpsNames[iterator]] = (
                        self.FinalParams[iterator][1]
                    )

#                plt.savefig(posterior_plot,'posterior.png')
            

            if debug_print:
                self.log_print(["step ", istep])
                self.log_print( ["has params: ", self.NewEval])
                self.log_print(["log total likelihood:",
                    self.TrackLogTotLikelihood[-1]]
                )


    # def resetPrior(self):
    #     print("[QML]\n\n\n\n\IN RESET PRIOR\n\n\n")
    #     self.Updater.prior = self.Prior
    #     self.Updater = qi.SMCUpdater(self.GenSimModel,
    #         self.NumParticles, self.Prior, 
    #         resample_thresh=self.ResamplerThresh,
    #         resampler = qi.LiuWestResampler(a=self.ResamplerA),
    #         debug_resampling=False
    #     )
    # # TODO  use UserFunctions.get_heuristic() instead
    #     self.Heuristic = mpgh.multiPGH(
    #         self.Updater, 
    #         self.SimOpList, 
    #         inv_field=self.Inv_Field
    #     )
    #     return 1
        
        
    def learned_info_dict(self):
        """
        Place essential information after learning has occured into a dict. 
        This can be used to recreate the model on another node. 
        """

        all_post_margs = []
        for i in range(len(self.FinalParams)):
            all_post_margs.append(
                self.Updater.posterior_marginal(idx_param=i)
            )

        learned_info = {}
        learned_info['times'] = self.TrackTime
        learned_info['final_params'] = self.FinalParams
        learned_info['normalization_record'] = self.Updater.normalization_record
        learned_info['log_total_likelihood'] = self.Updater.log_total_likelihood
        learned_info['data_record'] = self.Updater.data_record
        learned_info['name'] = self.Name
        learned_info['model_id'] = self.ModelID
        learned_info['updater'] = pickle.dumps(self.Updater, protocol=2) # TODO regenerate this from mean and std_dev instead of saving it
        learned_info['final_prior'] = self.Updater.prior # TODO regenerate this from mean and std_dev instead of saving it
        learned_info['initial_prior'] = self.InitialPrior
        """
         1st is still the initial prior! that object does not get updated by the learning!
        	modify e.g. using the functions defined in /QML_lib/Distrib.py
         2nd is fine
        """

        learned_info['posterior_marginal'] = all_post_margs
        learned_info['initial_params'] = self.SimParams
        learned_info['volume_list'] = self.VolumeList
        learned_info['track_eval'] = self.TrackEval
        learned_info['track_cov_matrices'] = self.TrackCovMatrices
        learned_info['resample_epochs'] = self.ResampleEpochs
        learned_info['quadratic_losses'] = self.QLosses
        learned_info['learned_parameters'] = self.LearnedParameters
        learned_info['final_sigmas'] = self.FinalSigmas
        learned_info['cov_matrix'] = self.covmat
        learned_info['num_particles'] = self.NumParticles
        learned_info['num_experiments'] = self.NumExperiments
        learned_info['growth_generator'] = self.GrowthGenerator
        learned_info['heuristic'] = self.HeuristicType
        if self.StoreParticlesWeights:
            self.log_print(
                [
                    "Storing particles and weights for model", 
                    self.ModelID
                ]
            )
            learned_info ['particles'] = self.Particles
            learned_info['weights'] = self.Weights

        # if DataBase.alph(self.Name) == DataBase.alph(self.TrueOpName):
        #     # print(
        #     #     "[QML] End of learning. Model", 
        #     #     self.ModelID, 
        #     #     "Prior mean", self.Updater.est_mean()
        #     # )
        #     # print("Samples from prior:", 
        #     #     self.Prior.sample(10)
        #     # )

        #     pickle.dump(
        #         self.Updater, 
        #         open(
        #             str(
        #                 self.ResultsDirectory
        #                 + '/TrueModUpdater.p'
        #             ),
        #             'wb'
        #         )
        #     )

        # self.log_print(
        #     [
        #         "learned_info dict successfully generated in QML instance."
        #     ]
        # )
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

        rds_dbs = rds.databases_from_qmd_id(
            host_name, 
            port_number, 
            qid
        )
        qmd_info_db = rds_dbs['qmd_info_db'] 
        #print("In reduced model. rds_dbs:", rds_dbs)
      #  print("QMD INFO DB has type", type(qmd_info_db), "\n", qmd_info_db)
        
        self.Name = model_name
        self.ModelID = modelID
        self.SimOpList = sim_oplist
        self.ModelID = modelID
        qmd_info = pickle.loads(qmd_info_db.get('QMDInfo'))
        self.ProbeDict = pickle.loads(qmd_info_db['ProbeDict'])
        self.MeasurementType = qmd_info['measurement_type']
        self.ExperimentalMeasurements = qmd_info['experimental_measurements']
        self.UseExperimentalData = qmd_info['use_experimental_data']
        # self.NumParticles = qmd_info['num_particles']
        # self.NumExperiments = qmd_info['num_experiments']
        self.NumProbes = qmd_info['num_probes']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.TrueOpList = qmd_info['true_oplist']
        self.TrueParams = qmd_info['true_params']
        self.TrueOpName  = qmd_info['true_name']
        self.PlotProbes = pickle.load(
            open(qmd_info['plot_probe_file'], 'rb')
        )
        self.PlotTimes = qmd_info['plot_times']
        self.QLE = qmd_info['qle']
        self.UseExpCustom = qmd_info['use_exp_custom']
        self.StoreParticlesWeights = qmd_info[
            'store_particles_weights'
        ]
        self.BayesFactors = {}
        self.NumQubits = DataBase.get_num_qubits(self.Name)
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
                self.log_print(
                    [
                        "Unable to load learned info", 
                        "model_id_str: ", model_id_str,
                        "model id: ", self.ModelID,
                        "learned info keys:, ", learned_models_info.keys(),
                        "learned info:, ", learned_models_info.get(model_id_str)
                    ]
                )
        self.NumParticles = learned_info['num_particles']
        self.NumExperiments = learned_info['num_experiments']
        self.Times = list(learned_info['times'])
        self.FinalParams = learned_info['final_params'] # should be final params from learning process
        self.SimParams_Final = np.array([[self.FinalParams[0,0]]]) # TODO this won't work for multiple parameters
        self.Prior = learned_info['final_prior'] # TODO this can be recreated from finalparams, but how for multiple params?
        self._normalization_record = learned_info['normalization_record']
        self.log_total_likelihod = learned_info['log_total_likelihood']
        self.RawVolumeList = learned_info['volume_list'] 
        self.VolumeList = {}
        for i in range(len(self.RawVolumeList)):
            self.VolumeList[i] = self.RawVolumeList[i]

        self.TrackEval = np.array(learned_info['track_eval'])
        self.TrackCovMatrices = np.array(learned_info['track_cov_matrices'])
        self.ResampleEpochs = learned_info['resample_epochs']
        self.QuadraticLosses = learned_info['quadratic_losses']
        self.LearnedParameters = learned_info['learned_parameters']
        self.FinalSigmas = learned_info['final_sigmas']
        self.cov_matrix = learned_info['cov_matrix']
        self.GrowthGenerator = learned_info['growth_generator']
        try:
            self.GrowthClass = GrowthRules.get_growth_generator_class(
                growth_generation_rule = self.GrowthGenerator,
                use_experimental_data = self.UseExperimentalData
            )
        except:
            # raise
            self.GrowthClass = None
        self.HeuristicType = learned_info['heuristic']

        try:
            self.LatexTerm = self.GrowthClass.latex_name(
                name = self.Name
            )
        except:
            if test_growth_class_implementation == True: raise
            self.LatexTerm = UserFunctions.get_latex_name(
                name = self.Name,
                growth_generator = self.GrowthGenerator
            )

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
        # plot_probe_path = None,
        # probe = None #  TODO generalise probe
    ):
        # TODO expectation_values dict only for |++> probe as is.
        # if probe is None and plot_probe_path is None:
        #     probe  = ExpectationValues.n_qubit_plus_state(self.NumQubits)
        # else:

        #     plot_probe_dict = pickle.load(
        #         open(plot_probe_path, 'rb')
        #     )
        #     probe = plot_probe_dict[self.NumQubits]

        probe = self.PlotProbes[self.NumQubits]
        
        self.log_print(
            [
            "Computing expectation values.", 
            "\nMeasurement Type:", self.MeasurementType, 
            "\nLearnedHamiltonian", self.LearnedHamiltonian,
            # "\nPlotProbePath:", plot_probe_path, 
            "\nProbe:", probe
            ]
        )

        present_expec_val_times = sorted(
            list(self.expectation_values.keys())
        )

        required_times = sorted(
            list( set(times) - set(present_expec_val_times) )
        )

        # print(
        #     "[QML - compute expectation values]", 
        #     "times to compute:", required_times
        # )
        for t in required_times:
            try:
                self.expectation_values[t] = self.GrowthClass.expectation_value(
                    ham = self.LearnedHamiltonian, 
                    t = t,
                    state = probe
                )
            except:
                if test_growth_class_implementation == True: raise
                self.expectation_values[t] = UserFunctions.expectation_value_wrapper(
                    method=self.MeasurementType,
                    ham = self.LearnedHamiltonian, 
                    t = t,
                    state = probe
                )

    def r_squared(
        self, 
        plot_probes,
        times=None, 
        min_time = 0,
        max_time = None 
    ):
        # TODO recheck R squared functions eg which probe used
        self.log_print(
            [
                "R squared function for", self.Name
            ]
        )
        if times == None:
            exp_times = sorted(
                list(self.ExperimentalMeasurements.keys())
            )
        else:
            exp_times = times
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
        # probe = np.array([0.5, 0.5, 0.5, 0.5+0j]) # TODO generalise
        # probe  = ExpectationValues.n_qubit_plus_state(self.NumQubits)
        # probe = plot_probes[self.NumQubits]
        probe = self.PlotProbes[self.NumQubits]

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
                # if self.UseExperimentalData==True:
                #     # sim = ExpectationValues.hahn_evolution(
                #     sim = UserFunctions.expectation_value_wrapper(
                #         method='hahn',
                #         ham=ham, t=t, state=probe
                #     )

                # else:
                #     # sim = ExpectationValues.traced_expectation_value_project_one_qubit_plus(
                #     sim = UserFunctions.expectation_value_wrapper(
                #         method='trace_all_but_first',
                #         ham=ham, t=t, state=probe
                #     )

                try:
                    sim = self.GrowthClass.expectation_value(
                        ham=ham, 
                        t=t, 
                        state=probe
                    )
                except:
                    if test_growth_class_implementation == True: raise
                    sim = UserFunctions.expectation_value_wrapper(
                        method=self.MeasurementType,
                        ham=ham, 
                        t=t, 
                        state=probe
                    )
                self.expectation_values[t] = sim

            true = self.ExperimentalMeasurements[t]
            diff_squared = (sim - true)**2
            sum_of_residuals += diff_squared
            self.r_squared_of_t[t] = 1 - sum_of_residuals/datavar

        Rsq = 1 - sum_of_residuals/datavar
        
        return Rsq


    def r_squared_by_epoch(
        self, 
        plot_probes, 
        times=None, 
        min_time=0,
        max_time=None,
        num_points=10 # maximum number of epochs to take R^2 at
    ):
        # TODO recheck R squared functions eg which probe used
        self.log_print(
            [
                "R squared by epoch function for", 
                self.Name,
                "Times passed:", 
                times
            ]
        )
        
        if times == None:
            exp_times = sorted(list(self.ExperimentalMeasurements.keys()))
        else:
            exp_times = times


        if max_time is None:
            max_time = max(exp_times)

        min_time = expdt.nearestAvailableExpTime(
            exp_times, 
            min_time
        )
        max_time = expdt.nearestAvailableExpTime(
            exp_times, 
            max_time
        )
        min_data_idx = exp_times.index(min_time)
        max_data_idx = exp_times.index(max_time)
        exp_times = exp_times[min_data_idx:max_data_idx]

        exp_data = [
            self.ExperimentalMeasurements[t] 
            for t in exp_times
        ]

        # exp_data = exp_data[0::10]
        # probe = np.array([0.5, 0.5, 0.5, 0.5+0j]) # TODO generalise
        # probe  = plot_probes[self.NumQubits]
        probe = self.PlotProbes[self.NumQubits]

        datamean = np.mean(exp_data[0:max_data_idx])
        datavar = np.sum( 
            (exp_data[0:max_data_idx] - datamean)**2  
        )
        r_squared_by_epoch =  {}

        # only use subset of epochs in case there are a large 
        # num experiments due to heavy computational overhead
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
                try:
                    sim = self.GrowthClass.expectation_value(
                    ham=ham, 
                    t=t, 
                    state=probe
                    )
                except:
                    if test_growth_class_implementation == True: raise
                    sim = UserFunctions.expectation_value_wrapper(
                        method=self.MeasurementType,
                        ham=ham, 
                        t=t, 
                        state=probe
                    )

                # if self.UseExperimentalData==True:
                #     # sim = ExpectationValues.hahn_evolution(
                #     sim = UserFunctions.expectation_value_wrapper(
                #         method='hahn',
                #         ham=ham, 
                #         t=t, 
                #         state=probe
                #     )
                # else:
                #     # sim = ExpectationValues.traced_expectation_value_project_one_qubit_plus(
                #     sim = UserFunctions.expectation_value_wrapper(
                #         method='trace_all_but_first',
                #         ham=ham, 
                #         t=t, 
                #         state=probe
                #     )

                true = self.ExperimentalMeasurements[t]
                diff_squared = (sim - true)**2
                sum_of_residuals += diff_squared

            Rsq = 1 - sum_of_residuals/datavar
            r_squared_by_epoch[e] = Rsq
        self.r_squared_by_epoch = r_squared_by_epoch
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

        rds_dbs = rds.databases_from_qmd_id(
            host_name, 
            port_number, 
            qid
        )
        qmd_info_db = rds_dbs['qmd_info_db'] 
        learned_models_info = rds_dbs['learned_models_info']
    
        model_id_float = float(modelID)
        model_id_str = str(model_id_float)
        try:
            learned_model_info = pickle.loads(
                learned_models_info.get(model_id_str), 
                encoding='latin1'
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
        self.PlotProbePath = qmd_info['plot_probe_file']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.TrueOpList = qmd_info['true_oplist']
        self.TrueParams = qmd_info['true_params']
        self.TrueOpName  = qmd_info['true_name']
        self.UseExpCustom = qmd_info['use_exp_custom']
        self.MeasurementType = qmd_info['measurement_type']
        self.UseExperimentalData = qmd_info['use_experimental_data']
        self.ExperimentalMeasurements = qmd_info['experimental_measurements']
        self.ExperimentalMeasurementTimes = qmd_info['experimental_measurement_times']
        updater_from_prior = qmd_info['updater_from_prior']

        self.log_file = log_file
        self.Q_id = qid

        self.Name = learned_model_info['name']
        op = DataBase.operator(self.Name)
        self.SimOpList = op.constituents_operators # todo, put this in a lighter function
        self.Times = learned_model_info['times']
        self.FinalParams = learned_model_info['final_params'] 
        self.SimParams_Final = np.array([[self.FinalParams[0,0]]]) # TODO this won't work for multiple parameters
        self.InitialParams = learned_model_info['initial_params']
        self.GrowthGenerator = learned_model_info['growth_generator']
        self.GrowthClass = GrowthRules.get_growth_generator_class(
            growth_generation_rule = self.GrowthGenerator,
            use_experimental_data = self.UseExperimentalData
        )
        self.Prior = learned_model_info['final_prior'] # TODO this can be recreated from finalparams, but how for multiple params?
        self.PosteriorMarginal = learned_model_info['posterior_marginal']
        self.InitialPrior = learned_model_info['initial_prior']
        self._normalization_record = learned_model_info['normalization_record']
        self.log_likelihood = learned_model_info['log_total_likelihood']
        
        log_identifier = str("Bayes "+str(self.ModelID)) 
                
        self.GenSimModel = gsi.GenSimQMD_IQLE(
            oplist=self.SimOpList,
            modelparams=self.SimParams_Final, 
            true_oplist = self.TrueOpList,
            trueparams = self.TrueParams, 
            truename=self.TrueOpName,
            measurement_type = self.MeasurementType,
            growth_generation_rule = self.GrowthGenerator, 
            use_experimental_data = self.UseExperimentalData,
            experimental_measurements = self.ExperimentalMeasurements,
            experimental_measurement_times=(
                self.ExperimentalMeasurementTimes
            ),             
            model_name=self.Name, num_probes = self.NumProbes, 
            probe_dict=self.ProbeDict, log_file=self.log_file,
            log_identifier=log_identifier
        )    
        # print("[QML] upd from prior:", updater_from_prior)
        if updater_from_prior == True:
            self.Updater = qi.SMCUpdater(
                self.GenSimModel, 
                self.NumParticles,
                self.Prior, 
                resample_thresh=self.ResamplerThresh , 
                resampler=qi.LiuWestResampler(a=self.ResamplerA), 
                debug_resampling=False
            )
            self.Updater._normalization_record = self._normalization_record
            self.Updater.log_likelihood = self.log_likelihood
        else:
            self.Updater = pickle.loads(
                learned_model_info['updater']
            )

        # print(
        #     "Providing prior to BF model instance {}:\n{}".format(
        #             self.ModelID, 
        #             self.Prior
        #         ),
        #     "\n updater.est_mean():", self.Updater.est_mean()
        # )


        #self.GenSimModel = pickle.loads(learned_model_info['gen_sim_model'])
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
        


