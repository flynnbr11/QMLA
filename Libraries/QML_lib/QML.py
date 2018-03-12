from __future__ import print_function # so print doesn't show brackets
import numpy as np
import scipy as sp
import os 
import Evo as evo
from Distrib import *
import ProbeStates as pros
import GenSimQMD_IQLE as gsi
import multiPGH as mpgh
import DataBase as DB
from MemoryTest import print_loc
from EvalLoss import *
from psutil import virtual_memory
from RedisSettings import *
global debug_print
debug_print = False

global print_mem_status
print_mem_status = True

global_print_loc = False

class ModelLearningClass():
    def __init__(self, name, num_probes=20, probe_dict=None):
        self.TrueOpList = np.array([])        # These are the true operators of the true model for time evol in the syst
        self.SimOpList = np.array([])            # Operators for the model under test for time evol. in the sim.
        self.TrueParams = np.array([])        #True parameters of the model of the system for time evol in the syst
        self.SimParams = np.array([])         #Parameters for the model under test for time evol. in the sim.
        self.ExpParams = np.array([])         #ExpParams of the simulator used inside the GenSimQMD_IQLE class
        self.Particles = np.array([])         #List of all the particles
        self.Weights = np.array([])           # List of all the weights of the particles
        self.BayesFactorList = np.array([]) #probably to be removed
        self.KLogTotLikelihood = np.array([]) #Total Likelihood for the BayesFactor calculation
        self.VolumeList = np.array([])        #List with the Volume as a function of number of steps
        self.Name = name
        self.Operator = DB.operator(name)
      #  self.Matrix = self.Operator.matrix
        self.Dimension = self.Operator.num_qubits
        self.NumExperimentsToDate = 0
        self.BayesFactors = {}
        self.NumProbes = num_probes
        self.ProbeDict = probe_dict



        
    def InsertNewOperator(self, NewOperator):

        self.NumParticles = len(self.Particles)
        self.OpList = MatrixAppendToArray(self.OpList, NewOperator)
        self.Particles =  np.random.rand(self.NumParticles,len(self.OpList))
        self.ExpParams = np.append(self.ExpParams, 1)
        self.Weights = np.full((1, self.NumParticles), 1./self.NumParticles)
    
    """Initilise the Prior distribution using a uniform multidimensional distribution where the dimension d=Num of Params for example using the function MultiVariateUniformDistribution"""
    
    def InitialiseNewModel(self, trueoplist, modeltrueparams, simoplist, simparams, numparticles, modelID, resample_thresh=0.5, resampler_a = 0.95, pgh_prefactor = 1.0, checkloss=True,gaussian=True, use_exp_custom=True, enable_sparse=True, debug_directory=None, qle=True):

        import pickle
        qmd_info = pickle.loads(qmd_info_db.get('QMDInfo'))

        self.ProbeDict = pickle.loads(qmd_info_db['ProbeDict'])
        self.NumParticles = qmd_info['num_particles']
        self.NumProbes = qmd_info['num_probes']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.TrueOpList = qmd_info['true_oplist']
        self.TrueParams = qmd_info['true_params']
        self.TrueOpName  = qmd_info['true_name']
        self.QLE = qmd_info['qle']


        
#        self.TrueOpList = np.asarray(trueoplist)
 #       self.TrueParams = np.asarray(modeltrueparams)
        self.SimOpList  = np.asarray(simoplist)
        #self.SimParams = simparams[0]
        self.SimParams = np.asarray([simparams[0]])
        self.InitialParams = np.asarray([simparams[0]])
#        self.NumParticles = numparticles # function to adapt by dimension
#        self.ResamplerThresh = resample_thresh
#        self.ResamplerA = resampler_a
#        self.PGHPrefactor = pgh_prefactor
        self.ModelID = int(modelID)
        self.UseExpCustom = use_exp_custom
        self.EnableSparse = enable_sparse
#        self.QLE = qle
        self.checkQLoss = True
        
#        print("Model instance ", self.Name, " has initial parameters: ", self.SimParams, "\nTrue op list: \n", self.TrueOpList)
        
        if debug_directory is not None: 
            self.debugSave = True
            self.debugDirectory = debug_directory 
        else:            
            self.debugSave = False
        #self.TrueHam = evo.getH(self.TrueParams, self.TrueOpList) # This is the Hamiltonian for the time evolution in the system
#         self.Prior = MultiVariateUniformDistribution(len(self.OpList))
        if gaussian:
            self.Prior = MultiVariateNormalDistributionNocov(len(self.SimOpList))
        else:
             self.Prior = MultiVariateUniformDistribution(len(self.SimOpList)) #the prior distribution is on the model we want to test i.e. the one implemented in the     simulator
#            self.Prior = MultiVariateNormalDistributionNocov(len(self.SimOpList), mean = self.TrueParams[0])
  
        self.ProbeCounter = 0 #probecounter for the choice of the state
#         if len(oplist)>1:
#             self.ProbeState = pros.choose_probe(self.OpList, self.TrueParams)
#         else:
#             self.ProbeState = (sp.linalg.orth(oplist[0])[0]+sp.linalg.orth(oplist[0])[1])/np.sqrt(2)
        
#         if len(self.SimOpList)>1:
# #             self.ProbeState = pros.choose_probe(self.TrueOpList, self.TrueParams)#change to pros.choose_randomprobe
#             self.ProbeState = pros.choose_randomprobe(self.SimOpList, self.TrueParams)#change to pros.choose_randomprobe
#         else:
# #             self.ProbeState = (sp.linalg.orth(trueoplist[0])[0]+sp.linalg.orth(trueoplist[0])[1])/np.sqrt(2)
#             self.ProbeState = pros.choose_randomprobe(self.SimOpList, self.TrueParams)
        
    
        #self.ProbeList = np.array([evo.zero(),evo.plus(),evo.minusI()])
        
        # self.ProbeList = np.array([evo.zero()])
        
        self.ProbeList = list(map(lambda x: pros.def_randomprobe(self.TrueOpList), range(15)))
        #self.ProbeList =  [pros.def_randomprobe(self.TrueOpList)]
        
        #When ProbeList is not defined the probestate will be chosen completely random for each experiment.
        self.GenSimModel = gsi.GenSimQMD_IQLE(oplist=self.SimOpList, modelparams=self.SimParams, true_oplist = self.TrueOpList, trueparams = self.TrueParams, truename=self.TrueOpName, num_probes = self.NumProbes, probe_dict=self.ProbeDict, probecounter = self.ProbeCounter, solver='scipy', trotter=True, qle=self.QLE, use_exp_custom=self.UseExpCustom, enable_sparse=self.EnableSparse, model_name=self.Name)    # probelist=self.TrueOpList,,

        
                
        #print('Chosen probestate: ' + str(self.GenSimModel.ProbeState))
        #print('Chosen true_params: ' + str(self.TrueParams))
        #self.GenSimModel = gsi.GenSim_IQLE(oplist=self.OpList, modelparams=self.TrueParams, probecounter = self.ProbeCounter, probelist= [self.ProbeState], solver='scipy', trotter=True)
        #Here you can turn off the debugger change the resampling threshold etc...
        self.Updater = qi.SMCUpdater(self.GenSimModel, self.NumParticles, self.Prior, resample_thresh=self.ResamplerThresh , resampler = qi.LiuWestResampler(a=self.ResamplerA), debug_resampling=False)
        
        #doublecheck and coment properly
        self.Inv_Field = [item[0] for item in self.GenSimModel.expparams_dtype[1:] ]
        #print('Inversion fields are: ' + str(self.Inv_Field))
#        self.Heuristic = mpgh.multiPGH(self.Updater, self.SimOpList, inv_field=self.Inv_Field)
        self.Heuristic = mpgh.multiPGH(self.Updater, inv_field=self.Inv_Field)
        
        
        #TODO: should heuristic use TRUEoplist???
        
        #print('Heuristic output:' + repr(self.Heuristic()))
        self.ExpParams = np.empty((1, ), dtype=self.GenSimModel.expparams_dtype)
        
        self.NumExperiments = 0
        if checkloss == True or self.checkQLoss==True:     
            self.QLosses = np.array([])
       # self.TrackEval = np.array([]) #only for debugging
      #  self.Covars= np.array([])
        self.TrackLogTotLikelihood = np.array([])
        self.TrackTime = np.array([]) #only for debugging
        self.Particles = np.array([])
        self.Weights = np.array([])
        self.Experiment = self.Heuristic()   
        self.ExperimentsHistory = np.array([])
        self.FinalParams = np.empty([len(self.SimOpList),2]) #average and standard deviation at the final step of the parameters inferred distributions

        print('Initialization Ready')
        
        
        
        
        
    
    
#     #UPDATER e quanto segue VA RESETTATO COME PROPRIETA DELLA CLASSE
#     updater= qi.SMCUpdater(model, n_particles, prior, resample_thresh=0.5, resampler = qi.LiuWestResampler(a=0.95), debug_resampling=True)
    
    
#     probestate=pros.choose_probe(oplist,true_params)
    
#     Model = gsi.GenSim_IQLE(oplist=oplist, modelparams=true_params, probecounter = 0, probelist= [probestate], solver='scipy', trotter=True)

    
#     inv_field = [item[0] for item in model.expparams_dtype[1:] ]
#     print('Inversion fields are: ' + str(inv_field))
#     heuristic = mpgh.multiPGH(updater, oplist, inv_field=inv_field)

    
#     self.ExpParams = np.empty((1, ), dtype=model.expparams_dtype)
#     experiment = heuristic()
    
    
    
    
    
    
#     """Function which perfoms IQLE on the particular instance of the model for given number of experiments etc... and 
#     updates all the relevant quanttities in the object for comparison with other models -- It must be run after InitialiseNewModel"""
    def UpdateModel(self, n_experiments, sigma_threshold=10**-13,checkloss=True):
   
    #Insert check and append old data to a storing list like ExperimentsHistory or something similar. 
        self.NumExperiments = n_experiments
        if self.checkQLoss == True: 
            self.QLosses = np.empty(n_experiments)
        self.Covars= np.empty(n_experiments)
        self.TrackEval = []
        self.TrackTime =np.empty(n_experiments)#only for debugging
    
        self.Particles = np.empty([self.NumParticles, len(self.SimParams[0]), self.NumExperiments])
#        self.Particles = np.empty([self.NumParticles, len(self.SimParams), self.NumExperiments]) ## I changed this to test init from db-- Brian
        self.Weights = np.empty([self.NumParticles, self.NumExperiments])
        self.Experiment = self.Heuristic()    
        self.SigmaThresh = sigma_threshold   #This is the value of the Norm of the COvariance matrix which stops the IQLE 
        self.LogTotLikelihood=[] #log_total_likelihood

        #print("sigma_threshold = ", self.SigmaThresh)
        for istep in range(self.NumExperiments):
            # self.Experiment =  self.PGHPrefactor * (self.Heuristic()) ## TODO: use PGH prefactor, either here or in multiPGH
            #print("\n\nUpdate at exp # ", istep)
            #print("Memory used : ", virtual_memory().percent, "%")
            print_loc(global_print_loc)
            
            self.Experiment =  self.Heuristic()
            print_loc(global_print_loc)
            self.Experiment[0][0] = self.Experiment[0][0] * self.PGHPrefactor
            global_print_loc
            self.NumExperimentsToDate += 1
            print_loc(global_print_loc)
            #print('Chosen experiment: ' + repr(self.Experiment))
            if istep == 0:
                print_loc(global_print_loc)
                print('Initial time selected > ' + str(self.Experiment[0][0]))
            
            
            self.TrackTime[istep] = self.Experiment[0][0]
            print_loc(global_print_loc)
            
            #TODO should this use TRUE params???
            true_params = np.array([[self.TrueParams[0]]])
            self.Datum = self.GenSimModel.simulate_experiment(self.SimParams, self.Experiment, repeat=10) # todo reconsider repeat number
#            self.Datum = self.GenSimModel.simulate_experiment(true_params, self.Experiment, repeat=1) # todo reconsider repeat number
            
            
            print_loc(global_print_loc)
            
            #print(str(self.GenSimModel.ProbeState))
            self.Updater.update(self.Datum, self.Experiment)
            print_loc(global_print_loc)

            if len(self.Experiment[0]) < 3:
                print_loc(global_print_loc)
                covmat = self.Updater.est_covariance_mtx()
                self.VolumeList = np.append(self.VolumeList, covmat)
                print_loc(global_print_loc)

            else:
                print_loc(global_print_loc)

                #covmat = self.Updater.region_est_ellipsoid()
                covmat = self.Updater.est_covariance_mtx()
                self.VolumeList = np.append(self.VolumeList,  np.linalg.det( sp.linalg.sqrtm(covmat) )    )
                print_loc(global_print_loc)
            
            """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!this one is probably to remove from here, as superfluous????????????????????????????"""
            self.Heuristic = mpgh.multiPGH(self.Updater, self.SimOpList, inv_field=self.Inv_Field)
            print_loc(global_print_loc)
            
            self.TrackEval.append(self.Updater.est_mean())
            print_loc(global_print_loc)
            self.Covars[istep] = np.linalg.norm(self.Updater.est_covariance_mtx())
            print_loc(global_print_loc)
            self.Particles[:, :, istep] = self.Updater.particle_locations
            self.Weights[:, istep] = self.Updater.particle_weights
            #self.TrackLogTotLikelihood = np.append(self.TrackLogTotLikelihood, self.Updater.log_total_likelihood)

            self.NewEval = self.Updater.est_mean()
            print_loc(global_print_loc)
#            print("At epoch", istep, "loglikelihood=", self.Updater.log_total_likelihood)

                
            if checkloss == True: 
                #self.NewLoss = eval_loss(self.GenSimModel, self.NewEval, self.TrueParams)
                #self.QLosses[istep] = self.NewLoss
            
#                if self.NewLoss<(10**(-17)) and False: #  I don't want it to stop learning - Brian
                if False:
                    if self.debugSave: 
                        self.debug_store()
                    print('Final time selected > ' + str(self.Experiment[0][0]))
                    print('Exiting learning for Reaching Num. Prec. -  Iteration Number ' + str(istep))
                    for iterator in range(len(self.FinalParams)):
                        self.FinalParams[iterator]= [np.mean(self.Particles[:,iterator,istep]), np.std(self.Particles[:,iterator,istep])]
                        print('Final Parameters mean and stdev:'+str(self.FinalParams[iterator])) 
#                        print('Final quadratic loss: ', str(self.QLosses[-1]))

                    self.LogTotLikelihood=self.Updater.log_total_likelihood                
                    self.QLosses=(np.resize(self.QLosses, (1,istep)))[0]
                    self.Covars=(np.resize(self.Covars, (1,istep)))[0]
                    self.Particles = self.Particles[:, :, 0:istep]
                    self.Weights = self.Weights[:, 0:istep]
                    self.TrackTime = self.TrackTime[0:istep] 
                    break #TODO: Reinstate this break; disabled to test different cases while chasing memory leak.
            
            if self.Covars[istep]<self.SigmaThresh and False: #  I don't want it to stop learning - Brian
                if self.debugSave: 
                    self.debug_store()
                print('Final time selected > ' + str(self.Experiment[0][0]))
                print('Exiting learning for Reaching Cov. Norm. Thrshold of '+ str(self.Covars[istep]))
                print(' at Iteration Number ' + str(istep)) 
                for iterator in range(len(self.FinalParams)):
                    self.FinalParams[iterator]= [np.mean(self.Particles[:,iterator,istep]), np.std(self.Particles[:,iterator,istep])]
                    print('Final Parameters mean and stdev:'+str(self.FinalParams[iterator]))
#                    print('Final quadratic loss: ', self.QLosses[-1]  )
                self.LogTotLikelihood=self.Updater.log_total_likelihood
                if checkloss == True: 
                    self.QLosses=(np.resize(self.QLosses, (1,istep)))[0]
                
                self.Covars=(np.resize(self.Covars, (1,istep)))[0]
                self.Particles = self.Particles[:, :, 0:istep]
                self.Weights = self.Weights[:, 0:istep]
                self.TrackTime = self.TrackTime[0:istep]
                
                break 
            
            ####Need to ADD check with dereivative of sigmas!!!!
            
            
            
            if istep == self.NumExperiments-1:
                print('Final time selected > ' + str(self.Experiment[0][0]))
                self.LogTotLikelihood=self.Updater.log_total_likelihood
                if self.debugSave: 
                    self.debug_store()
        
                for iterator in range(len(self.FinalParams)):
                    self.FinalParams[iterator]= [np.mean(self.Particles[:,iterator,istep-1]), np.std(self.Particles[:,iterator,istep-1])]
                    print('Final Parameters mean and stdev:'+str(self.FinalParams[iterator]))
#                   print('Final quadratic loss: ', str(self.QLosses[-1]))
#                final_ham = evo.getH(self.)

            if debug_print:
                print("step ", istep)
                print( " has params: ", self.NewEval)
                print(" log tot like  : ", self.TrackLogTotLikelihood[-1])


    def resetPrior(self):
        self.Updater.prior = self.Prior
        self.Updater = qi.SMCUpdater(self.GenSimModel, self.NumParticles, self.Prior, resample_thresh=self.ResamplerThresh , resampler = qi.LiuWestResampler(a=self.ResamplerA), debug_resampling=False)
        self.Heuristic = mpgh.multiPGH(self.Updater, self.SimOpList, inv_field=self.Inv_Field)
        print("model params reset")
        return 1
        
        
    def learned_info_dict(self):
        """
        Place essential information after learning has occured into a dict. This can be used to recreate the model on another node. 
        """
        learned_info = {}
        learned_info['times'] = self.TrackTime
        learned_info['final_params'] = self.FinalParams
        learned_info['normalization_record'] = self.Updater.normalization_record
        learned_info['data_record'] = self.Updater.data_record
        learned_info['name'] = self.Name
        learned_info['model_id'] = self.ModelID
        learned_info['final_prior'] = self.Updater.prior # TODO regenerate this from mean and std_dev instead of saving it
        learned_info['initial_params'] = self.InitialParams
        learned_info['updater'] = pickle.dumps(self.Updater)
        return learned_info
        
        
    
    def UpdateKLogTotLikelihood(self, epoch, tpool, stepnum):
        # Calcalate total log likelihood when the model finishes, compared with all previously completed but still active models. 
        
        mytpool = np.setdiff1d(tpool, self.TrackTime[-stepnum-1:-1])
        
        self.TrackLogTotLikelihood = np.append(self.TrackLogTotLikelihood, LogL_UpdateCalc(self, tpool))


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
            print("Need to pass debug_dir to QML.debug_save function")
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
            print("Need to pass debug_dir to QML.debug_save function")
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
    def __init__(self, model_name, sim_oplist, true_oplist, true_params, numparticles, modelID, resample_thresh=0.5, resample_a=0.9, qle=True, probe_dict= None):

        self.Name = model_name
        self.ModelID = modelID
        self.SimOpList = sim_oplist
        self.ModelID = modelID
        
        import pickle
        qmd_info = pickle.loads(qmd_info_db.get('QMDInfo'))
        
        self.ProbeDict = pickle.loads(qmd_info_db['ProbeDict'])
        self.NumParticles = qmd_info['num_particles']
        self.NumProbes = qmd_info['num_probes']
        self.ResamplerThresh = qmd_info['resampler_thresh']
        self.ResamplerA = qmd_info['resampler_a']
        self.PGHPrefactor = qmd_info['pgh_prefactor']
        self.TrueOpList = qmd_info['true_oplist']
        self.TrueParams = qmd_info['true_params']
        self.TrueOpName  = qmd_info['true_name']
        self.QLE = qmd_info['qle']
        self.BayesFactors = {}


        
    def updateLearnedValues(self, learned_info):
        """
        Pass a dict, learned_info, with essential info on reconstructing the state of the model, updater and GenSimModel
        """
        self.Times = learned_info['times']
        self.FinalParams = learned_info['final_params'] # should be final params from learning process
        self.SimParams_Final = np.array([[self.FinalParams[0,0]]]) # TODO this won't work for multiple parameters
        self.Prior = learned_info['final_prior'] # TODO this can be recreated from finalparams, but how for multiple params?
        self._normalization_record = learned_info['normalization_record']


        self.GenSimModel = gsi.GenSimQMD_IQLE(oplist=self.SimOpList, modelparams=self.SimParams_Final, true_oplist = self.TrueOpList, trueparams = self.TrueParams, truename=self.TrueOpName, model_name=self.Name, probe_dict = self.ProbeDict)    # probelist=self.TrueOpList,,

        self.Updater = qi.SMCUpdater(self.GenSimModel, self.NumParticles, self.Prior, resample_thresh=self.ResamplerThresh , resampler = qi.LiuWestResampler(a=self.ResamplerA), debug_resampling=False)
        self.Updater._normalization_record = self._normalization_record
    
        
class modelClassForRemoteBayesFactor():
    def __init__(self, modelID):
        model_id_float = float(modelID)
        model_id_str = str(model_id_float)
        import pickle
        learned_model_info = pickle.loads(learned_models_info.get(model_id_str))        
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

        self.Name = learned_model_info['name']
        op = DB.operator(self.Name)
        self.SimOpList = op.constituents_operators # todo, put this in a lighter function
        self.Times = learned_model_info['times']
        self.FinalParams = learned_model_info['final_params'] 
        self.SimParams_Final = np.array([[self.FinalParams[0,0]]]) # TODO this won't work for multiple parameters
        self.InitialParams = learned_model_info['initial_params']
        
        self.Prior = learned_model_info['final_prior'] # TODO this can be recreated from finalparams, but how for multiple params?
        self._normalization_record = learned_model_info['normalization_record']

        self.GenSimModel = gsi.GenSimQMD_IQLE(oplist=self.SimOpList, modelparams=self.SimParams_Final, true_oplist = self.TrueOpList, trueparams = self.TrueParams, truename=self.TrueOpName, model_name=self.Name, num_probes = self.NumProbes, probe_dict = self.ProbeDict)    

        self.Updater_regenerated = qi.SMCUpdater(self.GenSimModel, self.NumParticles, self.Prior, resample_thresh=self.ResamplerThresh , resampler = qi.LiuWestResampler(a=self.ResamplerA), debug_resampling=False)
        self.Updater_regenerated._normalization_record = self._normalization_record
        self.Updater_regenerated._data_record = learned_model_info['data_record']
        
        self.Updater = pickle.loads(learned_model_info['updater'])
        del qmd_info, learned_model_info
        
        # could pickle updaters to a redis db for updaters, but first construct these model classes each time a BF is to be computed. 

        


