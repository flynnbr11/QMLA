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

global debug_print
debug_print = False

class ModelLearningClass():
    def __init__(self, name):
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
        self.Matrix = self.Operator.matrix
        self.NumExperimentsToDate = 0
        self.BayesFactors = {}
        
    
    def InsertNewOperator(self, NewOperator):
        self.NumParticles = len(self.Particles)
        self.OpList = MatrixAppendToArray(self.OpList, NewOperator)
        self.Particles =  np.random.rand(self.NumParticles,len(self.OpList))
        self.ExpParams = np.append(self.ExpParams, 1)
        self.Weights = np.full((1, self.NumParticles), 1./self.NumParticles)

    
    """Initilise the Prior distribution using a uniform multidimensional distribution where the dimension d=Num of Params for example using the function MultiVariateUniformDistribution"""
    
    def InitialiseNewModel(self, trueoplist, modeltrueparams, simoplist, simparams, numparticles, modelID, resample_thresh=0.5,checkloss=False,gaussian=False, use_exp_custom=True, debug_directory=None):
        
        self.TrueOpList = np.asarray(trueoplist)
        self.TrueParams = np.asarray(modeltrueparams)
        self.SimOpList  = np.asarray(simoplist)
        self.SimParams = np.asarray(simparams)
        self.NumParticles = numparticles # function to adapt by dimension
        self.ResamplerTresh = resample_thresh
        self.ModelID = int(modelID)
        self.UseExpCustom = use_exp_custom
        
        if debug_directory is not None: 
            self.debugSave = True
            self.debugDirectory = debug_directory            
        #self.TrueHam = evo.getH(self.TrueParams, self.TrueOpList) # This is the Hamiltonian for the time evolution in the system
#         self.Prior = MultiVariateUniformDistribution(len(self.OpList))
        if gaussian:
            self.Prior = MultiVariateNormalDistributionNocov(len(self.SimOpList))
        else:
            self.Prior = MultiVariateUniformDistribution(len(self.SimOpList)) #the prior distribution is on the model we want to test i.e. the one implemented in the simulator
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
        
        self.GenSimModel = gsi.GenSimQMD_IQLE(oplist=self.SimOpList, modelparams=self.SimParams, true_oplist = self.TrueOpList, trueparams = self.TrueParams, probelist=self.ProbeList, probecounter = self.ProbeCounter, solver='scipy', trotter=True, use_exp_custom=self.UseExpCustom)    # probelist=self.TrueOpList,,

        
                
        #print('Chosen probestate: ' + str(self.GenSimModel.ProbeState))
        #print('Chosen true_params: ' + str(self.TrueParams))
        #self.GenSimModel = gsi.GenSim_IQLE(oplist=self.OpList, modelparams=self.TrueParams, probecounter = self.ProbeCounter, probelist= [self.ProbeState], solver='scipy', trotter=True)
        #Here you can turn off the debugger change the resampling threshold etc...

        self.Updater = qi.SMCUpdater(self.GenSimModel, self.NumParticles, self.Prior , resample_thresh=self.ResamplerTresh , resampler = qi.LiuWestResampler(a=0.95), debug_resampling=False)
        
        #doublecheck and coment properly
        self.Inv_Field = [item[0] for item in self.GenSimModel.expparams_dtype[1:] ]
        #print('Inversion fields are: ' + str(self.Inv_Field))
        self.Heuristic = mpgh.multiPGH(self.Updater, self.SimOpList, inv_field=self.Inv_Field)
        #print('Heuristic output:' + repr(self.Heuristic()))
        self.ExpParams = np.empty((1, ), dtype=self.GenSimModel.expparams_dtype)
        
        self.NumExperiments = 0
        if checkloss == True:     
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
    def UpdateModel(self, n_experiments, sigma_threshold=10**-13,checkloss=False):
   
    #Insert check and append old data to a storing list like ExperimentsHistory or something similar. 
        self.NumExperiments = n_experiments
        if checkloss == True: 
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
    
        for istep in range(self.NumExperiments):
            self.Experiment = self.Heuristic()
            self.NumExperimentsToDate += 1
            #print('Chosen experiment: ' + repr(self.Experiment))
            if istep == 0:
                print('Initial time selected > ' + str(self.Experiment[0][0]))
            
            
            self.TrackTime[istep] = self.Experiment[0][0]
            
            self.Datum = self.GenSimModel.simulate_experiment(self.SimParams, self.Experiment)
            
            #print(str(self.GenSimModel.ProbeState))
            
            self.Updater.update(self.Datum, self.Experiment)

            if len(self.Experiment[0]) < 3:
                covmat = self.Updater.est_covariance_mtx()
                self.VolumeList = np.append(self.VolumeList, covmat)
            else:
                #covmat = self.Updater.region_est_ellipsoid()
                covmat = self.Updater.est_covariance_mtx()
                self.VolumeList = np.append(self.VolumeList,  np.linalg.det( sp.linalg.sqrtm(covmat) )    )
            
            """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!this one is probably to remove from here, as superfluous????????????????????????????"""
            self.Heuristic = mpgh.multiPGH(self.Updater, self.SimOpList, inv_field=self.Inv_Field)
            
            self.TrackEval.append(self.Updater.est_mean())
            self.Covars[istep] = np.linalg.norm(self.Updater.est_covariance_mtx())
            self.Particles[:, :, istep] = self.Updater.particle_locations
            self.Weights[:, istep] = self.Updater.particle_weights
            self.TrackLogTotLikelihood = np.append(self.TrackLogTotLikelihood, self.Updater.log_total_likelihood)

            self.NewEval = self.Updater.est_mean()
            
                
            if checkloss == True: 
                self.NewLoss = eval_loss(self.GenSimModel, self.NewEval, self.TrueParams)
                self.QLosses[istep] = self.NewLoss[0]
            
                if self.NewLoss[0]<(10**(-17)):
                    if self.debugSave: 
                        self.debug_store()
                    print('Final time selected > ' + str(self.Experiment[0][0]))
                    print('Exiting learning for Reaching Num. Prec. -  Iteration Number ' + str(istep))
                    for iterator in range(len(self.FinalParams)):
                        self.FinalParams[iterator]= [np.mean(self.Particles[:,iterator,istep]), np.std(self.Particles[:,iterator,istep])]
                        print('Final Parameters mean and stdev:'+str(self.FinalParams[iterator])) 
                    self.LogTotLikelihood=self.Updater.log_total_likelihood                
                    self.QLosses=(np.resize(self.QLosses, (1,istep)))[0]
                    self.Covars=(np.resize(self.Covars, (1,istep)))[0]
                    self.Particles = self.Particles[:, :, 0:istep]
                    self.Weights = self.Weights[:, 0:istep]
                    self.TrackTime = self.TrackTime[0:istep] 
                    break
            
            if self.Covars[istep]<self.SigmaThresh:
                if self.debugSave: 
                    self.debug_store()
                print('Final time selected > ' + str(self.Experiment[0][0]))
                print('Exiting learning for Reaching Cov. Norm. Thrshold of '+ str(self.Covars[istep]))
                print(' at Iteration Number ' + str(istep)) 
                for iterator in range(len(self.FinalParams)):
                    self.FinalParams[iterator]= [np.mean(self.Particles[:,iterator,istep]), np.std(self.Particles[:,iterator,istep])]
                    print('Final Parameters mean and stdev:'+str(self.FinalParams[iterator]))
                self.LogTotLikelihood=self.Updater.log_total_likelihood
                if checkloss == True: 
                    self.QLosses=(np.resize(self.QLosses, (1,istep)))[0]
                
                self.Covars=(np.resize(self.Covars, (1,istep)))[0]
                self.Particles = self.Particles[:, :, 0:istep]
                self.Weights = self.Weights[:, 0:istep]
                self.TrackTime = self.TrackTime[0:istep]
                
                break
            
            
            ####Need to ADD check with dereivative of sigmas!!!!
            
            
            
            if checkloss == True:
                self.QLosses[istep] = self.NewLoss[0]
            
            if istep == self.NumExperiments-1:
                print('Final time selected > ' + str(self.Experiment[0][0]))
                self.LogTotLikelihood=self.Updater.log_total_likelihood
                if self.debugSave: 
                    self.debug_store()
        
                for iterator in range(len(self.FinalParams)):
                    self.FinalParams[iterator]= [np.mean(self.Particles[:,iterator,istep-1]), np.std(self.Particles[:,iterator,istep-1])]
                    print('Final Parameters mean and stdev:'+str(self.FinalParams[iterator]))
            

            if debug_print:
                print("step ", istep)
                print( " has params: ", self.NewEval)
                print(" log tot like  : ", self.TrackLogTotLikelihood[-1])


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
