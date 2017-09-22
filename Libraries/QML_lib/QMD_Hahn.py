import numpy as np
import scipy as sp
from Utils import *
from QML_Hahn import *
from Distrib import *
from BayesF import *
from EvalLoss import *
import time as time
import Evo as evo
import ProbeStates as pros
import HahnSimQMD as gsi
import multiPGH as mpgh
import warnings as warnings




class ModelsDevelopmentClass():
    def __init__(self, fulloplist, allnameslist, maxmodnum = 6, checkloss=False, trotter=True, gaussian=False, IQLE=True, datasource='sim' ):
        
        self.MaxModNum = maxmodnum
        self.ModelsList=[None]*self.MaxModNum
        """CHANGE ME"""
        self.ModsOpList=[]
        self.mODS=[None]*self.MaxModNum
        self.ModelNames = allnameslist
        
        
        """CHANGE ME"""
        self.DiscModelsList = []
        self.DiscModsOpList = []
        self.DiscModelNames = []
        self.DiscModelDict = {}
        
        
        self.BayesFactorsList=[]
        self.BayesFactorDictionary=[]
        self.BayesFactorNames=[]        

        self.checkloss=checkloss
        self.trotter=trotter
        self.gaussian=gaussian
        self.IQLE=IQLE
        self.datasource = datasource



        #generate a list of all the possible lists of operators used in models
        self.AllOpListsList = []
        self.AllOpListsList.extend(fulloplist)
        
        
        #generate the truemodel
           
        self.AvailableModsOpList = self.AllOpListsList
        
        
        self.TrueParamNum = len(self.AllOpListsList[-1])
        self.TrueOpList = self.AllOpListsList[-1]
        
        
        ## WARNING, BOOTSTRAPPING IS HERE
        #generation of true parameters, for the moment the initialisation is bootstrapped "by hand"
        
        param_bootstrap = np.array([-2.7, -2.7, -2.14, 3., 1., 1.])
        
        # self.param_scale = np.array([0.5, 0.5, 0.5, 3, 3, 3])
        self.param_scale = np.array([0.2, 0.2, 0.2, 2, 2, 2])
        
        self.TrueParamsList= np.array([ np.random.normal(loc=param_bootstrap, scale=self.param_scale, size=self.TrueParamNum) ])

        
        
        self.TrueNames = allnameslist[-1]
        



#       Operator list for the different Models considered

        

        self.DiscardedModelNames = []
        self.DiscardedModelDict = []


        #generation of starting simulator parameters for the different Models considered 
        
        self.ModsParamsList=[]
        self.ModsParamsSigm=[]
        
        for i in range(len(self.AvailableModsOpList)):
            par_length = len(self.AvailableModsOpList[i])
            if i==0 or i==2:
                # gen_params = np.random.random([par_length])
                # self.ModsParamsList.append( gen_params )
                gen_params = np.random.normal(loc=param_bootstrap[0:par_length], scale=5*self.param_scale[0:par_length], size=par_length)
                self.ModsParamsList.append( gen_params )
                
            elif i==1:
                # gen_params = np.random.random([par_length])
                # self.ModsParamsList.append( gen_params )
                gen_params = np.random.normal(loc=param_bootstrap[0:par_length], scale=5*self.param_scale[0:par_length], size=par_length)
                self.ModsParamsList.append( gen_params )
            
            elif i==3:
                gen_params = np.random.normal(loc=param_bootstrap[0:par_length], scale=5*self.param_scale[0:par_length], size=par_length)
                self.ModsParamsList.append( gen_params )
                
            elif i==4:
                gen_params = np.random.normal(loc=param_bootstrap[0:par_length], scale=5*self.param_scale[[0,2,4]], size=par_length)
                self.ModsParamsList.append( gen_params )
                
            else:
                gen_params = np.random.normal(loc=param_bootstrap, scale=self.param_scale, size=par_length)
                self.ModsParamsList.append( gen_params )
              
            
        
        self.TrueHam = evo.getH(self.TrueParamsList, self.TrueOpList)

        
        
        #instantiate the models 
        for i in range(self.MaxModNum):
            self.ModelsList[i]=ModelLearningClass()
        
        
        #initilize starting models 
        #for the moment we initilise them all 

        
    # def InitiliaseAllModels(self,inputpartnum=400):
    def InitialiseAllActiveModels(self,inputpartnum=400):
        self.PartNum=inputpartnum
        del_list=[]
        
        warnings.warn("Did you adopt the corrct 'pr0fromHahn' definition???!!!")

        for i in range(self.MaxModNum):
            if i == 0:
                gaussianstart = self.gaussian
            else:
                gaussianstart = True
            self.ModelsList[i].InitialiseNewModel(trueoplist=self.TrueOpList, modeltrueparams=self.TrueParamsList,simoplist=self.AvailableModsOpList[i], simparams=np.array([self.ModsParamsList[i]]), sigmaparams = self.param_scale, numparticles=self.PartNum, resample_thresh=0.5, checkloss=self.checkloss, trotter=self.trotter, gaussian=gaussianstart, IQLE=self.IQLE, datasource = self.datasource)
        
        # CHANGE ME
            self.ModsOpList.append(self.AvailableModsOpList[i])
            del_list.append(i)
            
        del_list.reverse()
        
        for i in del_list:
            del(self.AvailableModsOpList[i])
            del(self.ModsParamsList[i])    
        

        """CHANGE ME"""
        self.ModelDict ={key:value for key, value in zip(self.ModelNames, self.ModsOpList)}

    
  


    


        
    def UpdateAllActiveModels(self, expnum, sigma_threshold=10**-6):
            
        for i in range(len(self.ModelsList)):
            start=time.clock()
            self.ModelsList[i].UpdateModel(n_experiments=expnum,sigma_threshold=sigma_threshold,checkloss=self.checkloss)
            
            end=time.clock()
            # #print('True model was: '+ str(self.TrueOpList)) 
            # print('True parameters were: '+ str(self.TrueParamsList)) 
            print('Batch single time '+ str(i) +' elapsed time: ' + str(end-start) + '\n\n')



    def UpdateAllLogLikelihoods(self, datalikesize = 20):
        ## Update Bayesfactor from data obtained from other sets    
        
        tpool = self.DataPool(datalikesize)
        
        start=time.clock()
        for i in range(len(self.ModelsList)):
            self.ModelsList[i].KLogTotLikelihood = LogL_UpdateCalc(self.ModelsList[i], i, tpool)
            print("KLogTotLikelihood update" + str(i) + " done")
        end=time.clock()
        print('New LogTotLikelihoods update! Elapsed time : ' + str(end-start))
    
    
    
    def UpdatePairModelLogLikelihoods(self, Model1Position, Model2Position, tpool):
    	self.ModelsList[Model1Position].KLogTotLikelihood = LogL_UpdateCalc(self.ModelsList[Model1Position], Model1Position, tpool)
    	self.ModelsList[Model2Position].KLogTotLikelihood = LogL_UpdateCalc(self.ModelsList[Model2Position], Model2Position, tpool)    
    	print('LogTotLikelihoods updated')    
    
    
  
    
    # Create a list (tpool) containing the evolution times of all the models considered in the ModelsList
    def DataPool(self, stepnum):
        
        tpool = np.empty([len(self.ModelsList), stepnum])
        #dpool = np.empty([len(self.ModelsList), stepnum, len(self.TrueHam) ])
        for i in range(len(self.ModelsList)):
            tpool[i,:] = self.ModelsList[i].TrackTime[-stepnum-1:-1]
            #dpool[i,:,0] = self.ModelsList[i].TrackTime[-stepnum:-1]
        #tpool = tpool.reshape(np.product(tpool.shape))
        return tpool
        #return([tpool, dpool])
    
        
    def ComputeAllBayesFactors(self, fromLogL = True):
        
        lst=np.arange(len(self.ModelsList))
        outlst = np.empty(0)
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i is not j:
                    outlst = np.append(outlst, np.array([lst[i],lst[j]]) )
        
        self.BayesFactorDictionary=outlst

        for i in range(int(len(outlst)/2)):
            #first two numbers are telling us which models in the ModelsList we are comparing the third number is the Bayes factorvalue for the two models under consideration
            #print('Iteration'+str(i)+' gives '+str(int(outlst[2*i]))+' and '+str(int(outlst[2*i+1])))
            self.BayesFactorNames.append("")
            self.BayesFactorNames[-1]= self.ModelNames[int(outlst[2*i])]+" VS "+self.ModelNames[int(outlst[2*i+1])]
            
            if fromLogL is False:
                self.BayesFactorsList.append(BayesFactorCalc(self.ModelsList[int(outlst[2*i])],self.ModelsList[int(outlst[2*i+1])]))
            else:
                self.BayesFactorsList.append(BayesFactorfromLogL(self.ModelsList[int(outlst[2*i])].KLogTotLikelihood, self.ModelsList[int(outlst[2*i+1])].KLogTotLikelihood) )
        
        return {key:value for key, value in zip(self.BayesFactorNames,self.BayesFactorsList)}
        
          
    def ChampionPosition(self):
        
        self.ComputeAllBayesFactors()
        bayeslocal = np.array(self.BayesFactorsList)
        bayessorted=np.reshape(bayesllocal,(int(bayesllocal.size/2),2))
        champion = []
        champ_threshold=1000
        for i in range(int(len(modeltest.BayesFactorsList)/2)):
            if np.all(bayessorted[i,:] > champ_threshold):
                champion.append(i)
        return(champion)
            

        # def ChampionPruning(self):
            # championpos=self.ChampionPosition()
            # if championpos == 
            
            
        


        self.ComparisonModelList = np.zeros
        for i in range(0,len(self.BayesFactorsList)):
            len(self.ModelsList)
    
    
    
    
    
    """Removes a single model if its 'weight' in terms of the LogLikelihood falls below a threshold"""
    
    def FloorPruningRule(self, floor_thresh = 0.15): #len(self.ModelsList)
    
        for i in range(len(self.ModelsList)):
            if(self.ModelsList[i].KLogTotLikelihood is []):
                warn("\nFloorPruningRule called before a list 'KLogTotLikelihood' was made available", UserWarning)
        
        array_KLogTotLikelihood = np.array(list(map(lambda model: model.KLogTotLikelihood, self.ModelsList)))
        renorm_KLogTotLikelihood = abs(1/array_KLogTotLikelihood)/abs(sum(1/abs(array_KLogTotLikelihood) ))
    
        del_list = []
        
        for i in range(len(self.ModelsList)):
            if renorm_KLogTotLikelihood[i] < floor_thresh:
                self.DiscModelsList.append(self.ModelsList[i])
                self.DiscModsOpList.append(self.ModelsList[i])
                self.DiscModelNames.append(self.ModelNames[i])
                self.DiscModelDict.update({ self.ModelNames[i]: self.ModelDict[self.ModelNames[i]] })
                
                del_list.append(i)
        
        del_list.sort(reverse=True)  
        
        for index in del_list:
            print('Model ' + str(self.ModelNames[index]) + ' discarded upon FloorPruningRule')
            
            del(self.ModelsList[index])
            del(self.ModsOpList[index])
            del(self.ModelDict[self.ModelNames[index] ])
            del(self.ModelNames[index])
            

    def DetectSaturation(self, use_datalength = 10, saturate_STN=3.):                  
    
        detect_events = [False]*len(self.ModelsList)
        
        for i in range( len(self.ModelsList) ):
        
            volume_list = self.ModelsList[i].VolumeList
            volume_data = volume_list[0:use_datalength]
                    
            slope = np.polyfit(range(len(volume_data)), volume_data, deg=1)[0]
            volume_data_flatten = np.array([ list(map(lambda i: volume_data[i]  - slope*i, range( len(volume_data) ) )) ]) 
            noise = np.std(volume_data_flatten)
            
            if abs(noise)>abs(saturate_STN*slope):
                detect_events[i] = True
                
        return(detect_events)

