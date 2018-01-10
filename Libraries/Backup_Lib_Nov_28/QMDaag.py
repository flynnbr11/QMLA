import numpy as np
import scipy as sp
from Utils import *
from QMLaag import *
from Distrib import *
from BayesF import *
from EvalLoss import *
import time as time
import Evo as evo
import ProbeStates as pros
import GenSimQMD as gsi
import multiPGH as mpgh

class ModelsDevelopmentClass():
    def __init__(self, maxmodnum=3, singoplist=[evo.sigmax(),evo.sigmay(), evo.sigmaz()], checkloss=False, trotter=True, gaussian=False, IQLE=True ):
        
        self.MaxModNum = maxmodnum
        self.ModelsList=[None]*self.MaxModNum
        """CHANGE ME"""
        self.ModsOpList=[]
        self.mODS=[None]*self.MaxModNum
        self.ModelNames = None
        
        
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



        #generate a list of all the possible lists of operators used in models
        self.AllOpListsList = PossibleOPListCombGen(singoplist)
        
        
        #generate a Random truemodel
        
        coin = np.random.random_sample()
        
        #for the moment we are limiting ourselves to the cases where there are max 2 parameters
        
        
        #Only valid for 1D models
        
        if coin <0.333333333:
            self.TrueParamNum = 1
            coin2 = np.random.random_sample()
            if coin2<=1./3:
                self.TrueOpList = self.AllOpListsList[0]
            elif coin2<=2./3 and coin2>1./3:
                self.TrueOpList = self.AllOpListsList[1]
            else:
                self.TrueOpList = self.AllOpListsList[2]
        elif coin<0.66666666 and coin>=0.333333333:
            self.TrueParamNum = 2
            coin2 = np.random.random_sample()
            if coin2<=1./3:
                self.TrueOpList = self.AllOpListsList[3]
            elif coin2<=2./3 and coin2>1./3:
                self.TrueOpList = self.AllOpListsList[4]
            else:
                self.TrueOpList = self.AllOpListsList[5]
        else:
        	self.TrueParamNum = 3
        	self.TrueOpList = self.AllOpListsList[6]
        
        #generation of true parameters
        
        self.TrueParamsList= np.array([ np.random.normal(loc=0.5, scale=0.15, size=self.TrueParamNum) ])
#         self.TrueParamsList= np.array([np.random.random_sample(self.TrueParamNum)])
        
        self.TrueNames = ModelNamesPauli([self.TrueOpList], PauliNames() ) 
        



#       Operator list for the different Models considered


        self.AvailableModsOpList = self.AllOpListsList
        

        self.DiscardedModelNames =[]
        self.DiscardedModelDict = []


        #generation of starting simulator parameters for the different Models considered 
        
        self.ModsParamsList=[]
        for i in range(len(self.AvailableModsOpList)):
            self.ModsParamsList.append(np.random.random([len(self.AvailableModsOpList[i])]))
            
        
        self.TrueHam = evo.getH(self.TrueParamsList, self.TrueOpList)
       
        
        #for now we are imposing that will try the whole space of possible models for one qubit hamniltonian is 7
        #we limit to 6 because we exclude the three parameters model till it works
        
        
        
        
        
        #instantiate the models 
        for i in range(self.MaxModNum):
            self.ModelsList[i]=ModelLearningClass()
        
        
        #initilize starting models 
        #for the moment we initilise them all 
    





        
    # def InitiliaseAllModels(self,inputpartnum=400):
    def InitialiseAllActiveModels(self,inputpartnum=400):
        self.PartNum=inputpartnum
        del_list=[]
        
        start=time.clock()
        for i in range(self.MaxModNum):
            self.ModelsList[i].InitialiseNewModel(trueoplist=self.TrueOpList, modeltrueparams=self.TrueParamsList,simoplist=self.AvailableModsOpList[i],simparams=np.array([self.ModsParamsList[i]]), numparticles=self.PartNum, checkloss=self.checkloss, trotter=self.trotter, gaussian=self.gaussian, IQLE=self.IQLE)
            # CHANGE ME
            self.ModsOpList.append(self.AvailableModsOpList[i])
            del_list.append(i)
            
        del_list.reverse()
        
        for i in del_list:
            del(self.AvailableModsOpList[i])
            del(self.ModsParamsList[i])    
        

        """CHANGE ME"""
        self.ModelNames = ModelNamesPauli(self.AvailableModsOpList, PauliNames())
        self.ModelDict ={key:value for key, value in zip(self.ModelNames, self.ModsOpList)}
        
        end=time.clock()
        #print('Initilisation time: ' + str(end-start))
    
  
    


        #print('Initilisation time: ' + str(end-start))    
    
    #maxnumofop is the maximum allowed number of operators in the model
#     def InitiliseModelDevelopment(self, opgenlist, maxnumofop, n_experiments):
            
        
        
#     def GenerateNewModel(self, opgenlist, param):


    


        
    def UpdateAllActiveModels(self, expnum, sigma_threshold=10**-6,datalikesize=20):
            
        for i in range(len(self.ModelsList)):
            start=time.clock()
            self.ModelsList[i].UpdateModel(n_experiments=expnum,sigma_threshold=sigma_threshold,checkloss=self.checkloss)
            
            end=time.clock()
            # #print('True model was: '+ str(self.TrueOpList)) 
            # print('True parameters were: '+ str(self.TrueParamsList)) 
            print('Batch single time '+ str(i) +' elapsed time: ' + str(end-start) + '\n\n')
        tpool= self.DataPool(datalikesize)
        self.UpdateAllLogLikelihoods(tpool)
        
            






    def UpdateAllLogLikelihoods(self, tpool):
            
        for i in range(len(self.ModelsList)):
            self.ModelsList[i].KLogTotLikelihood = LogL_UpdateCalc(self.ModelsList[i], i, tpool)
            print('LogTotLikelihoods updated')           
    
    
    
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
            self.BayesFactorNames[-1]= str(ModelNamesPauli([self.ModelsList[int(outlst[2*i])].SimOpList], PauliNames()))+"VS"+str(ModelNamesPauli([self.ModelsList[int(outlst[2*i+1])].SimOpList], PauliNames()))
            
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

