import numpy as np
import scipy as sp
import time as time
import pandas as pd

from Utils import *
from QML import *
from Distrib import *
from BayesF import *

import Evo as evo
import ProbeStates as pros
import GenSimQMD_IQLE as gsi
import multiPGH as mpgh

class ModelsDevelopmentGraph():
    def __init__(self, oplibrary, trotterize=True, gaussian=False):
        
        
        
        
        
        self.OpenModelsList=singoplist
        self.DiscardModelsList=[]
        
        self.ModelNames = None
        self.BayesFactorsList=[]
        self.BayesFactorDictionary=[]
        self.BayesFactorNames=[]        
        self.gaussian=gaussian
        
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


        self.ModsOpList = self.AllOpListsList
        
        self.ModelNames = ModelNamesPauli(self.ModsOpList, PauliNames())
        self.ModelDict ={key:value for key, value in zip(self.ModelNames, self.ModsOpList)}
        
        #generation of starting simulator parameters for the different Models considered 
        
        self.ModsParamsList=[]
        for i in range(len(self.ModsOpList)):
            self.ModsParamsList.append(np.random.random([len(self.ModsOpList[i])]))
            
        
        self.TrueHam = evo.getH(self.TrueParamsList, self.TrueOpList)
       
        
        #for now we are imposing that will try the whole space of possible models for one qubit hamniltonian is 7
        #we limit to 6 because we exclude the three parameters model till it works
        
        
        
        
        
        #instantiate the models 
        for i in range(0, self.MaxModNum ):
            self.ModelsList[i]=ModelLearningClass()
        
        
        #initilize starting models 
        #for the moment we initilise them all 
        
        
    def InitiliaseAllModels(self,inputpartnum=400):
        self.PartNum=inputpartnum
        start=time.clock()
        for i in range(0, self.MaxModNum):
            self.ModelsList[i].InitialiseNewModel(trueoplist=self.TrueOpList, modeltrueparams=self.TrueParamsList,simoplist=self.ModsOpList[i],simparams=np.array([self.ModsParamsList[i]]), numparticles=self.PartNum, gaussian=self.gaussian)

        end=time.clock()
        #print('Initilisation time: ' + str(end-start))
    
    
    
    #maxnumofop is the maximum allowed number of operators in the model
#     def InitiliseModelDevelopment(self, opgenlist, maxnumofop, n_experiments):
            
        
        
#     def GenerateNewModel(self, opgenlist, param):

        
    def UpdateAllModels(self, expnum,sigma_threshold=10**-6):
            
        for i in range(len(self.ModelsList)):
            start=time.clock()
            self.ModelsList[i].UpdateModel(n_experiments=expnum,sigma_threshold=sigma_threshold)
            end=time.clock()
            #print('True model was: '+ str(self.TrueOpList)) 
            print('True parameters were: '+ str(self.TrueParamsList)) 
            print('Single iteration '+ str(i) +' elapsed time: ' + str(end-start))


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
    
        
    def ComputeAllBayesFactors(self, fromLogL = False):
        
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
        
          
    def ChampionPruning(self):
        
        self.ComparisonModelList = np.zeros
        for i in range(0,len(self.BayesFactorsList)):
            len(self.ModelsList)

