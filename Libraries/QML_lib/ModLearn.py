import qinfer as qi
import numpy as np
import scipy as sp
import Evo as evo


def MatrixAppendToArray(OldMatricesList, NewMatrix):
    NewMatricesList = np.append(OldMatricesList, NewMatrix)
    NewMatricesList = np.reshape(NewMatricesList, (len(OldMatricesList)+1,np.shape(NewMatrix)[0],np.shape(NewMatrix)[1]))
    return(NewMatricesList)
    

# CLASS MODEL LEARNING MODEL


 
    """OpList is thje array of the Operators which compose the Hamiltonian 
    
    TrueParams (array) are the corresponding True values which together with OpList let us write the Hamiltonian 
    H= Sum OpList[i].TrueParams[i]
    
    Particles is the array with all the particles contained in this model, each element of this array is an array
    with length equal to the number of operators/Trueparams in the Hamiltonian

    Weights are the weights of the particles
    
    BayesFactorList is the list with all the BayesFactors of this model
    
    VolumeList array containing Volume vs number of experiments
    
    NumParticles is the number of particles
    
    
    """
    
class ModelLearningClass():
    def  __init__(self):
        self.OpList = np.array([])
        self.ExpParams = np.array([])
        self.Particles = np.array([])
        self.Weights = np.array([])
        self.BayesFactorList = np.array([])
        self.AvgLikelihoodList = np.array([])
        self.VolumeList = np.array([])
        self.NumParticles = len(self.Particles)

    def InsertNewOperator(self, NewOperator):
        self.NumParticles = len(self.Particles)
        self.OpList = MatrixAppendToArray(self.OpList, NewOperator)
        self.Particles =  np.random.rand(self.NumParticles,len(self.OpList))
        self.ExpParams = np.append(self.ExpParams, 1)
        self.Weights = np.full((1, self.NumParticles), 1./self.NumParticles)
    
    
       
        
"""        
ciccio=ModelLearningClass()

ciccio.Particles = np.random.rand(4,1)
ciccio.NumParticles = len(ciccio.Particles)
print("ExpParams")
print(ciccio.ExpParams)
print("Particles")
print(ciccio.Particles)
print("Weights")
print(ciccio.Weights)
print("Num of particles")
print(ciccio.NumParticles)
print("OpList")
print(ciccio.OpList)

ciccio.InsertNewOperator(evo.sigmaz())

print("ExpParams")
print(ciccio.ExpParams)
print("Particles")
print(ciccio.Particles)
print("Weights")
print(ciccio.Weights)
print("Num of particles")
print(ciccio.NumParticles )
print("OpList")
print(ciccio.OpList)"""