import qinfer as qi
import numpy as np

#Function which generate a distribution of multiple uniformly [0,1] distributed values of length NumMulti
#NumMulti usually is the number of uniform distributions we want to sample from dimultaneously, 
#usually chosen as len(OpList), where OpList is the list of operators of the Hamiltonian 
#in the model under consideration

def MultiVariateUniformDistribution(NumMulti, DistroLimits=np.array([[-0.5,1.5]]) ):
    DistroBoundsList = np.repeat(DistroLimits, [NumMulti], axis=0)
    DistroList = list(map(lambda DistroBoundsList: qi.UniformDistribution(DistroBoundsList), DistroBoundsList))
    OutputDistribution = qi.ProductDistribution(DistroList)
    return(OutputDistribution)

def MultiVariateNormalDistributionNocov(NumMulti, mean=None, sigmas=None):
    if mean is None:
        mean = np.repeat(0.5, NumMulti)
    if sigmas is None:
        sigmas = np.repeat(0.5, NumMulti)
    cov_matrix = np.diag(sigmas**2)
    OutputDistribution = qi.MultivariateNormalDistribution(mean, cov_matrix)
    return(OutputDistribution)
    
