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
        sigmas = np.repeat(0.1, NumMulti)
        
    cov_matrix = np.diag(sigmas**2)
    OutputDistribution = qi.MultivariateNormalDistribution(mean, cov_matrix)
    return(OutputDistribution)
    
    
    
# Specific distributions for Ising type model development. 

def means_sigmas_ising_term(term, specific_terms={}, 
    rotation_mean=0.5, rotation_sigma=2, 
    hyperfine_mean=2.5, hyperfine_sigma=0.5,
    transverse_mean=0.5, transverse_sigma=1.0
):
    """
    Get means and sigmas for models in Ising type configurations
        ie only valid for 2 qubit systems following Ising convention. 
        to provide specific values of mean/sigma for terms, 
        pass {term:[mean, sigma]}, eg: specific_terms = {'xTi' : [0.5, 0.2]}.

    """

    individual_terms = term.split('PP')
    num_params = len(individual_terms)
    means = []
    sigmas = []

    rotation_terms = ['xTi', 'yTi', 'zTi']
    hyperfine_terms = ['xTx', 'yTy', 'zTz']
    transverse_terms = ['xTy', 'xTz', 'yTz']
    
    for k in individual_terms:
        if k in specific_terms:
            mean = specific_terms[k][0]
            sigma = specific_terms[k][1]
            means.append(mean)
            sigmas.append(sigma)
        elif k in rotation_terms:
            means.append(rotation_mean)
            sigmas.append(rotation_sigma)
        elif k in hyperfine_terms:
            means.append(hyperfine_mean)
            sigmas.append(hyperfine_sigma)
        elif k in transverse_terms:
            means.append(transverse_mean)
            sigmas.append(transverse_sigma)

    return num_params, np.array(means), np.array(sigmas)

def normal_distribution_ising(term, specific_terms={}):
    num_params, means, sigmas = means_sigmas_ising_term(
        term = term, 
        specific_terms = specific_terms
    )
    cov_matrix = np.diag(sigmas**2)
    
    dist = qi.MultivariateNormalDistribution(means, cov_matrix)

#    print("[Normal] Given term ", term,
#        ", generated \nmeans:\n", means, "\nsigmas:\n", sigmas
#    )
    return dist
    
def uniform_distribution_ising(term, specific_terms={}, 
    lower_rotation=-0.5, upper_rotation=1.5,
    lower_hyperfine=-0.5, upper_hyperfine=1.5, 
    lower_transverse=-0.5, upper_transverse=1.5
):

    rotation = [lower_rotation, upper_rotation]
    hyperfine = [lower_hyperfine, upper_hyperfine]
    transverse = [lower_transverse, upper_transverse]
    
    single_qubit_terms = ['x', 'y', 'z']
    rotation_terms = ['xTi', 'yTi', 'zTi']
    hyperfine_terms = ['xTx', 'yTy', 'zTz']
    transverse_terms = ['xTy', 'xTz', 'yTz']
    
    if term in single_qubit_terms:
        # For use in QHL tests, possibly can be removed later. -BF
        limits = np.array([[-0.5, 1.5]])
        DistroList = list(map(
            lambda limits: qi.UniformDistribution(limits), limits)
        ) 
        dist = qi.ProductDistribution(DistroList)
        return dist
    
    individual_terms = term.split('PP')
    limits = []

    for k in individual_terms:
        if k in specific_terms:
            limits.append(specific_terms[k])
        elif k in rotation_terms:
            limits.append(rotation)
        elif k in hyperfine_terms:
            limits.append(hyperfine)
        elif k in transverse_terms:
            limits.append(transverse)
        else:
            print("Term", k, "not recongised as rotation, hyperfine or transverse") 

    limits = np.array(limits)
    DistroList = list(map(
        lambda limits: qi.UniformDistribution(limits), limits)
    ) 
    dist = qi.ProductDistribution(DistroList)

#    print("[Uniform] Given term ", term, ", generated limits:\n", limits)
    return dist
