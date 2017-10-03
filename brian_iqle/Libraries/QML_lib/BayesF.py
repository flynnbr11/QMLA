import numpy as np



# These next two function should be removed unless we understand they do something meaningful
def BayesFactorCalc(Model1, Model2):
    LogLikelihood1 = Model1.Updater.log_total_likelihood
    LogLikelihood2 = Model2.Updater.log_total_likelihood
    #print(str(LogLikelihood1))
    #print(str(LogLikelihood2))
    BayesFactorValue = np.expm1(LogLikelihood1-LogLikelihood2)+1
    return(BayesFactorValue)


def BayesFactorCalcLin(Model1, Model2):
    Likelihood1 = np.sum(Model1.Updater.normalization_record)
    Likelihood2 = np.sum(Model2.Updater.normalization_record)
    print(str(Likelihood1))
    print(str(Likelihood2))
    BayesFactorValue = Likelihood1/Likelihood2
    return(BayesFactorValue)
    
    


def BayesFactorfromLogL(LogLikelihood1, LogLikelihood2):
    BayesFactorValue = np.expm1(LogLikelihood1-LogLikelihood2)+1
    return(BayesFactorValue)
    
    
    
    
    
#  It calculates the likelihoods of one model with the experiments of all the others, including in tpool, which is the list of all the experiments you want to consider
#  tpool list of the time evolutions used in the models you want to confront to. It can be optained using the function DataPool or DataPairPool in the QMD class  
def LogL_UpdateCalc(Kmodel, idMod, tpool, Kupdater = None):
    
    """
    Kmodel from QMD class 
        e.g. modeltest.ModelsList[i] where
    idMod identifies the specific instance 
        e.g. idMod = i
    tpool from decision node collecting all experiments
        e.g. modeltest.DataPool(Nsamples)
    Kupdater is a safe copy of the updater to prevent altering a learning process from the batch_update application,
    using Kupdater only for BayesFactor comparisons
    """ 
    if Kupdater is None:
        Kupdater = Kmodel.Updater
    
    mytpool = np.empty(0)
    for j in range( len(tpool) ):
        if idMod is not j:
            mytpool=np.append(mytpool, tpool[j])
    #print('mytpool len: ', len(mytpool))
    
  
    """This can be adopted alternatively to sample the frequencies"""
    #myexperiments = np.array(list(map(lambda x: Kmodel.Heuristic(), range(len(tpool))) ) )
    
    myexperiments = np.empty((len(mytpool), ), dtype=Kmodel.GenSimModel.expparams_dtype)       #initialises the experiments to perform for the update
    
    myexperiments['t']  = mytpool
    
    inv_field = [item[0] for item in Kmodel.GenSimModel.expparams_dtype[1:] ]
    for i in range(len(inv_field)):
        myexperiments[inv_field[i]] = Kmodel.NewEval[i]
    
    mysimparams = Kmodel.SimParams
    
    mydata = Kmodel.GenSimModel.simulate_experiment(Kmodel.SimParams, myexperiments)[0][0]
    Kupdater.batch_update(mydata, myexperiments, resample_interval=100)
    
    LogLikelihood = np.sum(Kupdater.log_total_likelihood)
    
    return(LogLikelihood)