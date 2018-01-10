import numpy as np
import itertools
import Evo as evo

#Function which append a Matrix to a list of Matrices
#Need to be added a check if the size of the old matrices matches the size of the new matrices
def MatrixAppendToArray(OldMatricesList, NewMatrix):
    NewMatricesList = np.append(OldMatricesList, NewMatrix)
    NewMatricesList = np.reshape(NewMatricesList, (len(OldMatricesList)+1,np.shape(NewMatrix)[0],np.shape(NewMatrix)[1]))
    return(NewMatricesList)
    


#Function to generate the models which are all the combinations of some basic models generators

def PossibleOPListCombGen(singoplist):
    newoplist=[]
    for i in range(1,len(singoplist)+1):
        els = (np.array([
            list(x) for x in itertools.combinations(singoplist, i)
        ]))
        newoplist.append(els)
    newlist=[]
    for i in range(len(newoplist)):
        for j in range(len(newoplist[i])):
            newlist.append(newoplist[i][j])    
    
    return(newlist)
    
    
def BuildPossiblePool(qbitlist):
    newoplist=[]
    for i in range(1,len(qbitlist)+1):
        els = ([
            list(x) for x in itertools.combinations(qbitlist, i)
        ])
        newoplist.append(els)
    newlist=[]
    for i in range(len(newoplist)):
        for j in range(len(newoplist[i])):
            newlist.append(newoplist[i][j])    
    
    return(newlist)

    
## Funciton to evaluate losses ##########################################################

def eval_loss(
        model, est_mean, true_mps=None,
        true_model=None, true_prior=None
    ):
    
    if true_model is None:
        true_model = model

    if true_mps is None:
        true_mps = true_model.update_timestep(
            promote_dims_left(true_mps, 2), expparams
        )[:, :, 0]

    if model.n_modelparams != true_model.n_modelparams:
        raise RuntimeError("The number of Parameters in True and Simulated model are different.")
                           
    n_pars = model.n_modelparams

    delta = np.subtract(*qi.perf_testing.shorten_right(est_mean, true_mps))
    loss = np.dot(delta**2, model.Q[-n_pars:])

    return loss


## Functions to name models ##########################################################    
    
def PauliNames():
    return {"sx":evo.sigmax(), "sy":evo.sigmay(), "sz":evo.sigmaz()}

def OpIndex(lst, dictionary):
    red_oplist = list(dictionary.values())
    indices = list(map(lambda op: 
         [i for i, x in enumerate(lst) if np.all(x==op)]
         , red_oplist))
    return indices

def ModelNamesPauli(full_lst, dictionary):
    base_names = list(PauliNames().keys())
    model_names = []
    for i in range(len(full_lst)):
        model_names.append("")
        indices =  OpIndex(full_lst[i], dictionary)
        for j in range(len(dictionary)):
            if len(indices[j])>0:
                model_names[i]=base_names[j]+"_"+model_names[i]
    return model_names
        
        
## Funciton for fit quality ##########################################################
    