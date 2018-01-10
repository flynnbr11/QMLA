import numpy as np
import itertools
import Evo as evo

import os as os
import sys as sys 
import pandas as pd
import warnings


global mycolumns
mycolumns = ['<Name>' , 'N_Qbit' , 'N_params' , 'Status' , 'Selected' , 'LogL_Ext' , 'QML_Class' , 'Origin_epoch' ,'All_Operators']

global spinnames
spinnames  = ['x', 'y', 'z']

global MaxParamRate
MaxParamRate = 3


######### Operator list operations

def BuildAll_MultiParam_SameQ(singoplist):
    newoplist=[]
    newnamelist=[]
    
    for i in range(2,len(singoplist)+1):
        els = (np.array([
            list(x) for x in itertools.combinations(singoplist, i)
        ]))
        newoplist.append(els)
    newlist=[]
    for i in range(len(newoplist)):
        for j in range(len(newoplist[i])):
            newlist.append(newoplist[i][j])    
    
    return(newlist)
    
    
def BuildAll_newNparam_SameQ(singoplist, singopnames, oldNparam = 1, newNparam = None):
    if newNparam is None:
        newNparam = len(singoplist)+1

    newoplist=[]
    newnamelist=[]
    
    for i in range(oldNparam+1,newNparam):
        els = (np.array([
            list(x) for x in itertools.combinations(singoplist, i)
        ]))
        nls = [ list(x) for x in itertools.combinations(singopnames, i) ]
        
        newoplist.append(els)
        newnamelist.extend(nls)
  
    newlist=[]
    for i in range(len(newoplist)):
        newnamelist[i]=str(newnamelist[i])        
        for j in range(len(newoplist[i])):
            newlist.append(newoplist[i][j])  

    newnamelist = [''.join(e for e in singnls if e.isalnum()) for singnls in newnamelist]
    
    return([newlist, newnamelist])
    
    
def BuildScalar_newNparam_SameQ(singoplist_2, singopnames_2, oldNparam = 1, newNparam = None):
    if newNparam is None:
        newNparam = len(singoplist)+1

    [newlist, newnamelist] = BuildAll_newNparam_SameQ(singoplist_2, singopnames_2, oldNparam, newNparam)
    
    scalars = np.all(np.array([[  (modelname.count(string)==3 or modelname.count(string)==0) for 
    modelname in newnamelist] for string in ['x','y','z']]),axis=0)
    
    [newlist, newnamelist] = [ list(itertools.compress(newlist, scalars)), 
    list(itertools.compress(newnamelist, scalars)) ]
    
    return([newlist, newnamelist])
    

## TEST IT FOR DIMENSIONS HIGHER THAN 2

def AdjustDimSingOP(singop, dim, qbit_s):

    for i in range(dim):
        if i in qbit_s:
            myop = singop
        else:
            myop = np.eye(2)
        if i == 0:
            newop = myop
        else:
            newop = np.kron(newop,myop)
    
    return newop
    
    
def Build_Interact(dim, inter_qbits = [0,1], type = 'scalar'):
    paulilist = evo.paulilist()
    newoplist = []
    newnamelist = []
    
    iterid = range(len(paulilist))
    
    for axis in iterid:
        newop = AdjustDimSingOP(paulilist[axis], dim, inter_qbits)
        
        newoplist.append(np.array([newop]))
        newnamelist.append(spinnames[axis][-1]+"T"+spinnames[axis][-1]+str(inter_qbits))
    
    
    if type == 'all':
        combos = [list(x) for x in itertools.permutations(iterid, 2)]
        
        for axes in combos:
            for i in range(dim):
                if i in inter_qbits:
                    myop = paulilist[axes[0]]
                else:
                    myop = np.eye(2)
                if i == 0:
                    newop = myop
                else:
                    newop = np.kron(newop,paulilist[axes[1]])
            newoplist.append(np.array([newop]))
            newnamelist.append(spinnames[axes[0]][-1]+"T"+spinnames[axes[1]][-1]+str(inter_qbits)) 
    
    newnamelist = [''.join(e for e in singnls if e.isalnum() ) for singnls in newnamelist]
    return([newoplist, newnamelist])
    
    
def ExtractSing_AllList(DB, RootN_Qbit):
    extractsingoplist = (DB.loc[ [DB["N_Qbit"][i] == RootN_Qbit  and DB["N_params"][i] == 1 for i in range(len(DB["N_Qbit"]))] ])
    extractsingopnames = list(extractsingoplist['<Name>'])
    extractsingoplist = [list(eachlist)[0] for eachlist in extractsingoplist['All_Operators']]
    return([extractsingoplist, extractsingopnames])
    

def AddDimwiseSingOps(extractinteractlist, extractinteractnames, extractsingoplist, dim, inter_qbits, symop = False):
    iterid = range(len(extractsingoplist))
    # print("SingOplist " + str(extractsingoplist))
    
    newoplist = [AdjustDimSingOP(extractsingoplist[axis], dim, [min(inter_qbits)]) for axis in iterid]
    extractinteractlist.extend(newoplist)
    extractinteractnames.extend([spinname+str(min(inter_qbits)) for spinname in spinnames])
    
    if symop:
        print('oops')
        newoplist = [AdjustDimSingOP(extractsingoplist[axis], dim, [max(inter_qbits)]) for axis in iterid]
        extractinteractlist.extend(newoplist)
        extractinteractnames.extend([spinname+str(max(inter_qbits)) for spinname in spinnames])
    
    # print("InteractList3 " + str(extractinteractlist))
    # print("InteractNames " + str(extractinteractnames))
    
    return([extractinteractlist, extractinteractnames])
    

    ## CHANGE ME FOR BIGGER DIMENSIONS, DESIGNED SPECIFICALLY FOR 2
def BuildBase_1param_2Q(singoplist, singopnames, N_Qbit):

    newoplist= list(map(lambda j: [np.kron(np.eye(2),singoplist[j])], range(len(singopnames))))
    
    newnamelist=[spinname+str(N_Qbit) for spinname in singopnames]

    return([newlist, newnamelist])
    
    
def CheckMoreParamsAvailable(database, rootopname):

    return(database)


    
    
    
    
    
    
    
    
    
    
    
    
######### DB Operations

def InitialiseDB(RootN_Qbit = [0]):

    return( pd.DataFrame({'<Name>' : [spinname+str(RootN_Qbit[0]) for spinname in spinnames], 
                         'N_Qbit' : [RootN_Qbit for i in range(len(spinnames))], 'N_params' : 1, 'Status' : 'Ready', 'Selected' : False, 
                         'LogL_Ext' : None, 'QML_Class' : None, 'Origin_epoch' : 0,
                        'All_Operators': list(map(lambda j: np.array([evo.paulilist()[j]]) , 
                        range(  len(evo.paulilist())  ))) 
                        })  )
                        
                        
def AddDBnewNparam_SameQ(DB, epoch, RootN_Qbit = [0], MaxParamRate = MaxParamRate):
# Create parameters with additional Parameters upon the SAME qubit
    MaxN_paramsCHK = DB.loc[ [DB["N_Qbit"][i] == RootN_Qbit  for i in range(len(DB["N_Qbit"]))] ]
    
    if len(MaxN_paramsCHK['N_params']) == 0:
        warnings.warn("Root Qbit selected yet to initialise")
        return( DB )
    else:
        MaxN_paramsCHK = int(max(MaxN_paramsCHK['N_params']))
        print('Max Param being analysed for highest qubit operations: ' + str(MaxN_paramsCHK))
        
        NewModels= []
        
        if MaxN_paramsCHK+1 <= MaxParamRate:
            
            [extractsingoplist, extractsingopnames] = ExtractSing_AllList(DB, RootN_Qbit)
            
            if len(list(itertools.combinations(extractsingoplist, MaxN_paramsCHK+1))) > 0:
                NewModels = BuildAll_newNparam_SameQ(extractsingoplist, extractsingopnames, MaxN_paramsCHK, MaxN_paramsCHK+2)
            else:
                warnings.warn("The tree exploration has reached the highest N_params for " + str(RootN_Qbit) + " qubit")
                # include here MOVE to stop algorithm or add 1 qubit
        
        else:
            warnings.warn("The tree exploration has reached the highest N_params for " + str(RootN_Qbit) + " qubit")
            
        if len(NewModels) > 0:
            for i in range(len(NewModels[0])):
                if NewModels[1][i] not in list(DB["<Name>"]):
                    DB = DB.append( pd.DataFrame(
                        [[NewModels[1][i], RootN_Qbit , len(NewModels[0][i]),  'Ready', False,  None,  None, epoch, NewModels[0][i]]], 
                        columns=mycolumns) , ignore_index=True)
    
        return( DB )
    
    

def AddInteraction_NewQ(DB, epoch, dim =2, inter_qbits = [0,1], type = 'scalar'):
# Adds HF-like interaction SINGLE operators to the DB operators list
    NewModels = Build_Interact(dim, inter_qbits, type = type)
    
    if len(NewModels) > 0:
        for i in range(len(NewModels[0])):
            if NewModels[1][i] not in list(DB["<Name>"]):
                DB = DB.append( pd.DataFrame(
                    [[NewModels[1][i], inter_qbits , len(NewModels[0][i]),  'Ready', False,  None,  None, epoch, NewModels[0][i]]], 
                    columns=mycolumns) , ignore_index=True)
            else:
                warnings.warn("I already initialised the " + str(inter_qbits) + " interaction term")
    
    return( DB )
    
    
    

    
    
    
def AddComboOp_NewQ(DB, epoch, dim =2, inter_qbits = [0,1], MaxParamRate = MaxParamRate, type = 'scalar', symop=False):
# Adds combinations of HF-like interactions along with the single operators of a NEW qubit system
# to use once HF operators have already been introduced for the corresponding Qubit
# !!!TODO!!! it is enough to replace "BuildScalar_newNparam_SameQ" with "BuildAll_newNparam_SameQ" in order to have ALL the combinations as above

    
    # The number of qubits currently entertained
    ObsNqubit = max([len(Nqubit) for Nqubit in list(DB["N_Qbit"])])
    # Max number of pairwise HF interactions to be expected given the N qubits entertained
    # (use it for chcking that the number of terms does not overcome the theoretical MAX)
    N_pairwise_interact = ObsNqubit*(ObsNqubit-1)/2*MaxParamRate
    
    # Find the maximum parameter currently entertained for combinations of operators as above
    checks = [(ObsNqubit-1 in DB["N_Qbit"][i]) for i in range(len(DB["N_Qbit"]))]
    MaxN_paramsCHK = DB.loc[checks ] 
    MaxN_paramsCHK = int(max(MaxN_paramsCHK['N_params']))
    print('Max Param being analysed for operators combinations: ' + str(MaxN_paramsCHK))
    
    # Extract the basic operators and names for the interactions
    # TODO: this may be replaced by a SETTINGS dictionary whenever the interactions are identical among the various interacting qubits
    [extractsingoplist, extractsingopnames] = ExtractSing_AllList(DB, [0])
    
    extractinteractlist = (DB.loc[ [DB["N_Qbit"][i] == inter_qbits  and DB["N_params"][i] == 1 for i in range(len(DB["N_Qbit"]))] ])
    extractinteractnames = list(extractinteractlist['<Name>'])
    extractinteractlist = [list(eachlist)[0] for eachlist in extractinteractlist['All_Operators']]
    
    # add to the interact operators the usual basic spin Pauli operators
    [extractinteractlist, extractinteractnames] = AddDimwiseSingOps(extractinteractlist, extractinteractnames, extractsingoplist, dim, inter_qbits, symop)
        
    # this avoids odd number of operators where HF_i is not matched by any sigma_i 
    # TODO> check that the control makes sense in your specific evolution
    NewN_paramsCHK = MaxN_paramsCHK+2 if MaxN_paramsCHK%2 == 1 else MaxN_paramsCHK+3

    if len(list(itertools.combinations(extractinteractnames, NewN_paramsCHK-1))) > 0:  
        # print("New attempted N_param " + str(NewN_paramsCHK-1))
        if type == 'scalar':
            NewModels = BuildScalar_newNparam_SameQ(extractinteractlist, extractinteractnames, NewN_paramsCHK-2, NewN_paramsCHK)
        else:
            NewModels = BuildAll_newNparam_SameQ(extractinteractlist, extractinteractnames, NewN_paramsCHK-2, NewN_paramsCHK)
        
        # pick only terms that have at least a HF interaction included 
        # (based on physical intuition that terms with non interacting qubits can be excluded for the modelling
        # if not, modify accordingly!)
        newnqbit = []
        for i in range(len(NewModels[0])):
            myqubits  = inter_qbits if ('T' in NewModels[1][i]) else [int(max(inter_qbits))]
            newnqbit.append(myqubits)
        
    else:
        warnings.warn("The tree exploration has reached the highest N_params for " + str(max(inter_qbits)+1) + " qubit(s)")
        NewModels=[]
        # TODO: include here MOVE to stop algorithm or add 1 qubit
    
    

        
    if len(NewModels) > 0:
        for i in range(len(NewModels[0])):
            if NewModels[1][i] not in list(DB["<Name>"]):
                DB = DB.append( pd.DataFrame(
                    [[NewModels[1][i], newnqbit[i] , len(NewModels[0][i]),  'Ready', False,  None,  None, epoch, NewModels[0][i]]], 
                    columns=mycolumns) , ignore_index=True)
    
    return( DB )
    

    


def AddDB_1param_newQ(singoplist, singopnames, RootN_Qbit, epoch = 0):
    
    N_Qbit = RootN_Qbit[0]+1
    NewOp_and_Names = BuildBase_1param_newQ(singoplist, singopnames, N_Qbit)
    
    return(
    pd.DataFrame({'<Name>' : NewOp_and_Names[1],  'N_Qbit' : [[N_Qbit] for i in range(len(singopnames))], 'N_params' : 1, 'Status' : 'Ready', 'Selected' : False,  'LogL_Ext' : None, 'QML_Class' : None, 'Origin_epoch' : epoch, 'All_Operators':  NewOp_and_Names[0]   })
    )



    