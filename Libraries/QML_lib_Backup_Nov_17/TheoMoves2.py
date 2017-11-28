import numpy as np
import itertools as itr
import Evo as evo

import os as os
import sys as sys 
import pandas as pd
import warnings

#Activate warnings always
warnings.simplefilter('always')

################################
######### Definitions
################################

global mycolumns
mycolumns = ['<Name>' , 'N_Qbit' , 'N_params' , 'Status' , 'Selected' , 'LogL_Ext' , 'QML_Class' , 'Origin_epoch' ,'All_Operators']

global spinnames
spinnames  = ['x', 'y', 'z']

global MaxParamRate
MaxParamRate = 3

paulilist = evo.paulilist()




################################
######### Operator list operations
################################

def BuildAll_MultiParam_SameQ(singoplist):
    newoplist=[]
    newnamelist=[]
    
    for i in range(2,len(singoplist)+1):
        els = (np.array([
            list(x) for x in itr.combinations(singoplist, i)
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
            list(x) for x in itr.combinations(singoplist, i)
        ]))
        nls = [ list(x) for x in itr.combinations(singopnames, i) ]
        
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
    
    [newlist, newnamelist] = [ list(itr.compress(newlist, scalars)), 
    list(itr.compress(newnamelist, scalars)) ]
    
    return([newlist, newnamelist])
    

## TEST IT FOR DIMENSIONS HIGHER THAN 2

def AdjustDimSingOP(singop, dim, inter_qbits):

    for i in range(dim):
        if i in inter_qbits:
            myop = singop
        else:
            myop = np.eye(2)
        if i == 0:
            newop = myop
        else:
            newop = np.kron(newop,myop)
    
    return newop

def AdjustDimInterOP(twooplist, dim, inter_qbits):

    for i in range(dim):
        if i in inter_qbits:
            myop = twooplist[inter_qbits.index(i)]
        else:
            myop = np.eye(2)
        if i == 0:
            newop = myop
        else:
            newop = np.kron(newop,myop)
    
    return newop    
    
def Build_Interact(dim, inter_qbits = [0,1], type = 'scalar'):
    newoplist = []
    newnamelist = []
    
    iterid = range(len(paulilist))
    
    for axis in iterid:
        newop = AdjustDimSingOP(paulilist[axis], dim, inter_qbits)
        
        newoplist.append(np.array([newop]))
        newnamelist.append(spinnames[axis][-1]+"T"+spinnames[axis][-1]+str(inter_qbits))
    
    
    if type == 'all':
        combos = [list(x) for x in itr.permutations(iterid, 2)]
        
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
    
    
def BuildSingle_Interact(dim, inter_qbits, typelist):
    """Builds the new operator for a single interaction among inter_qbits, of the type specified by typelist"""
    newoplist = []
    newnamelist = []
    
    # build the list of two Pauli (2D) operators involved in the new interaction
    twooplist = [   paulilist[i] for i in [ spinnames.index(type) for type in typelist ]  ]
    newop = AdjustDimInterOP(twooplist, dim, inter_qbits)
    
    newoplist.append(np.array([newop]))
    newnamelist.append(typelist[0]+"T"+typelist[1]+str(inter_qbits))
    
    newnamelist = [''.join(e for e in singnls if e.isalnum() ) for singnls in newnamelist]
    return([ newoplist, [newnamelist] ])
    

def AddDimwiseSingOps(extractinteractlist, extractinteractnames, extractsingoplist, dim, inter_qbits, symop = False):
    iterid = range(len(extractsingoplist))
    # print("SingOplist " + str(extractsingoplist))
    
    newoplist = [AdjustDimSingOP(extractsingoplist[axis], dim, [min(inter_qbits)]) for axis in iterid]
    extractinteractlist.extend(newoplist)
    extractinteractnames.extend([spinname+str(min(inter_qbits)) for spinname in spinnames])
    
    if symop:
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


    
    
    
################################    
######### Reading DB Operations    
################################    
    
def ExtractSing_AllList(DB, RootN_Qbit = [0]):
    extractsingoplist = (DB.loc[ [DB["N_Qbit"][i] == RootN_Qbit  and DB["N_params"][i] == 1 for i in range(len(DB["N_Qbit"]))] ])
    list_length = len(extractsingoplist)
    extractsingopnames = [ list(extractsingoplist['<Name>'])[i][0] for i in range(list_length) ]
    extractsingoplist = [list(eachlist)[0] for eachlist in extractsingoplist['All_Operators']]
    return([extractsingoplist, extractsingopnames])
    

def Find_OpEXName(DB, Name):
    """Search the DataFrame for an operator with a certain name-list, and returns the corresponding operator"""
    if type(Name) is not list:
        print("Please provide the name of the model as list")
        raise
    else:
        return(list((DB.loc[ [DB['<Name>'][i] == Name for i in range(len(DB['Status']))] ])['All_Operators']))
        
        
def CheckRootMaxParam(DB, RootN_Qbit = None):
    """Search the DataFrame for checking how many parameters are already involved for operators concerning the current qubit to expand in number of operators"""
    if RootN_Qbit is None:  #it will select the last qubit being explored
        RootN_Qbit = max(list(itr.chain(*list(DB["N_Qbit"]))))
    else:
        if type(RootN_Qbit) is not list:
            print("Please provide the name of the model as list")
            raise
        else:
            RootN_Qbit = RootN_Qbit[0]
    extract = DB.loc[ [ 0 in DB['N_Qbit'][i] and str(RootN_Qbit) in DB['<Name>'][i][0] and 'T' not in DB['<Name>'][i][0] for i in range(len(DB['Status']))] ]
    
    if len(extract['N_params']) == 0:
        return None
    else:
        return int(max(list(extract['N_params'])))
    
        


def LogLCandidate(DB):
    return
        
    
################################ 
######### Writing DB Operations
################################

def InitialiseDB(RootN_Qbit = [0]):

    return( pd.DataFrame({'<Name>' : [   [spinname+str(RootN_Qbit[0])] for spinname in spinnames], 
                         'N_Qbit' : [RootN_Qbit for i in range(len(spinnames))], 'N_params' : 1, 'Status' : 'Ready', 'Selected' : False, 
                         'LogL_Ext' : None, 'QML_Class' : None, 'Origin_epoch' : 0, 'RootNode' : 'NaN',
                        'All_Operators': list(map(lambda j: np.array([evo.paulilist()[j]]) , 
                        range(  len(evo.paulilist())  ))) 
                        })  )
                        
                        
def AddDBnewNparam_SameQ(DB, epoch, RootN_Qbit = [0], RootNode = None, MaxParamRate = MaxParamRate):
# Create parameters with additional Parameters upon the SAME qubit
    #MaxN_paramsCHK = DB.loc[ [DB["N_Qbit"][i] == RootN_Qbit  for i in range(len(DB["N_Qbit"]))] ]
    MaxN_paramsCHK = CheckRootMaxParam(DB, RootN_Qbit)
    
    if MaxN_paramsCHK is None:
        warnings.warn("Aborted: Root Qbit selected yet to initialise")
        return( DB )
    else:
        print('Max Param being analysed for highest qubit operations: ' + str(MaxN_paramsCHK))
        
        NewModels= []
        
        if MaxN_paramsCHK+1 <= MaxParamRate:
            
            [extractsingoplist, extractsingopnames] = ExtractSing_AllList(DB, RootN_Qbit)
            
            if len(list(itr.combinations(extractsingoplist, MaxN_paramsCHK+1))) > 0:
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
                        [[  [NewModels[1][i]], RootN_Qbit , len(NewModels[0][i]),  'Ready', False,  None,  None, epoch, NewModels[0][i]]], 
                        columns=mycolumns) , ignore_index=True)
    
        return( DB )
    
    

def AddInteraction_NewQ(DB, epoch, dim =2, inter_qbits = [0,1], type = 'scalar'):
# Adds HF-like interaction SINGLE operators to the DB operators list
    NewModels = Build_Interact(dim, inter_qbits, type = type)
    
    if len(NewModels) > 0:
        for i in range(len(NewModels[0])):
            if NewModels[1][i] not in list(DB["<Name>"]):
                DB = DB.append( pd.DataFrame(
                    [[[NewModels[1][i]], inter_qbits , len(NewModels[0][i]),  'Ready', False,  None,  None, epoch, NewModels[0][i]]], 
                    columns=mycolumns) , ignore_index=True)
            else:
                warnings.warn("I already initialised the " + str(inter_qbits) + " interaction term")
    
    return( DB )
    

    
def AddSingleInteraction_NewQ(DB, epoch, inter_qbits, inttype, start_model = None, dim = None):
    # Addsadd $\sigma_i \otimes \sigma_j$ interaction SINGLE operators to the DB operators list
        #  inter_qbits = [i,j] couple of interacting qbits
        #  type = ['x','y'] interacting couple of operators, to choose according to prior information or randomly        
        # start_model = NAME of the model to expand upon introduction of new interaction term
    
    # list all the qbits (removing duplicates) explored so far
    allqbitslist = list(set(itr.chain(*list(DB['N_Qbit'])))) 
    # if no dimension had been provided, find the max N_qubit
    if dim is None:
        dim = max(allqbitslist) + 1 if max(inter_qbits) in allqbitslist else max(allqbitslist) + 2
        # 1 because of 0th qubit, 1 for new operator being added if not already enlisted
    
    #check that at least one of the interacting qubits is within the allqbitslist
    checksum = sum([item in allqbitslist for item in inter_qbits])

    if checksum > 0 and max(inter_qbits)<=max(allqbitslist)+1:    #at least one qubit in the list, reasonable new qubit
        if checksum < len(inter_qbits):
            print("New Qbit introduced, expanding Hilbert space for new models to dim = " + str(dim))
            
        NewModels = BuildSingle_Interact(dim, inter_qbits, inttype)
        #print(NewModels)

        if len(NewModels) > 0:
            
            # if building on top of a previous model, enrich the name and operators accordingly
            if start_model is not None:
                startops = Find_OpEXName(DB, start_model)
                startdim = np.shape(startops[0][0])[0]
                # the previous model might live in a different Hilbert space, adjusted here
                #if startdim < dim:
                    
            
            for i in range(len(NewModels[0])):
                if NewModels[1][i] not in list(DB["<Name>"]):
                    DB = DB.append( pd.DataFrame(
                        [[NewModels[1][i], inter_qbits , len(NewModels[0][i]),  'Ready', False,  None,  None, epoch, NewModels[0][i]]], 
                        columns=mycolumns) , ignore_index=True)
                else:
                    warnings.warn("I already initialised this " + str(inter_qbits) + " interaction term. Check your Model generation schedule!")
    
    else: 
        warnings.warn("Aborted: at least one inter_qbit selected yet to initialise, or qbits being skipped")
    return( DB )

    
    
    
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

    if len(list(itr.combinations(extractinteractnames, NewN_paramsCHK-1))) > 0:  
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



    
