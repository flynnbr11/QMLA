import os as os
import sys as sys 
import numpy as np
import warnings as warnings

sys.path.append(os.path.join("Libraries","QML_lib"))

import IOfuncts as mIO


def init(datatype = 'peak', directory = None):
    """
    Initialise here all the EXPERIMENTALLY obtaned/inferred parameters that are useful for the modelling
    """
    if directory is None:
        directory = os.path.join("..","..","..","Hamiltonian Learning With Seb","Decoherence_learning","NV05_HahnEcho02")
        
        warnings.warn("Using default data from > " + str(directory))
    
    
    if datatype is 'peak':
    
        global mydata

        mydata =  mIO.ImportAllFolder_Hahnpeak(directory, clean_duplicates = True)  

        myrange = range(0, min(mydata.shape[0],185)) 
        
        # the 1000 factor converts ns to us
        mydata = np.array([ [mydata[i,0]/1000 , mydata[i,1]] for i in myrange])

        offset = 0.18
        mydata[:,0] = mydata[:,0]-offset
        
        global decoTau_c 
        decoTau_c = 10 #in us
    
    
    
    
    elif datatype is 'signal':
    
        global sigdata
        
        sigdata = mIO.ImportAllFolder_Hahnsignal(directory, clean_duplicates = True)  
        
        myrange = range(0, min(sigdata.shape[0],440)) 
        
        prepydata = mIO.rescaledatatomin(sigdata[:,1], newrange = [0.5,0.99048])

        # the 1000 factor converts ns to us
        sigdata = np.array([ [sigdata[i,0]/1000 , prepydata[i]] for i in myrange])
        del prepydata

        global decoTau_c 
        decoTau_c = 5.43 #in us
        
        global decoT2
        decoT2  = 247 #in us
        
        global signal_offset
        signal_offset = 15. #in us
    
    
    print("For this run I am using data in directory: \n" + directory)
    
    warnings.warn("Remember to set the correct parameters in the SETTINGS.PY")
    
 
    
    """
    Initialise here all the SIMULATION parameters that are useful for the modelling
    """
    
    global mycolumns
    mycolumns = ['<Name>' , 'N_Qbit' , 'N_params' , 'Status' , 'Selected' , 'LogL_Ext' , 'QML_Class' , 'Origin_epoch' ,'All_Operators']

    global spinnames
    spinnames  = ['sx', 'sy', 'sz']
    
    global MaxParamRate
    MaxParamRate = 3