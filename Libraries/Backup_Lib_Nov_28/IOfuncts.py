import qinfer as qi
import numpy as np
import scipy as sp
import datetime
import os as os
import time as time

def writeToFile3d(filename, nparray):
    with open(filename, 'wb') as f:
        for exp in range(np.shape(nparray)[2]):
            for part in range(np.shape(nparray)[0]):
                np.savetxt(f, nparray[part,:,exp], delimiter=',')
    f.close
    
def writeToFile2d(filename, nparray):
    with open(filename, 'wb') as f:
        for exp in range(np.shape(nparray)[1]):
            np.savetxt(f, nparray[:,exp], delimiter=',')
    f.close
    

def mytimestamp():
    timestamp = str(datetime.datetime.now()).split('.')[0]
    timestamp = "_"+timestamp.replace(" ", "_")
    timestamp = timestamp.replace(":", ".")
    return(timestamp)

    
def ImportAllFolder_Hahnpeak(directory, clean_duplicates = True):
    
    Hahn_data = []
    for root,dirs,files in os.walk(directory):
        for filename in files:  
            if filename.endswith(".csv"):
                newfilename = os.path.join(directory, filename)
                #print(os.path.abspath(newfilename))
                #usecols only selects one normalisation 
                Hahn_data.append((np.loadtxt(os.path.abspath(newfilename), delimiter=",", usecols=(0,2), skiprows=1)).tolist())
                
    Hahn_data = [item for sublist in Hahn_data for item in sublist]

    Hahn_data = np.asarray(Hahn_data)
    Hahn_data = Hahn_data[Hahn_data[:,0].argsort()] 

    if clean_duplicates:
        u, indices = np.unique(Hahn_data[:,0], return_index=True)
        Hahn_data = np.array([[Hahn_data[i, 0], Hahn_data[i, 1]] for i in indices])
        
    return(Hahn_data)
    
    
    
def ImportAllFolder_Hahnsignal(directory, clean_duplicates = True):
    
    Hahn_data = []
    for root,dirs,files in os.walk(directory):
        for filename in files:  
            if filename.endswith(".csv") and filename.startswith("ana"):
                newfilename = os.path.join(directory, filename)
                # print(os.path.abspath(newfilename))
                #usecols only selects one normalisation
                if filename.endswith("s.csv"):
                    Hahn_data.append((np.loadtxt(os.path.abspath(newfilename), delimiter=",", usecols=(0,1), skiprows=1)).tolist())
                else:
                    temp = (np.loadtxt(os.path.abspath(newfilename), delimiter=",", usecols=(0,1), skiprows=1) )
                    rescale = (rescaledatatomin(temp[:,1], newrange = [0.8, 0.84]))
                    temp = [  [temp[i,0], rescale[i]] for i in range(len(temp)) ]
                    Hahn_data.append(temp[10:-5])
                
    Hahn_data = [item for sublist in Hahn_data for item in sublist]

    Hahn_data = np.asarray(Hahn_data)
    Hahn_data = Hahn_data[Hahn_data[:,0].argsort()] 

    if clean_duplicates:
        u, indices = np.unique(Hahn_data[:,0], return_index=True)
        Hahn_data = np.array([[Hahn_data[i, 0], Hahn_data[i, 1]] for i in indices])
        
    return(Hahn_data)
    
    
def EXPfromVector(datavector, time):
     
     idx = (np.abs(datavector[:,0]-time)).argmin()
     
     return datavector[idx]
     
     
def rescaledatatofullrange(datavector, newrange = [0.,1.]):

    newmean = np.mean(newrange)
    recenter = np.mean(datavector) - newmean

    datavector = datavector - recenter + newrange[0]

    rescale_factor_sup = (newrange[1]-newmean)/(np.amax(datavector)-np.mean(datavector))
    rescale_factor_inf = (newrange[1]-newmean)/(np.mean(datavector)-np.amin(datavector))

    for i in range(len(datavector)):
        if datavector[i] > newmean:
            datavector[i] = newmean + rescale_factor_sup*(datavector[i]-newmean)
        elif datavector[i] < newmean:
            datavector[i] = newmean + rescale_factor_inf*(datavector[i]-newmean)
    return datavector
    
    
def rescaledatatomin(datavector, newrange = [0.,1.]):

    newmean = np.mean(newrange)
    recenter = np.mean(datavector) - newmean

    datavector = datavector - recenter
    maxdatum = np.amax(datavector)
    mindatum = np.amin(datavector)

    for i in range(len(datavector)):
        if datavector[i] > newmean:
            datavector[i] = newmean + (newrange[1]-newmean)*(datavector[i]-newmean)/(maxdatum-newmean)
        elif datavector[i] < newmean:
            datavector[i] = newmean - (newrange[0]-newmean)*(datavector[i]-newmean)/(newmean-mindatum)
    
    return datavector