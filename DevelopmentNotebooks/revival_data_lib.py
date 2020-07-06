"""
useful functions for the ipynb
"""

import numpy as np
import scipy as sp
import sys, os
import qmla

def plus():
    return np.array([1, 1])/np.sqrt(2)

def minus():
    return np.array([1, -1])/np.sqrt(2)
    
def zero():
    return np.array([1, 0])
    
def one():
    return np.array([0, 1])
    
def plusI():
    return np.array([1, 1j])/np.sqrt(2)
    
def minusI():
    return np.array([1, -1j])/np.sqrt(2)

def sigmaz():
    return np.array([[1.0+0.j, 0.+0.j], [0.+0.j, -1.+0.j]])

def sigmax():
    return np.array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])

def sigmay():
    return np.array([[0.+0.j, 0.-1.j], [0.+1.j, 0+0j]])
    
def I2d():
    return np.eye(2)
    
    
def Hahn_evo(myham, t):
    u = sp.linalg.expm(-1.j*myham*t)
    hahn_gate = qmla.shared_functionality.hahn_y_gates.precomputed_hahn_y_inversion_gates[2]
    full_evo = np.dot(u, np.dot(hahn_gate, u))
    
    return full_evo

def likelihood_fromevo(myham, t, probe):
    evostate = np.dot(Hahn_evo(myham, t), probe)
    overlap = np.dot(evostate, probe)
    
    return np.abs(overlap**2)
    
    
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
    
def ImportAllFolder_Hahnsignal(directory, clean_duplicates = True):
    
    Hahn_data = []
    for root,dirs,files in os.walk(directory):
        for filename in files:  
            if filename.endswith(".csv") and filename.startswith("ana"):
                newfilename = os.path.join(directory, filename)
                print(os.path.abspath(newfilename))
                #usecols only selects one normalisation
                if filename.endswith("s.csv"):
                    Hahn_data.append((np.loadtxt(os.path.abspath(newfilename), delimiter=",", usecols=(0,1), skiprows=1)).tolist())
                elif filename.endswith("00.csv"):
                    temp = (np.loadtxt(os.path.abspath(newfilename), delimiter=",", usecols=(0,1), skiprows=1) )
                    rescale = (rescaledatatomin(temp[:,1], newrange = [min(temp[:,1]), 1.0]))
                    temp = [  [temp[i,0], rescale[i]] for i in range(len(temp)) ]
                    Hahn_data.append(temp[11:-5])
                else:
                    temp = (np.loadtxt(os.path.abspath(newfilename), delimiter=",", usecols=(0,1), skiprows=1) )
                    rescale = (rescaledatatomin(temp[:,1], newrange = [0.8, 0.84]))
                    temp = [  [temp[i,0], rescale[i]] for i in range(len(temp)) ]
                    Hahn_data.append(temp[10:-1])
                
    Hahn_data = [item for sublist in Hahn_data for item in sublist]

    Hahn_data = np.asarray(Hahn_data)
    Hahn_data[:,1] = (rescaledatatomin(Hahn_data[:,1], newrange = [0.5, 1]))
    Hahn_data = Hahn_data[Hahn_data[:,0].argsort()]

    if clean_duplicates:
        u, indices = np.unique(Hahn_data[:,0], return_index=True)
        clean_Hahn_data = np.array([[Hahn_data[i, 0], Hahn_data[i, 1]] for i in indices])
        
    return(Hahn_data)
    
def obtaindata(directory):

    sigdata = ImportAllFolder_Hahnsignal(directory, clean_duplicates = True)
    
    myrange = range(0, min(sigdata.shape[0],500)) 
    # prepydata = rescaledatatomin(sigdata[:,1], newrange = [0.5,1])

    xdata = sigdata[myrange,0]/1000 # converted to us
    ydata = sigdata[myrange,1]
    
    signal_offset = 15.
    
    return xdata, ydata
    
def retrieve_Hams_list(terms_params_list, n_qubits, iterable_term):
    
    try:
        param = terms_params_list['pauliSet_1_I_d{}'.format(n_qubits)][0]
        mtx = np.kron(np.eye(n_qubits), np.eye(n_qubits))
        hamiltonian = param * mtx
        del terms_params_list['pauliSet_1_I_d{}'.format(n_qubits)]
    except:
        hamiltonian = np.zeros([2**n_qubits,2**n_qubits])
        
    ham_array = np.array([hamiltonian for iter in terms_params_list[iterable_term]])
    
    for k in terms_params_list:
        # print(k)
        mtx = qmla.construct_models.compute(k)
        
        if k == iterable_term:
            new_terms = np.array([ terms_params_list[k][i]*mtx for i in range(len(ham_array)) ])
            # print(new_terms)
            ham_array = ham_array+new_terms
        
        else:
            param = terms_params_list[k][0]
            # print(param, ham_array)
            ham_array = ham_array+param*mtx
            
    return ham_array