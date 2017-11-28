import qinfer as qi
import numpy as np
import scipy as sp

from Evo import *


def choose_probe(oplist,modelpars=None):
    #TODO: check and decide whether better calculated for each particle or once for the model
    """
    Chooses a linear combination of eigenstates to maximise observed dynamics
    for the model
    By default, modelparams are weighted equally if none is provided
    """
    if modelpars is None:
        modelpars = np.array([np.ones(len(oplist))])
    
    Htemp = getH(modelpars, oplist)
    
    probe = np.sum(np.linalg.eig(Htemp)[1], axis=0)
    
    probe /= np.linalg.norm(probe)
    return probe
    
    
def Hahn_probe(oplist, base_states=None):
    
    dim = int(np.log2(oplist.shape[-1]))
    
    if base_states is None:
        base_states = np.repeat(plus(), dim).reshape(dim, 2)
    
    for i in range(len(base_states)-1):
        probe = np.kron(base_states[i], base_states[i+1])
    
    return probe
    
    
    
def def_randomprobe(oplist,modelpars=None):
    #TODO: check and decide whether better calculated for each particle or once for the model
    """
    Picks a random probe in the space
    """
    a = 2*np.random.random((1,len(oplist[0])))-1
    b = 2*np.random.random((1,len(oplist[0])))-1
    probe = (a+1j*b)[0]
    probe /= np.linalg.norm(probe)
    return probe

    
    
    

def choose_randomprobe(probelist, modelpars=None):
    #TODO: check and decide whether better calculated for each particle or once for the model
    """
    Chooses a linear combination of eigenstates to maximise observed dynamics
    for the model
    By default, modelparams are weighted equally if none is provided
    """
    idx = np.random.randint(0,len(probelist))  
    probe = probelist[idx]
    return probe



def list_randomeig(oplist, modelpars=None):
    #TODO: check and decide whether better calculated for each particle or once for the model
    """
    Ouputs a list of a random eigenvector for each operator
    """
    idx = np.random.randint(len(oplist[0]))
    probes = list(map(lambda oplist: np.linalg.eig(oplist)[1][idx], oplist))
    return probes



def probes_randomcombo(probelist):
    #TODO: check and decide whether better calculated for each particle or once for the model
    """
    Chooses a random linear combination of probestates in probelist
    """
    wgts=np.random.uniform(0,1,len(probelist))
    newprobe = np.sum(wgts*probelist, axis=0)
    
    newprobe /= np.linalg.norm(newprobe)
    return newprobe
    
    
def probes_pondercombos(probelist, length):
    #TODO: check and decide whether better calculated for each particle or once for the model
    """
    Chooses a random linear combination of probestates in probelist
    """
    wgts=np.linspace(0.1,0.9,length)
    wgts=np.append(a, np.flipud(a[1:-1]))
    index = range((len(wgts)))
    wgts_list = np.asarray(list(map(lambda index: np.roll(wgts_list, index), index )))
    newprobe = list(map(lambda wgts_list: np.sum(wgts_list*probelist,axis=0), wgts_list))
    newprobe = list(map(lambda newprobe: newprobe/np.linalg.norm(newprobe), newprobe))
    return newprobe