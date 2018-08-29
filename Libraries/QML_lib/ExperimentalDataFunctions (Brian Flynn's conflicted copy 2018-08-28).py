import sys
import os
import numpy as np
from bisect import bisect_left
import random



def experimentalMeasurementDict(directory, max_time=1e4):
    experimental_times_exp_vals = importExperimentalData(
        directory=directory, rescale=False
    )
    
    # get rid of times too high

    times=list(experimental_times_exp_vals[:,0])
    exp=list(experimental_times_exp_vals[:,1])

    useful_times=[]
    useful_exps=[]
    
    for i in range(len(times)):
        if times[i] < max_time:
            useful_times.append(times[i])
            useful_exps.append(exp[i])
        

    useful_exps = rescaleData(useful_exps)

    experimental_data = {}
    for i in range(len(useful_times)):
        experimental_data[useful_times[i]] = useful_exps[i]    

    return experimental_data


def importExperimentalData(directory, max_time=1e4, rescale = False, clean_duplicates = True):
    
    exp_data = []
    
    for root,dirs,files in os.walk(directory):
        for filename in files:  
            if filename.endswith(".csv"):
                newfilename = os.path.join(directory, filename)
                new_exp_data = np.loadtxt(os.path.abspath(newfilename), 
                    delimiter=",", usecols=(0,2), skiprows=1
                )            
                if rescale: 
                    new_exp_data[:,1] =  rescaleData(new_exp_data[:,1])
                
                exp_data.append(new_exp_data.tolist())
                
    exp_data = [item for sublist in exp_data for item in sublist]

    exp_data = np.asarray(exp_data)
    exp_data = exp_data[exp_data[:,0].argsort()] 

    if clean_duplicates:
        u, indices = np.unique(exp_data[:,0], return_index=True)
        exp_data = np.array([[exp_data[i, 0], exp_data[i, 1]] for i in indices])
        
    return(exp_data)

def rescaleData(datavector, newrange = [0.,1.]):

    newmean = np.mean(newrange)
    recenter = np.mean(datavector) - newmean

    datavector = datavector - recenter + newrange[0]

    rescale_factor_sup = ( (newrange[1]-newmean)/
        (np.amax(datavector)-np.mean(datavector))
    )
    rescale_factor_inf = ( (newrange[1]-newmean)/
        (np.mean(datavector)-np.amin(datavector))
    )

    for i in range(len(datavector)):
        if datavector[i] > newmean:
            datavector[i] = newmean + rescale_factor_sup*(datavector[i]-newmean)
        elif datavector[i] < newmean:
            datavector[i] = newmean + rescale_factor_inf*(datavector[i]-newmean)
    return datavector


def nearestAvailableExpTime(times, t):
    """
    - times: Sorted time list
    - experimental_data: dict where key is time and value is expectation value
    - t: time to get nearest available experimental data point for. 
    
    If two times are equally close, return the smallest.
    """
    if t > max(times):
        nearest = random.choice(times)

    else:
        pos = bisect_left(times, t)
        if pos == 0:
            return times[0]
        if pos == len(times):
            return times[-1]
        before = times[pos - 1]
        after = times[pos]
        if after - t < t - before:
            nearest =  after
        else:
            nearest = before
    
    return nearest



def nearestAvailableExpVal(times, experimental_data, t):
    """
    - times: Sorted time list
    - experimental_data: dict where key is time and value is expectation value
    - t: time to get nearest available experimental data point for. 
    
    If two times are equally close, return the smallest.
    """
    if t > max(times):
        nearest = random.choice(times)

    else:
        pos = bisect_left(times, t)
        if pos == 0:
            return times[0]
        if pos == len(times):
            return times[-1]
        before = times[pos - 1]
        after = times[pos]
        if after - t < t - before:
            nearest =  after
        else:
            nearest = before
    
    return experimental_data[nearest]

def experimental_NVcentre_ising_probes(max_num_qubits=2, num_probes=40):
    """
    Returns a dict of separable probes where the first qubit always acts on 
    a plus state. 
    """
    plus_state = np.array([1, 1])/np.sqrt(2)
    noise_level = 0.03 # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    random_noise = noise_level * random_probe(1)    
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus/norm_factor
    
    seperable_probes = {}
    for i in range(num_probes):
#        seperable_probes[i,0] = plus_state
        seperable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = (np.tensordot(seperable_probes[i,j-1],
                    random_probe(1), axes=0).flatten(order='c')
                )
            while (np.isclose(1.0, np.linalg.norm(seperable_probes[i,j]), 
                atol=1e-14) is  False
            ):
                print("non-unit norm: ", np.linalg.norm(seperable_probes[i,j]))
                # keep replacing until a unit-norm 
                seperable_probes[i,j] = (
                    np.tensordot(seperable_probes[i,j-1], random_probe(1),
                    axes=0).flatten(order='c')
                )
    return seperable_probes
    
    
    

def experimental_NVcentre_ising_probes_plusplus(max_num_qubits=2, num_probes=40):
    """
    Returns a dict of separable probes where the first qubit always acts on 
    a plus state. 
    """
    plus_state = np.array([1, 1])/np.sqrt(2)
    noise_level = 0.03 # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    random_noise = noise_level * random_probe(1)    
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus/norm_factor
    
    seperable_probes = {}
    for i in range(num_probes):
#        seperable_probes[i,0] = plus_state
        seperable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = (np.tensordot(seperable_probes[i,j-1],
                    noisy_plus, axes=0).flatten(order='c')
                )
            while (np.isclose(1.0, np.linalg.norm(seperable_probes[i,j]), 
                atol=1e-14) is  False
            ):
                print("non-unit norm: ", np.linalg.norm(seperable_probes[i,j]))
                # keep replacing until a unit-norm 
                seperable_probes[i,j] = (
                    np.tensordot(seperable_probes[i,j-1], random_probe(1),
                    axes=0).flatten(order='c')
                )
    return seperable_probes
    
    


def random_probe(num_qubits):
    dim = 2**num_qubits
    real = []
    imaginary = []
    complex_vectors = []
    for i in range(dim):
        real.append(np.random.uniform(low=-1, high=1))
        imaginary.append(np.random.uniform(low=-1, high=1))
        complex_vectors.append(real[i] + 1j*imaginary[i])

    a=np.array(complex_vectors)
    norm_factor = np.linalg.norm(a)
    probe = complex_vectors/norm_factor
    if np.isclose(1.0, np.linalg.norm(probe), atol=1e-14) is False:
        print("Probe not normalised. Norm factor=", np.linalg.norm(probe)-1)
        return random_probe(num_qubits)

    return probe    
