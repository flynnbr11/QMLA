import sys
import os
from scipy import linalg
import numpy as np
import time
import random
import tensorflow as tf
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

sys.path.append("/home/bf16951/QMD")
import qmla

def random_hamiltonian(number_qubits):
    """
    Generate a random Hamiltonian - will be square with length/width= 2**number_qubits.
    Hamiltonian will be Hermitian so diagonally symmetrical elements are complex conjugate.
    Hamiltonian also formed by Pauli matrices.
    """

    sigmax = np.array([[0+0j, 1+0j], [1+0j, 0+0j]])
    sigmay = np.array([[0+0j, 0-1j], [0+1j, 0+0j]])
    sigmaz = np.array([[1+0j, 0+0j], [0+0j, -1+0j]])

    oplist =  [sigmax, sigmay, sigmaz]
    size = number_qubits
    select=np.round((len(oplist)-1)*np.random.rand(size))
    newoplist = [oplist[int(i)] for i in select]
    params=np.random.rand(size)

    if len(params)==1:
        output = params[0]*newoplist[0]
    else:
        for i in range(len(params)-1):
            if i==0:
                output = np.kron(params[i]*newoplist[i], params[i+1]*newoplist[i+1])
            else:
                output = np.kron(output, params[i+1]*newoplist[i+1])
    output = np.reshape(output, [2**size,2**size])
    return output


def linalg_expm(
    ham, 
    t, 
    probe, 
):
    U = linalg.expm(-1j*ham*t)
    Up = np.dot(U, probe)
    return np.abs(np.dot(probe.conj().T, Up))**2    
        
def tensorflow_expm(
    ham, 
    t, 
    probe
):
    for t in time_list:
        with tf.Session() as sess: 
            tf_ex = tf.linalg.expm(
                tf.convert_to_tensor(-1j*ham*t, dtype='complex128')
            ).eval()
            
    Up = np.dot(tf_ex, probe)
    return np.abs(np.dot(probe.conj().T, Up))**2    

def sparse_expm_csc(
    ham, 
    t,
    probe,
):
    sparse_ham = sparse.csc_matrix(-1j*ham*t)
    U = sparse.linalg.expm(sparse_ham).todense()        
    Up = np.array(np.dot(U, probe))[0]
    return np.abs(np.dot(probe.conj().T, Up))**2    

def sparse_expm_multiply_csc(
    ham, t, probe
):
    sparse_ham = sparse.csc_matrix(-1j*ham*t)
    Up = sparse.linalg.expm_multiply( 
        sparse_ham, 
        probe
    )
    return np.abs(np.dot(probe.conj().T, Up))**2    

def sparse_expm_csr(
    ham, 
    t,
    probe,
):
    sparse_ham = sparse.csr_matrix(-1j*ham*t)
    U = sparse.linalg.expm(sparse_ham).todense()        
    Up = np.array(np.dot(U, probe))[0]
    return np.abs(np.dot(probe.conj().T, Up))**2    


def sparse_expm_multiply_csr(
    ham, t, probe
):
    sparse_ham = sparse.csr_matrix(-1j*ham*t)
    b = sparse.csr_matrix(probe)
    Up = np.array(sparse.linalg.expm_multiply( 
        sparse_ham, 
        probe
    ))
    return np.abs(np.dot(probe.conj().T, Up))**2

# Setup 
methods = {
    'linalg_expm' : linalg_expm, 
    # 'tensorflow_expm' : tensorflow_expm
    'csc' : sparse_expm_csc, 
    'csc_multiply' : sparse_expm_multiply_csc,
    'csr' : sparse_expm_csr, 
    'csr_multiply' : sparse_expm_multiply_csr,
}


timings = pd.DataFrame()
num_samples = 50
max_num_qubits = 8
probes = qmla.shared_functionality.probe_set_generation.separable_probe_dict(
    num_probes=1, 
    max_num_qubits = max_num_qubits
)
sim_times = [random.uniform(0, 1000) for _ in range(int(num_samples))]
evolutions = {}

# Time each method
for n in range(1, 1+max_num_qubits):
    H = random_hamiltonian(number_qubits=n) # TODO make a random Hamiltonian model

    for m in methods:
        exp_method = methods[m]
        begin = time.time()
        evolutions[m] = [
            exp_method(
                ham = H, 
                t = t, 
                probe = probes[0, n]
            ) for t in sim_times
        ]
            
        time_taken = time.time()-begin
        avg_time = time_taken / num_samples
        
        method_time = pd.Series({
            'method' : m,
            'num_qubits' : n, 
            'time_per_exp' : avg_time
        })
        timings = timings.append(
            method_time,
            ignore_index = True
        )

# Plot timings
fig, ax = plt.subplots(figsize=(15, 10))
sns.lineplot(
    y = 'time_per_exp', 
    x = 'num_qubits',
    hue='method',
    data = timings,
    ax = ax, 
)

ax.set_xlabel('# Qubits')
ax.set_ylabel('Time (s)')
ax.set_yscale('log')

test_description = "{}_qubits_{}_samples".format(max_num_qubits, num_samples)
fig.savefig("timings_{}.png".format(test_description))

# Ensure results consistency via different methods
tested_methods = list(evolutions.keys())
method_pairs = list(itertools.combinations(tested_methods, 2))
close_checks = pd.DataFrame(columns=['m1', 'm2', 'close'])

for m1, m2 in method_pairs:
    close = np.allclose( evolutions[m1], evolutions[m2] )
    
    this_pair = pd.Series({
        'm1' : m1, 
        'm2' : m2, 
        'close' : close
    })
    
    close_checks = close_checks.append(
        this_pair, ignore_index=True
    )
close_checks.to_csv(
    "results_similarity_{}.csv".format(test_description)
)