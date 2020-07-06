import scipy as sp
import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import time

def random_mtx(num_qubits=2): 
    mtx_size = 2**num_qubits
    return np.random.rand(mtx_size, mtx_size)

max_num_qubits = 10
num_trials = 25

timings = {
    'scipy' : {}, 
    'tensorflow' : {}
}
linestyles = {'scipy': '--', 'tensorflow': '-'}

for nq in range(1, max_num_qubits + 1):
    timings['scipy'][nq] = []
    timings['tensorflow'][nq] = []
    for i in range(num_trials):
        ham = random_mtx(nq)

        # scipy
        a = time.time()
        assign = sp.linalg.expm(ham)
        timings['scipy'][nq].append(time.time() - a)
        
        # tensorflow
        b = time.time()
        with tf.Session() as sess: 
            tf_ex = tf.linalg.expm(
                tf.convert_to_tensor(ham, dtype='complex128')
            ).eval()
        timings['tensorflow'][nq].append(time.time() - b)
plt.clf()

for method in timings: 
    avg_times = []
    num_qubit_list = list(timings[method].keys())
    for num_qubits in num_qubit_list:
        avg_times.append(np.median(timings[method][num_qubits]))
    plt.semilogy(
        num_qubit_list, 
        avg_times,
        label = str(method),
        ls = linestyles[method]
    )
plt.legend()

plt.savefig('tensorflow_vs_scipy_expm.png')
    