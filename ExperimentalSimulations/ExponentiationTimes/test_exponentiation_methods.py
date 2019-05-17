import numpy as np
import qutip
import hamiltonian_exponentiation as h
import sys, os
sys.path.append(os.path.join("..","Libraries","QML_lib"))
import DataBase
# import ModelGeneration
# from ModelGeneration import random_model_name

import matplotlib.pyplot as plt
import time
import scipy
import random 

import argparse

### Parse arguments from bash
parser = argparse.ArgumentParser(
	description='Pass files to pickel QHL parameters.'
)

parser.add_argument(
  '-min_q', '--min_num_qubits', 
  help="Lowest number of qubits to time.",
  type=int,
  default=1
)

parser.add_argument(
  '-max_q', '--max_num_qubits', 
  help="Highest number of qubits to time.",
  type=int,
  default=5
)

parser.add_argument(
  '-tests', '--num_tests', 
  help="Number of iterations to average times over.",
  type=int,
  default=10
)

parser.add_argument(
  '-plot_dir', '--plot_directory', 
  help="Where to store plots. ",
  type=str,
  default=''
)






def generate_term(
    num_dimensions, 
):
    """
    For use only in random_model_name() function. 
    """
    paulis = DataBase.pauli_cores_with_identity
    import random
    t_str = ''
    running_str =''
    
    if num_dimensions == 1:
        return random.choice(paulis)
    else:
        for j in range(num_dimensions):
            t_str += 'T'
            running_str += random.choice(paulis)
            
            if j != num_dimensions -1:
                running_str += t_str

        return running_str
 

def random_model_name(num_dimensions=1, num_terms=1):
    """
    Return a valid (simple) model name of given number of dimensions and terms. 
    """
    import random
    paulis = DataBase.pauli_cores_with_identity
    p_str = ''
    t_str = ''
    
    for i in range(num_dimensions):
        p_str += 'P'
    for j in range(num_dimensions -1):
        t_str += 'T'
    
    
    summed_term = ''
    for j in range(1, num_terms+1):
        this_term = generate_term(num_dimensions)
        summed_term += this_term
        if(j!=num_terms):
            summed_term += p_str
    
    
    # Don't allow returning just identity in any dimension #TODO?
    while summed_term == (
        'i' or 'iTi' or 'iTiTTi' or 'iTiTTiTTTi' or
        'iTiTTiTTTiTTTTi' or 'iTiTTiTTTiTTTTiTTTTTi' or 
        'iTiTTiTTTiTTTTiTTTTTiTTTTTTi' or 
        'iTiTTiTTTiTTTTiTTTTTiTTTTTTiTTTTTTTi'
    ):
        summed_term = random_model_name(num_dimensions, num_terms) 
    
    return summed_term

arguments = parser.parse_args()
num_tests = arguments.num_tests
min_num_qubits = arguments.min_num_qubits
max_num_qubits = arguments.max_num_qubits
plot_dir = arguments.plot_directory
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)            

qubit_num = 1
qutip_time=0
custom_time = 0
max_num_qubits = max_num_qubits + 1 # index runs to N-1
times_by_method_and_q_num = {
    'qutip' : {},
    'custom' : {},
    'linalg' : {},
    'np_linalg' : {}
}

differences = {
	'custom_qutip' : {}, 
	'custom_linalg' : {}, 
	'qutip_linalg' : {}
}


for qubit_num in range(min_num_qubits, max_num_qubits):
    print("Qubit number:{}".format(qubit_num))
    times_by_method_and_q_num['qutip'][qubit_num] = []
    times_by_method_and_q_num['custom'][qubit_num] = []
    times_by_method_and_q_num['linalg'][qubit_num] = []

    differences['custom_qutip'][qubit_num] = []
    differences['custom_linalg'][qubit_num] = []
    differences['qutip_linalg'][qubit_num] = []

    if qubit_num > 8:
        num_tests = max(num_tests, 5)
    for i in range(num_tests):
        # random_model_name = ModelGeneration.random_model_name(
        new_random_model_name = random_model_name(
            num_terms=5, 
            num_dimensions=int(qubit_num)
        )
        op = DataBase.operator(str(new_random_model_name))
        hamiltonian =  op.matrix
        ham_time = random.uniform(9.9, 10)
        a=time.time()
        qexp=qutip.Qobj(-1j*hamiltonian*ham_time).expm().full()
        times_by_method_and_q_num['qutip'][qubit_num].append(time.time()-a)
        
        b=time.time()
#        hexp = h.exp_ham(hamiltonian, ham_time, precision=1e-14)
        hexp = h.exp_ham(hamiltonian, ham_time, precision=1e-1)
        times_by_method_and_q_num['custom'][qubit_num].append(time.time()-b)
        
        c=time.time()
        lexp = scipy.linalg.expm(-1j*hamiltonian*ham_time)
        times_by_method_and_q_num['linalg'][qubit_num].append(time.time()-c)
        ltime=time.time()-c
        
        # print("\n\n")
        # print("qubit_num:", qubit_num, "t=", ham_time)
        # print("Differences:  \n scipy-qexp", np.max(np.abs(lexp-qexp)))
        # print("\n custom-linalg", np.max(np.abs(hexp-lexp)))
        # print("\n custom-qexp", np.max(np.abs(hexp-qexp)))
        
        differences['custom_qutip'][qubit_num] = np.max(np.abs(qexp-hexp))
        differences['custom_linalg'][qubit_num] = np.max(np.abs(hexp-lexp))
        differences['qutip_linalg'][qubit_num] = np.max(np.abs(qexp-lexp))

        if np.allclose(qexp, hexp, atol=1e-14)==False:
            diff = np.max(np.abs(qexp-hexp))
            print("[", qubit_num, "qubits] max diff:", diff)
            print("Hermitian hamiltonian:", test_hermiticity(hamiltonian))
            if diff > 1: 
                print("!!!!!!")

avg_times = {
    'qutip' : {},
    'custom' : {},
    'linalg': {}
}
avg_diff = {
	'custom_qutip' : {}, 
	'custom_linalg' : {}, 
	'qutip_linalg' : {}
}

for qubit_num in range(min_num_qubits, max_num_qubits):
    q_avg = np.min(times_by_method_and_q_num['qutip'][qubit_num])
    c_avg = np.min(times_by_method_and_q_num['custom'][qubit_num])
    l_avg = np.min(times_by_method_and_q_num['linalg'][qubit_num])

    avg_times['qutip'][qubit_num] = q_avg
    avg_times['custom'][qubit_num] = c_avg
    avg_times['linalg'][qubit_num] = l_avg

    max_diff_custom_qt = np.max(differences['custom_qutip'][qubit_num])
    max_diff_custom_lin = np.max(differences['custom_linalg'][qubit_num])
    max_diff_lin_qt = np.max(differences['qutip_linalg'][qubit_num])
    avg_diff['custom_qutip'][qubit_num] = max_diff_custom_qt
    avg_diff['custom_linalg'][qubit_num] = max_diff_custom_lin
    avg_diff['qutip_linalg'][qubit_num] = max_diff_lin_qt

    
qubit_numbers = range(min_num_qubits, max_num_qubits)
ratio = [avg_times['qutip'][q]/avg_times['custom'][q] for q in range(min_num_qubits, max_num_qubits)]


plot_descriptor = "_{}_qubits__{}_tests".format(
	arguments.max_num_qubits, 
	num_tests
)

# Plot Times
plt.clf()
for k in avg_times:
    times = [avg_times[k][q] for q in qubit_numbers]
    print(k, "\t times:", times)
    plt.semilogy(
    	qubit_numbers, 
    	times, 
    	label=str(k), 
    	linestyle='--'
	)

plt.axhline(1, label='1 sec', color='black')
plt.axhline(2, label='2 sec', linestyle='--', color='black')

plt.plot(qubit_numbers, ratio, label='Ratio', color='r')
plt.legend()
plt.ylabel('Median Time to compute')
plt.xlabel('Num Qubits')
plt.title(
	str(
		'Exponentiation Times' + 
		plot_descriptor
	)

)
plt.savefig(
	str(
		plot_dir + 
		'ExponentiationTimeComparison' + 
		plot_descriptor + '.png'
	)
)

# Plot Differences
plt.clf()
for k in avg_diff:
    diffs = [avg_diff[k][q] for q in qubit_numbers]
    print(k, "\t Differences:", diffs)
    plt.scatter(
    	qubit_numbers, 
    	diffs, 
    	label=str(k), 
    	linestyle='--'
	)
ax = plt.gca()
ax.set_yscale('log')
plt.ylabel('Maximum difference')
plt.xlabel('Num Qubits')
plt.legend()
plt.title(
	str(
		'Max difference between Hamiltonians between methods' + 
		plot_descriptor
	)

)

plt.savefig(
	str(
		plot_dir + 
		'ExponentiationDifference' + 
		plot_descriptor + '.png'
	)
)
