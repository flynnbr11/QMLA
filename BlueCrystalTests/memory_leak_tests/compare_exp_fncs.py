import hamiltonian_exponentiation as h
from scipy import linalg
import random
from psutil import virtual_memory

num_qubits = 9

ham = h.random_hamiltonian(num_qubits)
t=random.random()

sp1=virtual_memory().used
sp=h.exp_ham(ham, t, enable_sparse_functionality=True, sparse_min_qubit_number=1)
del sp
sp2=virtual_memory().used
sp_diff = sp2 - sp1


ns1 = virtual_memory().used
ns = h.exp_ham(ham, t, enable_sparse_functionality=False, sparse_min_qubit_number=1)
del ns
ns2=virtual_memory().used
ns_diff = ns2-ns1

lin1= virtual_memory().used
lin = linalg.expm(-1j*ham*t)
del lin
lin2= virtual_memory().used
lin_diff = lin2-lin1

print("Differences : ")
print("Sparse : ", sp_diff)
print("Nonsparse : ", ns_diff)
print("Linalg : ", lin_diff)

