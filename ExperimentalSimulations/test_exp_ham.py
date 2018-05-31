import hamiltonian_exponentiation as h
import numpy as np
from scipy import linalg

# from qmd.7062222 -- (QML 0)

ham = np.array(
    [[ 0.00000000+0.j , 0.00000000+0.j , 0.25918219+0.j , 0.00000000+0.j],
 [ 0.00000000+0.j,  0.00000000+0.j , 0.00000000+0.j , 0.25918219+0.j],
 [ 0.25918219+0.j,  0.00000000+0.j , 0.00000000+0.j , 0.00000000+0.j],
 [ 0.00000000+0.j  ,0.25918219+0.j , 0.00000000+0.j , 0.00000000+0.j]] )

t =  2948472195.13
probe = np.array([-0.00087474+0.37269619j , 0.01355581+0.04259673j , 
                  0.36092475-0.84655081j,  0.01020239-0.10990682j] )

U = linalg.expm(-1j*ham*t)
probe_bra = probe.conj().T
up=np.dot(U, probe)
pup = np.dot(probe_bra, up)
exp_lin = np.abs(pup)**2

U = h.exp_ham(ham,t)
probe_bra = probe.conj().T
up=np.dot(U, probe)
pup = np.dot(probe_bra, up)
exp_ham = np.abs(pup)**2


print("Lin:", exp_lin)
print("Ham:", exp_ham)
