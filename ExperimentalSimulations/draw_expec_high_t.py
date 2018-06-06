import hamiltonian_exponentiation as h
import numpy as np
from scipy import linalg
import sys, os
sys.path.append(os.path.join("..", "Libraries","QML_lib"))
from RedisSettings import *
import Evo as evo


ham = np.array(
 [[ 0.00+0.j , 0.00+0.j , 0.25+0.j , 0.00+0.j],
 [ 0.00+0.j , 0.00+0.j , 0.00+0.j , 0.25+0.j],
 [ 0.25+0.j , 0.00+0.j , 0.00+0.j , 0.00+0.j],
 [ 0.00+0.j , 0.25+0.j,  0.00+0.j  ,0.00+0.j]] 
)
probe = np.array( [0.25511523-0.4123109j,   0.33407614+0.09911499j  ,0.62111568-0.19628137j,
  0.25332773+0.3936995j])

times=[]
expecs=[]
for t in np.linspace(10,1e16,400):
    times.append(t)
    exp = evo.expectation_value(ham,t,probe)
    expecs.append(exp)



import matplotlib.pyplot as plt

plt.plot(times, expecs)    
plt.show()
