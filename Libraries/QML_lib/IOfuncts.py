import qinfer as qi
import numpy as np
import scipy as sp

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