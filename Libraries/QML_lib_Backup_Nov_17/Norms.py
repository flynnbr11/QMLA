import qinfer as qi
import numpy as np
import scipy as sp

def singvalnorm(matrix):
    return np.real(max(np.sqrt(np.linalg.eig(np.dot(
        matrix.conj().T,matrix))[0])))

def minsingvalnorm(matrix):
    return np.real(min(np.sqrt(np.absolute((np.linalg.eig(np.dot(
        matrix.conj().T,matrix))[0])))))