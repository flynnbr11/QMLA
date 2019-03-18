import qinfer as qi
import numpy as np
import scipy as sp

from Evo import *
from inspect import currentframe, getframeinfo

frameinfo = getframeinfo(currentframe())

def singvalnorm(matrix):
    return np.real(max(np.sqrt(np.linalg.eig(np.dot(
        matrix.conj().T,matrix))[0])))

def minsingvalnorm(matrix):
    return np.real(min(np.sqrt(np.absolute((np.linalg.eig(np.dot(
        matrix.conj().T,matrix))[0])))))


def identity(arg): return arg

class multiPGH(qi.Heuristic):
    
    def __init__(
        self, 
        updater, 
        oplist=None, 
        norm='Frobenius', 
        inv_field='x_', 
        t_field='t',
        inv_func=identity,
        t_func=identity,
        pgh_exponent=1,
        maxiters=10,
        other_fields=None
     ):
        super(multiPGH, self).__init__(updater)
        self._oplist = oplist
        self._norm = norm
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._maxiters = maxiters
        self._other_fields = other_fields if other_fields is not None else {}
        self._pgh_exponent = pgh_exponent
        print("[multipgh] PGH exponent ", self._pgh_exponent)

    def __call__(self):
        idx_iter = 0
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError(
                "PGH did not find distinct particles in \
                {} iterations.".format(self._maxiters)
            )
            
        #print('Selected particles: #1 ' + repr(x) + ' #2 ' + repr(xp))
            
        eps = np.empty(
            (1,),
            dtype=self._updater.model.expparams_dtype
        )
        # print (frameinfo.filename, frameinfo.lineno)
        
        idx_iter = 0 # modified in order to cycle through particle parameters with different names
        for field_i in self._x_:
            eps[field_i] = self._inv_func(x)[0][idx_iter]
            idx_iter += 1
        if self._oplist is None:   #Standard QInfer geom distance
            sigma = self._updater.model.distance(x, xp)
            # print("sigma = ", sigma)
            eps[self._t]  = self._t_func(
                1 / sigma**self._pgh_exponent
            )

        else:
            deltaH = getH(x, self._oplist)-getH(xp, self._oplist)
            if self._norm=='Frobenius':
                print (frameinfo.filename, frameinfo.lineno)
                eps[self._t] = 1/np.linalg.norm(deltaH)   #Frobenius norm
            elif self._norm=='MinSingVal':
                print (frameinfo.filename, frameinfo.lineno)
                eps[self._t] = 1/minsingvalnorm(deltaH)   #Min SingVal norm
            elif self._norm=='SingVal':
                print (frameinfo.filename, frameinfo.lineno)
                eps[self._t] = 1/singvalnorm(deltaH)   #Max SingVal
            else:
                print (frameinfo.filename, frameinfo.lineno)
                eps[self._t] = 1/np.linalg.norm(deltaH)
                raise RuntimeError("Unknown Norm: using Frobenius norm instead")
        for field, value in self._other_fields.items():
            eps[field] = value**self._pgh_exponent
        # print(
        #     "[multipgh] returning [shape ", 
        #     np.shape(eps), 
        #     "] eps: ", 
        #     repr(eps)
        # )
        return eps
        
        
class tHeurist(qi.Heuristic):
    
    def identity(arg): return arg
    
    def __init__(self, updater, t_field='t',
                 t_func=identity,
                 maxiters=10
                 ):
        super(tHeurist, self).__init__(updater)
        self._t = t_field
        self._t_func = t_func
        self._maxiters = maxiters
        
    def __call__(self):
        idx_iter = 0
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError("PGH did not find distinct particles in {} \
                iterations.".format(self._maxiters)
            )
            
        eps = np.empty((1,), dtype=self._updater.model.expparams_dtype)
        eps[self._t]  = self._t_func(1 / self._updater.model.distance(x, xp))
        
        return eps
