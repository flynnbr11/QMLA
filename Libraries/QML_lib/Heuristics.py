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
        increase_time=False, 
        maxiters=10,
        other_fields=None,
        **kwargs
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
        self._increase_time = increase_time
        print(
            "[Heuristics - multiPGH]",
        )

    def __call__(
        self,
        epoch_id = 0,
        num_params = 1,
        test_param = None, 
    ):

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
            # print("[multipgh] oplist is None")

            sigma = self._updater.model.distance(x, xp)
            # print("sigma = ", sigma)
            if self._increase_time == True:
                # Increase time 
                # ie get 1/sigma and add another time factor on top
                # to reach higher times
                orig_time = self._t_func(
                    (1 / sigma**self._pgh_exponent)
                )

                new_time = self._t_func(
                    (1 / sigma**self._pgh_exponent)
                    + ((1/sigma) * epoch_id * num_params)/10
                )
                # print(
                #     "[multipgh]", 
                #     "\norig time:", orig_time, 
                #     "\nnew_time:", new_time
                # )
                eps[self._t] = new_time
            else:
                new_time = self._t_func(
                    1 / sigma**self._pgh_exponent
                )
                # print(
                #     "[multipgh]", 
                #     "\nnew_time:", new_time
                # )
                eps[self._t] = new_time 


        else:
            deltaH = getH(x, self._oplist)-getH(xp, self._oplist)
            if self._norm=='Frobenius':
                print("[multipgh] Froebenius")
                print (frameinfo.filename, frameinfo.lineno)
                eps[self._t] = 1/np.linalg.norm(deltaH)   #Frobenius norm
            elif self._norm=='MinSingVal':
                print("[multipgh] MinSingVal")
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
        print(
            "[multipgh] returning [shape ", 
            np.shape(eps), 
            "] eps: ", 
            repr(eps)
        )

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

class time_from_list(qi.Heuristic):
    
    def __init__(
        self, 
        updater, 
        oplist=None, 
        pgh_exponent=1,
        increase_time=False, 
        norm='Frobenius', 
        inv_field='x_', 
        t_field='t',
        inv_func=identity,
        t_func=identity,
        maxiters=10,
        other_fields=None,
        time_list=None,
        **kwargs
     ):
        super(time_from_list, self).__init__(updater)
        self._oplist = oplist
        self._norm = norm
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._maxiters = maxiters
        self._other_fields = other_fields if other_fields is not None else {}
        self._pgh_exponent = pgh_exponent
        self._increase_time = increase_time
        # self._time_list = kwargs['time_list'] 
        self._time_list = time_list 
        self._len_time_list = len(self._time_list)
        print(
            "[Heuristics - time_from_list]",
            "\n kwargs:", **kwargs 
        )


    def __call__(
        self,
        epoch_id = 0,
        num_params = 1,
        test_param = None, 
    ):

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
        
        time_id = epoch_id % self._len_time_list
        new_time = self._time_list[time_id] 
        eps[self._t] = new_time 

        # print("[Hueristic - time list] time idx {}  chosen {}:".format(
        #         time_id, 
        #         eps[self._t]
        #     )
        # )
        return eps
