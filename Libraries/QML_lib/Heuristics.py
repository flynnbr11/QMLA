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
        # print(
        #     "[Hueirstic] - multipgh\n", 
        #     "kwargs:", kwargs
        # )

    def __call__(
        self,
        epoch_id = 0,
        num_params = 1,
        test_param = None,
        **kwargs 
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
        
    def __call__(self, **kwargs):
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

        # print(
        #     "[Heuristics - time_from_list]",
        #     # "num experiments:", self._num_experiments,
        #     "\n kwargs:", kwargs 
        # )

        try:
            self._num_experiments = kwargs.get('num_experiments')
            print("self.num_experiments:", self._num_experiments)
        except:
            print("Can't find num_experiments in kwargs")

    def __call__(
        self,
        epoch_id = 0,
        num_params = 1,
        test_param = None, 
        **kwargs
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


class one_over_sigma_then_linspace(qi.Heuristic):
    
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
        # num_experiments=200,
        **kwargs
     ):
        super(one_over_sigma_then_linspace, self).__init__(updater)
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
        self._num_experiments = kwargs.get('num_experiments', 200)

        # self._time_list = kwargs['time_list'] 
        self._time_list = time_list 
        self._len_time_list = len(self._time_list)
        self.num_epochs_for_first_phase = self._num_experiments/2

        

        # print(
        #     "[Heuristics - 1/sigma then linspace]",
        #     "num epochs for first phase:", self.num_epochs_for_first_phase,
        #     "\n kwargs:", kwargs 
        # )
        # try:
        #     self._num_experiments = kwargs.get('num_experiments')
        #     print("self.num_experiments:", self._num_experiments)
        # except:
        #     print("Can't find num_experiments in kwargs")


    def __call__(
        self,
        epoch_id = 0,
        num_params = 1,
        test_param = None, 
        **kwargs 
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
        
        if epoch_id < self.num_epochs_for_first_phase : 
            sigma = self._updater.model.distance(x, xp)
            new_time = self._t_func(
                1 / sigma**self._pgh_exponent
            )
        else:
            time_id = epoch_id % self._len_time_list
            new_time = self._time_list[time_id] 
        # print(
        #     "[Hueristic] 1/sigma then linspace", 
        #     "\t time:", new_time
        # )
        eps[self._t] = new_time 
        return eps

class inverse_min_eigvalue(qi.Heuristic):
    
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
        # num_experiments=200,
        **kwargs
     ):
        super(inverse_min_eigvalue, self).__init__(updater)
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
        self._num_experiments = kwargs.get('num_experiments', 200)

        # self._time_list = kwargs['time_list'] 
        self._time_list = time_list 
        self._len_time_list = len(self._time_list)
        self.num_epochs_for_first_phase = self._num_experiments/2

    def __call__(
        self,
        epoch_id = 0,
        num_params = 1,
        test_param = None, 
        **kwargs
    ):
        print(
            "[Heuristic - inverse_min_eigvalue]", 
            "kwargs:", kwargs
        )
        current_params = kwargs['current_params']
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
        
        if epoch_id < self.num_epochs_for_first_phase : 
            sigma = self._updater.model.distance(x, xp)
            new_time = self._t_func(
                1 / sigma**self._pgh_exponent
            )
        else:
            new_time = new_time_based_on_eigvals(
                params = current_params, 
                raw_ops = self._oplist
            )
        #     time_id = epoch_id % self._len_time_list
        #     new_time = self._time_list[time_id] 
        # print(
        #     "[Hueristic] 1/sigma then linspace", 
        #     "\t time:", new_time
        # )

        eps[self._t] = new_time 
        return eps



def new_time_based_on_eigvals(
    params, 
    raw_ops, 
    time_scale=1
):

    # print(
    #     "[Heuristic - time from eigvals]", 
    #     "params : {} \n raw ops: {}".format(params, raw_ops)
    # )
    param_ops = [
        (params[i] * raw_ops[i]) for i in range(len(params))
    ]
    max_eigvals = []
    for i in range(len(params)):
        param_eigvals = sp.linalg.eigh(param_ops[i])[0]

        max_eigval_this_op = max(np.abs(param_eigvals))
        max_eigvals.append(max_eigval_this_op)
    min_eigval = min(max_eigvals)
    new_time = time_scale*1/min_eigval
    # print("[Heuristic - new time:", new_time)
    return new_time
