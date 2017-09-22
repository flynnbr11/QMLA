import qinfer as qi
import numpy as np
import scipy as sp
import sys, os

sys.path.append(os.path.join(".."))

from Norms import *
from IOfuncts import *
from EvalLoss import *

import ProbeStates as pros
import multiPGH as mpgh
import HahnSimQMD as gsi
import Evo as evo
import Distrib as distr
import HahnTheoModels as HTM


def param_sweep_anaqle(param, idparam, model, true_params, scramble, sigmas):
    
    n_trials = 25
    
    n_particles = 100
    n_experiments = 200
    
    track_loss = np.empty([n_trials, n_experiments])
    
    resample_thresh = 0.53
    mya = 0.84
    pgh = 1.26
    
    if idparam == 0:
        resample_thresh = param
    elif idparam == 1:
        mya = param
    else:
        pgh = param
    
    for trial in range(n_trials):

        prior = distr.MultiVariateNormalDistributionNocov(len(true_params[0]), mean = true_params[0]+scramble, sigmas = sigmas)
        updater = qi.SMCUpdater(model, n_particles, prior, resample_thresh=resample_thresh, resampler = qi.LiuWestResampler(a=mya), debug_resampling=False)
        
        inv_field = [item[0] for item in model.expparams_dtype[1:] ]
        expparams = np.empty((1, ), dtype=model.expparams_dtype)
        
        heuristic = mpgh.multiPGH(updater, inv_field=inv_field)

        for idx_experiment in range(n_experiments):

            experiment = heuristic()
            experiment[0][0] = pgh*experiment[0][0]

            datum = model.simulate_experiment(true_params, experiment)
            updater.update(datum, experiment)
        #     heuristic = mpgh.multiPGH(updater, oplist, inv_field=inv_field)
            new_loss = eval_loss(model, updater.est_mean(), true_params)
            track_loss[trial, idx_experiment] = new_loss[0]

    print("Done parameter " + str(param))
    
    return(track_loss)
    
    
    
    
    
def multiple_anaqle(trial, model, true_params, scramble, sigmas, n_particles=100, n_experiments=200, resample_thresh = 0.53, mya = 0.84, pgh = 1.27):
    
    track_loss = np.empty([n_experiments])
    track_eval = np.empty([n_experiments,len(true_params[0])])
    track_cov = np.empty([n_experiments])
    track_stdev = np.empty([n_experiments,len(true_params[0])])
    track_time = np.empty([n_experiments])

    inv_field = [item[0] for item in model.expparams_dtype[1:] ]
    expparams = np.empty((1, ), dtype=model.expparams_dtype)
    
    resample_thresh = 0.53
    mya = 0.84
    pgh = 1.27
    

    prior = distr.MultiVariateNormalDistributionNocov(len(true_params[0]), mean = true_params[0]+scramble, sigmas = sigmas)
    updater = qi.SMCUpdater(model, n_particles, prior, resample_thresh=resample_thresh, resampler = qi.LiuWestResampler(a=mya), debug_resampling=False)
    heuristic = mpgh.multiPGH(updater, inv_field=inv_field)

    for idx_experiment in range(n_experiments):

        experiment = heuristic()
        experiment[0][0] = pgh*experiment[0][0]
        track_time[idx_experiment] = experiment[0][0]

        datum = model.simulate_experiment(true_params, experiment)
        updater.update(datum, experiment)
    #     heuristic = mpgh.multiPGH(updater, oplist, inv_field=inv_field)
        
        track_cov[idx_experiment] = np.linalg.norm(updater.est_covariance_mtx())
        track_stdev[idx_experiment] = np.diag(updater.est_covariance_mtx())

        new_eval = updater.est_mean()
        track_eval[idx_experiment] = new_eval
        
        new_loss = eval_loss(model, new_eval, true_params)
        track_loss[idx_experiment] = new_loss[0]
    
    if trial%20 == 0:
        print("Done iteration " + str(trial))
    
    return([track_loss, track_eval, track_cov, track_stdev, track_time])