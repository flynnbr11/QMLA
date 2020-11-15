import hexp

import sys
import os
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
from matplotlib.lines import Line2D
from scipy import linalg
import qinfer
import scipy
import time

import lfig
sys.path.append("/home/bf16951/QMD")
import qmla

# Test hexp package

def random_model(num_qubits):
    connections = 'J'.join(list([str(i) for i in range(1, 1+dim)]))
    model_terms = [
        'pauliLikewise_l{t}_{conn}_d{N}'.format(
            t = term, conn = connections, N=num_qubits
        ) for term in ['x', 'y', 'z']
    ]
    model_params = {
        # t : np.random.uniform(0, 1)
        t : 1
        for t in model_terms
    }
    
    hamiltonian = None
    for t in model_params: 
        if hamiltonian is None: 
            hamiltonian = model_params[t] * qmla.construct_models.compute(t)
        else:
            hamiltonian += model_params[t] * qmla.construct_models.compute(t)
    
    return hamiltonian

hexp_tests_df = pd.DataFrame()

count_failures = 0 

test_run = False
if test_run: 
    print("Test setup")
    num_iterations = 2
    num_qubits = [2]
    precision_orders = [9]
    dt = 0.25
    ev_times = list(np.arange(dt, 0.51, dt))
else:
    num_iterations = 100
    num_qubits = range(1, 10)
    precision_orders =[12] # [3, 6, 9, 12, 15]
    # Times
    dt = 0.25
    linear_low_times = list(np.arange(dt, 5, dt))
    higher_times = list(np.arange(5, 51, 5))
    vhigh_times = [100, 500, 1000]
    ev_times = linear_low_times + higher_times + vhigh_times

precisions = [10**(-1*p) for p in precision_orders]


for precision_required in precisions:
    for dim in num_qubits:
        for ev_time in ev_times:
            for i in range(num_iterations):
                hamiltonian = random_model(num_qubits = dim)
                h = hexp.UnitaryEvolvingMatrix(
                    hamiltonian, 
                    evolution_time = ev_time,
                    precision_required = precision_required, 
                )
                
                default_t_start = time.time()
                true_u = h.expm_default()
                default_time = time.time() - default_t_start

                result = pd.Series({
                    'num_qubits' : int(h.num_qubits), 
                    'method_time' : default_time, 
                    'default_time' : default_time, 
                    'evolution_time' : h.evolution_time, 
                    'scalar' : np.round(h.scalar, 1), 
                    'method' : r"{}".format("default"), 
                    'iteration' : i, 
                    'did_not_crash' : 1, 
                    'within_precision' : 1, 
                    'objectively_close' : 1, 
                    'diff_from_default' : 0,
                    'precision_required' : precision_required,  
                    'fraction_time_reqd' : 1
                })

                hexp_tests_df = hexp_tests_df.append(result, ignore_index=True)


                for method in [
                    # 'linalg' , 
                    # 'linalg_sparse', 
                    'hermitian', 'hermitian_sparse'
                ]:
                    t1 = time.time()
                    try:
                        u = h.expm_via_method(method=method)
                        did_not_crash = True
                    except:
                        did_not_crash = False
                    time_taken = time.time() - t1
                    
                    if did_not_crash:
                        within_precision = np.allclose(u, true_u, 
                                               atol = 1*precision_required)
                        diff = np.max(np.abs(u - true_u))
                        # objectively close if small frobenius norm
                        # objectively_close = np.linalg.norm(u - true_u) < 1e-6
                        objectively_close = (diff <= 1e-4)
                        
                    else:
                        within_precision = np.NaN
                        diff = np.NaN
                        objectively_close = np.NaN

                    result = pd.Series({
                        'num_qubits' : int(h.num_qubits), 
                        'method_time' : time_taken, 
                        'default_time' : default_time, 
                        'evolution_time' : h.evolution_time, 
                        'scalar' : np.round(h.scalar, 1), 
                        'method' : r"{}".format(method), 
                        'did_not_crash' : bool(did_not_crash), 
                        'iteration' : i, 
                        'within_precision' : within_precision, 
                        'objectively_close' : objectively_close, 
                        'diff_from_default' : diff,
                        'precision_required' : precision_required,  
                        'fraction_time_reqd' : time_taken/default_time
                    })

                    hexp_tests_df = hexp_tests_df.append(result, ignore_index=True)


# Work out success rates
# hexp_tests_df['scalar'] = hexp_tests_df.scalar.round(1)

qubit_numbers = list(hexp_tests_df.num_qubits.unique())
evolution_time_values = list(hexp_tests_df.evolution_time.unique())
methods = list(hexp_tests_df.method.unique())
precisions_available = list(hexp_tests_df.precision_required.unique())

scalar_interval = 0.5
scalar_ranges = np.arange(
    hexp_tests_df.scalar.min(), 
    hexp_tests_df.scalar.max()+0.1, 
    scalar_interval
)
scalar_range_pairs = [
    (scalar_ranges[i], scalar_ranges[i+1]) 
    for i in range(len(scalar_ranges)-1)
]


success_rates = pd.DataFrame()
for prec in precisions_available:
    for m in methods:
        for q in qubit_numbers:
            for s1, s2 in scalar_range_pairs:

                subset = hexp_tests_df[
                    (hexp_tests_df.num_qubits == q)
                    & (hexp_tests_df.scalar >= s1)
                    & (hexp_tests_df.scalar < s2)
                    & (hexp_tests_df.method == m)
                    & (hexp_tests_df.precision_required == prec)
                ]
                # print(subset)
                if len(subset) == 0:
                    continue

                # summarise result
                good_results = subset[
                    (subset.did_not_crash == 1)
                    & (subset.objectively_close == 1)
                ]
                median_compute_time = good_results.method_time.median()
                success_probability = len(good_results) / len(subset)
                fraction_time_reqd = subset.fraction_time_reqd.median()
                delta = success_probability - fraction_time_reqd
                if m == 'default': 
                    delta = 0 

                this_config = pd.Series({
                    'method' : m, 
                    'scalar' : s2, # upper limit for this range
                    'num_qubits' : q,
                    'precision_required' : prec,
                    'time' : median_compute_time, 
                    'success_probability' :  success_probability,
                    'fraction_time_reqd' : fraction_time_reqd,
                    'delta' : delta, 
                })
                success_rates = success_rates.append(
                    this_config, ignore_index=True)


                
lookup_table = pd.DataFrame()
for prec in precisions_available:
    for q in qubit_numbers:
        computed_scalars = sorted(success_rates[
            (success_rates.num_qubits == q)
            & ( success_rates.precision_required == prec)
        ].scalar.unique())

        for s in computed_scalars:     
            lookup = success_rates[
                (success_rates.num_qubits == q)
                & (success_rates.scalar == s)
                & (success_rates.precision_required == prec)    
            ]
            idx_max_delta = lookup.delta.idxmax()
            best_method = lookup.loc[idx_max_delta].method

            lookup_summary = {
                'num_qubits' : q, 
                'scalar' : s, 
                'precision_required' : prec, 
                'best_method' : best_method
            }
            
            lookup_table = lookup_table.append(
                lookup_summary, 
                ignore_index=True
            )


for p in precision_orders:
    
    prec = 10**(-1*p)
    lookup_pivot = pd.pivot(
        columns = 'num_qubits',
        index = 'scalar',
        values = 'best_method',
        data = lookup_table[lookup_table.precision_required == prec]
    )
    lookup_pivot.fillna('default', inplace=True)

    results_folder = "/home/bf16951/hexp/hexp/precision_{}".format(p)
    try:
        os.makedirs(results_folder)
    except:
        pass
    print("Storing results to {}".format(results_folder))

    hexp_tests_df[
        hexp_tests_df.precision_required == prec
    ].to_pickle(
        os.path.join(results_folder, "hexp_tests.p")
    )                
    success_rates[
        success_rates.precision_required == prec
    ].to_pickle(
        os.path.join(results_folder, "success_rates.p")
    )
    lookup_table[
        lookup_table.precision_required == prec
    ].to_pickle(
        os.path.join(results_folder, "full_lookup_table.p")
    )
    lookup_pivot.to_pickle(
        os.path.join(results_folder, "method_pivot_table.p")
    )