import numpy as np
import scipy as sp
import os
import time
import copy
import qinfer as qi
import random

import redis
import pickle
import matplotlib.colors


class StorageUnit():
    r"""
    Generic object to which results can be pickled for later use/analysis.
    """

    def __init__(self, data_to_store=None):
        if data_to_store is not None:
            for k in data_to_store:
                self.__setattr__(
                    k, data_to_store[k]
                )


def round_nearest(x, a):
    r"""
    Round to nearest specified interval, e.g. a=0.05.
    """
    return round(round(x / a) * a, 2)


def resource_allocation(
    base_qubits,
    base_terms,
    max_num_params,
    this_model_qubits,
    this_model_terms,
    num_experiments,
    num_particles,
    given_resource_as_cap=True
):
    r"""
    Given a set of resources, work out the proportional resources that
    this model should have.
    i.e. if Ne experiments and Np particles are available to the most complex
    model, simpler models get a fraction of those experiments/particles
    on the assumption that they will learn faster.
    """
    new_resources = {}
    if given_resource_as_cap == True:
        # i.e. reduce number particles for models with fewer params
        proportion_of_particles_to_receive = (
            this_model_terms / max_num_params
        )
        print(
            "Model gets proportion of particles:",
            proportion_of_particles_to_receive
        )

        if proportion_of_particles_to_receive < 1:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = max(
                int(
                    proportion_of_particles_to_receive
                    * num_particles
                ),
                10
            )
        else:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = num_particles

    else:
        # increase proportional to number params/qubits
        qubit_factor = float(this_model_qubits / base_qubits)
        terms_factor = float(this_model_terms / base_terms)

        overall_factor = int(qubit_factor * terms_factor)

        if overall_factor > 1:
            new_resources['num_experiments'] = overall_factor * num_experiments
            new_resources['num_particles'] = overall_factor * num_particles
        else:
            new_resources['num_experiments'] = num_experiments
            new_resources['num_particles'] = num_particles

    print("New resources:", new_resources)
    return new_resources


def format_experiment(
    qinfer_model,
    qhl_final_param_estimates,
    time,
    final_learned_params=None,
):
    r"""
    Format a given set of data as an experiment that can be interpreted by the QInfer model.
    """
    exp = np.empty(
        len(time),
        dtype=qinfer_model.expparams_dtype
    )
    exp['t'] = time
    try:
        for dtype in qinfer_model.expparams_dtype:
            term = dtype[0]
            if term in qhl_final_param_estimates:
                exp[term] = qhl_final_param_estimates[term]
    except BaseException:
        print("failed to set exp from param estimates.\nReturning:", exp)
    return exp


def flatten(l):
    r"""
    Flatten a list of lists into a single list.
    """
    return [item for sublist in l for item in sublist]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    r"""
    Cuf of the ends of a given colour map.
    """
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def n_qubit_nv_gali_model(
    n_qubits,
    rotation_terms=['x', 'y', 'z'],
    coupling_terms=['x', 'y', 'z'],
):
    r"""
    Compose a model string for an NV system according to Gali model.

    :param int n_qubits: number of qubits to construct model of
    :param list rotation_terms: Pauli terms acting on the electron spin (generically: first qubit)
    :param list coupling_terms: Pauli terms of interaction between first and all other qubits,
        i.e. terms like PauliX TENSOR_PROD PauliX for 2 qubit, x-coupling
    """

    terms = [
        'pauliSet_1_{o}_d{N}'.format(o=operator, N=n_qubits)
        for operator in rotation_terms
    ]
    for k in range(2, n_qubits + 1):
        new_terms = [
            'pauliSet_1J{k}_{o}J{o}_d{N}'.format(k=k, o=operator, N=n_qubits)
            for operator in coupling_terms
        ]
        terms.extend(new_terms)
    terms = sorted(terms)
    return '+'.join(terms)
