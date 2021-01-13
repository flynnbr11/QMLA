import numpy as np
import scipy as sp
import os
import time
import copy
import random
import itertools

import qinfer as qi
import redis
import pickle
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import qmla.construct_models

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


def ensure_consisten_num_qubits_pauli_set(initial_model, new_dimension=None):
    individual_terms = qmla.construct_models.get_constituent_names_from_name(initial_model)
    
    if new_dimension is None: 
        max_dimension = max([
            qmla.construct_models.get_num_qubits(term)
            for term in individual_terms            
        ])
        new_dimension = max_dimension

    separate_terms = []
    for model in individual_terms:
        components = model.split('_')

        for c in components:
            if c[0] == 'd':
                # remove the dimension indicator from model
                components.remove(c)

        new_component = "d{}".format(new_dimension)
        components.append(new_component)
        new_mod = '_'.join(components)
        separate_terms.append(new_mod)

    full_model = '+'.join(separate_terms)
    return full_model


def plot_probes_on_bloch_sphere(
    probe_dict, 
    # num_probes, 
    save_to_file=None,
    **kwargs
):
    try:
        import qutip as qt
    except:
        print("Qutip not installed")
        raise

    bloch = qt.Bloch()

    # isolate 1 qubit probes
    probe_ids = [t for t in list(probe_dict.keys()) if t[1] == 1]

    for pid in probe_ids:
        state = probe_dict[pid]
        a = state[0]
        b = state[1]
        A = a * qt.basis(2, 0)
        B = b * qt.basis(2, 1)
        vec = (A + B)
        bloch.add_states(vec)

    if save_to_file is not None:
        bloch.save(save_to_file)
    else:
        bloch.show()


def plot_subset_eval_probes(
    true_hamiltonian,
    probe_dict, 
    subset_probes, 
    measurement_probability_function, 
    times, 
    fig, 
    dynamics_ax, 
    bloch_ax, 
):
    r"""
    Retained separately in case we later want to plot all eval probes instead of just a sample
    """ 
    try:
        import qutip as qt
    except:
        print("Qutip not installed")
        raise

    colours = ['red', 'green', 'cyan', 'orange', 'brown', 'blue', 'pink']
    linestyles=['dashed', 'dotted', 'dashdot']
    linestyles = itertools.cycle(linestyles)
    iter_colours = itertools.cycle(colours)
    num_probes_per_subplot = len(colours)

    bloch = qt.Bloch(fig=fig, axes = bloch_ax)
    try:
        bloch_ax.axis('square') # to get a nice circular plot
    except:
        pass

    for pid in subset_probes:
        probe = probe_dict[pid]

        ev = [
            measurement_probability_function(
                ham = true_hamiltonian, 
                t = t, 
                state = probe
            )
            for t in times
        ]

        dynamics_ax.plot(
            times, 
            ev, 
            c=next(iter_colours),
            ls=next(linestyles),
            lw = 3,
            label="{}".format(pid[0])
        )

        corresponding_single_qubit_probe = probe_dict[(pid[0], 1)]   
        A = corresponding_single_qubit_probe[0] * qt.basis(2, 0)
        B = corresponding_single_qubit_probe[1] * qt.basis(2, 1)
        vec = (A + B)
        bloch.add_states(vec)

    bloch.vector_color = colours
    bloch.render(fig=fig, axes=bloch_ax) # render to the correct subplot 
    dynamics_ax.set_ylabel('Expectation Value')
    dynamics_ax.set_xlabel('Time')
    dynamics_ax.legend()




def plot_evaluation_dataset(
    evaluation_data, 
    true_hamiltonian,
    measurement_probability_function,
    num_probes_to_plot=6, 
    save_to_file=None
):
    times = sorted(np.array(evaluation_data['experiments'])['t'])
    probe_dict = evaluation_data['probes']
    keys = list(probe_dict.keys())
    true_model_num_qubits = np.log2( np.shape(true_hamiltonian)[0] )
    probe_ids = sorted([t for t in list(probe_dict.keys()) if t[1] == true_model_num_qubits])

    # Plot
    fig, axes = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True,
    )

    nrows = 1 # TODO plot more than just a sample
    ncols = 2
    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        width_ratios=[3, 1]
    )

    row = 0
    dynamics_ax = fig.add_subplot(gs[row, 0])
    bloch_ax = fig.add_subplot(gs[row, 1], projection='3d')


    subset_probes = sorted(probe_ids[:num_probes_to_plot])

    plot_subset_eval_probes(
        true_hamiltonian = true_hamiltonian, 
        measurement_probability_function = measurement_probability_function,
        subset_probes = subset_probes,
        probe_dict = probe_dict, 
        times = times, 
        fig = fig,  
        dynamics_ax = dynamics_ax, 
        bloch_ax = bloch_ax, 
    )

    if save_to_file is not None:
        fig.savefig(save_to_file)
    else:
        plt.show()
    