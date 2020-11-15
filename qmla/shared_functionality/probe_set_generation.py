r"""
Functions to generate sets of probe states to be used for training models.

These functions are set to exploration strategy attributes, which are then called in wrapper functions. 
- probe_generation_function: 
    used for training, assumed to be the probes used on the system
    i.e. probes implemented during experiments. 
- simulator_probe_generation_function: 
    used for training, for the simulator. 
    Should be the same as used for system, 
    but in practice this is not always the case. 
    Note exploration_strategy.shared_probes controls whether
    to default to the same probe set. 
- plot_probe_generation_function: 
    State to use for plotting purposes only (not training). 
    Plots all use the same set of probes for consistency

"""

import numpy as np
import itertools
from scipy import linalg, stats
import random

import qmla.utilities
import qmla.construct_models

###################################
# General useful functions
###################################

def random_probe(num_qubits):
    r"""
    Random probe of dimension num_qubits.
    """

    dim = 2**num_qubits
    real = []
    imaginary = []
    complex_vectors = []
    for i in range(dim):
        real.append(np.random.uniform(low=-1, high=1))
        imaginary.append(np.random.uniform(low=-1, high=1))
        complex_vectors.append(real[i] + 1j * imaginary[i])

    a = np.array(complex_vectors)
    norm_factor = np.linalg.norm(a)
    probe = complex_vectors / norm_factor
    while (
        np.abs(np.linalg.norm(probe)) - 1
        >
        1e-14
    ):
        print("generating new random probe..")
        probe = random_probe(num_qubits)

    return probe

def n_qubit_plus_state(num_qubits):
    r"""
    Probe of dimension num_qubits, where for each qubit |+> is appended.
    """

    one_qubit_plus = (1 / np.sqrt(2) + 0j) * np.array([1, 1])
    plus_n = one_qubit_plus
    for i in range(num_qubits - 1):
        plus_n = np.kron(plus_n, one_qubit_plus)
    return plus_n

def n_qubit_repeat_probe(num_qubits, input_state = np.array([1,0]) ):
   
    state = input_state
    for i in range(num_qubits - 1):
        state = np.tensordot(state, input_state, axes=0).flatten('c')
    return state

def harr_random_probe(num_qubits=1):

    random_unitary = stats.unitary_group.rvs(2**num_qubits)
    zero_probe = n_qubit_repeat_probe(num_qubits = num_qubits)
    random_state = np.dot(random_unitary, zero_probe)
    if not np.isclose(np.linalg.norm(random_state), 1, atol=1e-14):
        # call again until a normalised probe is generated
        print("Probe generated is not normalised")
        random_state = harr_random_probe(num_qubits = num_qubits)
    return random_state

###################################
# Default probe set
###################################

def separable_probe_dict(
    max_num_qubits,
    num_probes,
    **kwargs
):
    r"""
    Random separable probes. 

    Produces num_probes random states up to max_num_qubits. 
    Probes are indexed by dimension and an identifier, 
        e.g. (2, 10) is the 2-qubit version of probe 10.
    For each probe, 1-qubit states are generated at random,
        and tensor-producted with the probe id of smaller dimension, 
        such that, for N qubits,  probe i is:
        (N+1, i) = (N, i) \otimes r, 
        where r is a random 1-qubit probe.
    
    :param int max_num_qubits: Largest number of qubits to generate probes up to.
    :param int num_probes: How many probes to produce. 
    :return dict separable_probes: probe library indexed by (num-qubit, probe-id)
    """
    separable_probes = {}
    for i in range(num_probes):
        separable_probes[i, 0] = harr_random_probe(1)
        for j in range(1, 1 + max_num_qubits):
            if j == 1:
                separable_probes[i, j] = separable_probes[i, 0]
            else:
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        harr_random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
            norm = np.linalg.norm(separable_probes[i, j])
            while (
                np.abs(norm - 1) >
                1e-13

            ):
                print(
                    "non-unit norm: ",
                    norm
                )
                # keep replacing until a unit-norm
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        harr_random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
                norm = np.linalg.norm(separable_probes[i, j])
            # print("unit norm:", np.abs(1-norm) )

    return separable_probes

def tomographic_basis(
    max_num_qubits = 2, 
    num_probes = 10, 
    noise_level = 0.01, 
    **kwargs
):
    r"""
    To manually check something. Currently should be ideal for learning Y. 
    """
    probes = {}
    probe_list = [

        np.array([1, 0j]),
        np.array([0j, 1]),
        
        1/np.sqrt(2)*np.array([ 1+0j,  1+0j ]),
        1/np.sqrt(2)*np.array([ 1+0j, -1+0j ]),

        1/np.sqrt(2)*np.array([ 1, 1j ]),
        1/np.sqrt(2)*np.array([ 1,-1j ]),

    ]

    available_probes = itertools.cycle(probe_list)
    for j in range(num_probes):
        probe = next(available_probes)
        # add noise and normalise
        probe += noise_level * random_probe(1)
        probe /= np.linalg.norm(probe)
        
        probes[(j, 1)] = probe
    
    for N in range(2, max_num_qubits+1):

        for j in range(num_probes):
            # add noise and normalise
            new = next(available_probes)
            new += noise_level * random_probe(1)
            new /= np.linalg.norm(new)

            probes[(j, N)] = np.kron(
                probes[(j, N-1)], 
                new
            )
    return probes
    

###################################
# Specific probe sets
## e.g. matching experiment.
###################################

def manual_set_probes(
    max_num_qubits = 2, 
    num_probes = 10, 
    noise_level = 0, 
    **kwargs
):
    r"""
    To manually check something. Currently should be ideal for learning Y. 
    """
    probes = {}
    probe_list = [

        np.array([1, 0j]),
        np.array([0, 1]),
        
        # np.array([1j, 0]),
        # np.array([0, 1j]),

        1/np.sqrt(2)*np.array([1,1]),
        1/np.sqrt(2)*np.array([1,-1]),

        1/np.sqrt(2)*np.array([1,1j]),
        1/np.sqrt(2)*np.array([1,-1j]),

    ]
    available_probes = itertools.cycle(probe_list)
    for j in range(num_probes):
        probes[(j, 1)] = next(available_probes)
    
    for N in range(2, max_num_qubits+1):
        for j in range(num_probes):
            new = next(available_probes)
            new += noise_level * random_probe(1)
            new /= np.linalg.norm(new)

            probes[(j, N)] = np.kron(
                probes[(j, N-1)], 
                new
            )
    return probes


# test the probe transformer - 
# probes should match in first quantisation and second quantisation, 
# when generated consistently from these methods


def get_fh_amplitudes():
    r"""For consistency, use this both for first and second quantisation test probes."""
    amplitudes = [
        ( 1, 0 ) ,
        ( np.sqrt(1/2), np.sqrt(1/2) ) ,
        ( np.sqrt(1/3), np.sqrt(2/3) ) ,
        ( np.sqrt(1/4), np.sqrt(3/4) ) ,
        ( np.sqrt(1/5), np.sqrt(4/5) ) ,
        ( np.sqrt(1/6), np.sqrt(5/6) ) ,        
    ]

    random_generated_probes = [
        # this set worked when learned by FH
        # down                                      up
        (0.2917277328868723-0.8933988007482713j, 0.10473305565400699-0.32521454416986145j),
        (0.11122644079879992+0.1595943558100543j, -0.7513991464733198-0.6305217229723107j),
        (0.28064370126754434-0.351305094219218j, -0.8493632346334402-0.27641624295163864j),
        (0.2697895657085366-0.048882904448255944j, -0.7554567811194581+0.595070671221603j),
        (-0.011033351809655593-0.2666075610217924j, 0.8261679156032085+0.49623104375049476j),
        (-0.41877834698776184-0.2531210159787065j, -0.5076736765270682-0.7090993481350797j),
        (-0.7550629339850182+0.4348391445682299j, -0.399627248364159-0.2847682328455842j),
        (-0.4911366925901858+0.06873816771050045j, -0.1506997100526583+0.8551896929228165j),
        (0.512704399952025+0.2746240218810056j, -0.5892032449747409-0.5608523700466731j),
        (-0.8708048547409585-0.19844529152441565j, 0.24732142630473528+0.3756999911125355j),
    ]
    return random_generated_probes # amplitudes
   

def one_site_probes_first_quantisation():
    amplitudes = get_fh_amplitudes()
    
    one_site_probes = []
    for a in amplitudes:
        phases = [
            np.array([a[0], a[1]]),
            # np.array([a[0], -a[1]]),

            # np.array([a[0], 1j*a[1]]),
            # np.array([1j*a[0], a[1]]),

        ]
        one_site_probes.extend(phases)
        
        
    one_site_probes = [ np.array(a) for a in one_site_probes ]
    one_site_probes = itertools.cycle(one_site_probes)
    return one_site_probes

def one_site_probes_second_quantisation():
    r"""
    This picture uses the occupation basis:
    |down> = |10>  = (0,0,1,0);
    |up> = |10>  = (0,1,0,0);
    
    """
    
    amplitudes = get_fh_amplitudes()
    
    one_site_probes = []
    for a in amplitudes:
        phases = [
            np.array([ 0, a[1], a[0], 0 ]),
            # np.array([ 0, -a[1], a[0], 0 ]),

            # np.array([ 0, 1j*a[1], a[0], 0 ]),
            # np.array([ 0 , a[1], 1j*a[0], 0 ]),

        ]
        one_site_probes.extend(phases)
        
        
    one_site_probes = [ np.array(a) for a in one_site_probes ]
    one_site_probes = itertools.cycle(one_site_probes)
    return one_site_probes

def test_probes_first_quantisation(
    num_probes = 10, 
    max_num_qubits = 4, 
    **kwargs
):
    
    designed_probes = one_site_probes_first_quantisation()
    
    probes = {}
    
    for p in range(num_probes):
        probes[(p, 1)] = next(designed_probes)
        
        for nq in range(2, max_num_qubits+1):
        
            pid = (p, nq)
            new_probe = np.tensordot(
                probes[(p, nq-1)], 
                probes[(p, 1)], 
                axes = 0
            ).flatten('c')
            probes[pid] = new_probe
            
            norm = np.linalg.norm(new_probe)
            if not np.isclose(norm, 1, atol=1e-6):
                print("norm=", norm)
    return probes

def test_probes_second_quantisation(
    num_probes = 10, 
    max_num_qubits = 4, 
    **kwargs
):
    
    designed_probes = one_site_probes_second_quantisation()
    
    probes = {}
    
    for p in range(num_probes):
        probes[(p, 1)] = next(designed_probes)
        
        for nq in range(2, max_num_qubits+1):
        
            pid = (p, nq)
            new_probe = np.tensordot(
                probes[(p, nq-1)], 
                probes[(p, 1)], 
                axes = 0
            ).flatten('c')
            probes[pid] = new_probe
            
            norm = np.linalg.norm(new_probe)
            if not np.isclose(norm, 1, atol=1e-6):
                print("norm=", norm)
            

    return probes



def eigenbasis_of_first_qubit(
    max_num_qubits=2,
    num_probes=40,
    **kwargs
):
    probes = {}
    bases_to_learn = ['x', 'y', 'z']
    for N in range(1, max_num_qubits+1):
        bases = ['pauliSet_1_{}_d{}'.format(b, N) for b in bases_to_learn ]
        base_matrices = [qmla.construct_models.compute(b) for b in bases]
        eig_vectors_list = qmla.utilities.flatten([np.linalg.eig(b)[1] for b in base_matrices])
        eig_vectors = itertools.cycle(eig_vectors_list)

        for j in range(num_probes):
            probes[(j, N)] = next(eig_vectors)
    return probes
    

def NV_centre_ising_probes_plus(
    max_num_qubits=2,
    num_probes=40,
    noise_level=0.03,  # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    **kwargs
):
    r"""
    Returns a dict of separable probes where the first qubit always acts on |+>.

    Used for QMLA on NV centre, experiment in Bristol 2016.
    Probe library has each probe like |+>|r>..|r>, where |r> is random 1-qubit state.

    :param int max_num_qubits: Largest number of qubits to generate probes up to.
    :param int num_probes: How many probes to produce. 
    :param float noise_level: factor to multiple generated states by to simulate noise.
    :return dict separable_probes: probe library. 
    """
    minimum_tolerable_noise=1e-6
    # minimum_tolerable_noise needed
    # or else run the risk of having
    # exact eigenstate and no learning occurs, and crashes.
    print(
        "[NV_centre_ising_probes_plus] min tol noise:",
        minimum_tolerable_noise,
        "noise level:", noise_level
    )
    if minimum_tolerable_noise > noise_level:
        noise_level = minimum_tolerable_noise
        # print("using minimum_tolerable_noise")
    plus_state = np.array([1 + 0j, 1]) / np.sqrt(2)
    random_noise = noise_level * random_probe(1)
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus / norm_factor
    print("\n\t noisy plus:", noisy_plus)
    # print("\n\t has type:", type(noisy_plus))

    separable_probes = {}
    for i in range(num_probes):
        #        separable_probes[i,0] = plus_state
        separable_probes[i, 0] = noisy_plus
        for j in range(1, 1 + max_num_qubits):
            if j == 1:
                separable_probes[i, j] = separable_probes[i, 0]
            else:
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        # noisy_plus,
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0,
                    np.linalg.norm(separable_probes[i, j]),
                    atol=1e-14
                ) is False
            ):
                print("non-unit norm: ",
                      np.linalg.norm(separable_probes[i, j])
                      )
                # keep replacing until a unit-norm
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
    return separable_probes


def plus_plus_with_phase_difference(
    max_num_qubits=2,
    num_probes=40,
    noise_level=0.03,  # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    # noise_level=0.03,  # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    # *args,
    **kwargs
):
    r"""
    Probes |+> |+'> ... |+'>

    To match NV centre experiment in Bristol, 2016. 
    First qubit is prepared in |+>; 
        second (and subsequent) qubits (representing environment)
        assumed to be in |+'> = |0> + e^{iR}|1> (normalised, R = random phase)
    i.e. 
        1 qubit  : |+>
        2 qubits : |+>|+'> 
        N qubits : |+> |+'> ... |+'>

    :param int max_num_qubits: Largest number of qubits to generate probes up to.
    :param int num_probes: How many probes to produce. 
    :param float noise_level: factor to multiple generated states by to simulate noise.
    :return dict separable_probes: probe library. 

    """

    # minimum_tolerable_noise=1e-6
    # minimum_tolerable_noise needed
    # or else run the risk of having
    # exact eigenstate and no learning occurs, and crashes.

    # if minimum_tolerable_noise > noise_level:
    #     noise_level = minimum_tolerable_noise
    # noise_level = 0.01
    plus_state = np.array([1 + 0j, 1]) / np.sqrt(2)
    random_noise = noise_level * random_probe(1)
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus / norm_factor
    print("[|++'> probes] Noise factor:", noise_level)

    separable_probes = {}
    for i in range(num_probes):
        separable_probes[i, 0] = noisy_plus
        for j in range(1, 1 + max_num_qubits):
            if j == 1:
                separable_probes[i, j] = separable_probes[i, 0]
            else:
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        # noisy_plus,
                        random_phase_plus(noise_level=noise_level),
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0,
                    np.linalg.norm(separable_probes[i, j]),
                    atol=1e-14
                ) is False
            ):
                print("non-unit norm: ",
                      np.linalg.norm(separable_probes[i, j])
                      )
                # keep replacing until a unit-norm
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_phase_plus(noise_level=noise_level),
                        axes=0
                    ).flatten(order='c')
                )
    return separable_probes


def random_phase_plus(
    noise_level=1e-5
):
    r"""
    To produce |+'> = |0> + e^{iR}|1> (normalised, R = random phase)
    """

    random_phase = random.uniform(0, np.pi)
    rand_phase_plus = np.array(
        [
            1.0 + 0.j,
            np.exp(1.0j * random_phase)
        ]
    ) / np.sqrt(2)

    noisy_state = noise_level * random_probe(1)
    rand_phase_plus += noisy_state
    norm = np.linalg.norm(rand_phase_plus)
    rand_phase_plus = rand_phase_plus / norm
    return rand_phase_plus


###################################
# General purpose probe dictionaries for specific cases
## e.g. experimental method, generalised to multiple dimensions
###################################

def plus_probes_dict(
    max_num_qubits,
    noise_level=0.0,  # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    minimum_tolerable_noise=0,
    **kwargs
):
    r"""
    Produces exactly |+>|+>...|+> with no noise. 
    """

    num_probes = kwargs['num_probes']
    # if minimum_tolerable_noise > noise_level:
    #     noise_level = minimum_tolerable_noise
    probe_dict = {}
    for j in range(num_probes):
        for i in range(1, 1 + max_num_qubits):
            # dict key is tuple of form (0,i) for consistency with other probe
            # dict generation functions.
            new_probe = n_qubit_plus_state(i)
            noisy_state = random_probe(i) * noise_level
            noisy_probe = new_probe + noisy_state
            norm = np.linalg.norm(noisy_probe)
            noisy_probe = noisy_probe / norm
            probe_dict[(j, i)] = noisy_probe
    return probe_dict


def zero_state_probes(max_num_qubits=9, **kwargs):
    r"""
    Probe library: |0>|0> ... |0>
    """

    zero = np.array([1 + 0j, 0])
    num_probes = kwargs['num_probes']
    probes = {}

    for q in range(1, 1 + max_num_qubits):
        for j in range(num_probes):
            state = zero
            for i in range(q - 1):
                state = np.tensordot(state, zero, axes=0).flatten('c')
            probes[(j, q)] = state

    return probes

###################################
# Exploration Strategy specific probes
###################################

# Fermi Hubbard model -- requires encoding via Jordan-Wigner transformation.
def separable_fermi_hubbard_half_filled(
    max_num_qubits,
    num_probes,
    **kwargs
):
    r"""
    Probes for Fermi-Hubbard Hamiltonians. 

    Generates separable probes using a half filled system,
    i.e. N spins in N sites.
    First generates completely random probes,
    then projects so that for each dimension
    the probe is projected onto the subspace of n
    fermions on the n dimensional space, 
    i.e. the half-filled basis. 
    
    :param int max_num_qubits: Largest number of qubits to generate probes up to.
    :param int  num_probes: How many probes to produce. 
    :return dict separable_probes: probe library. 
    """
    
    separable_probes = {}
    for i in range(num_probes):
        separable_probes[i, 0] = random_superposition_occupation_basis()
        for j in range(1, 1 + max_num_qubits):
            if j == 1:
                separable_probes[i, j] = separable_probes[i, 0]
            else:
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_superposition_occupation_basis(),
                        axes=0
                    ).flatten(order='c')
                )
            norm = np.linalg.norm(separable_probes[i, j])
            while (
                np.abs(norm - 1) >
                1e-13

            ):
                print(
                    "non-unit norm: ",
                    norm
                )
                # keep replacing until a unit-norm
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_superposition_half_filled_occupation_basis(),
                        axes=0
                    ).flatten(order='c')
                )
                norm = np.linalg.norm(separable_probes[i, j])
            # print("unit norm:", np.abs(1-norm) )

    # project onto subspace representing half-filled (n fermions on n sites)
    # by removing amplitude of basis vectors of different form
    # eg 2 sites, keep |0101>; remove |0111>, etc
    combined_base_vectors = {
        i: sum(
            get_half_filled_basis_vectors(i)
        ) for i in range(1, max_num_qubits + 1)
    }
    for j in range(1, 1 + max_num_qubits):
        bv = combined_base_vectors[j]
        for i in range(num_probes):
            p = separable_probes[(i, j)]
            p *= bv
            p /= np.linalg.norm(p)
            separable_probes[(i, j)] = p

    return separable_probes


def get_half_filled_basis_vectors(
    num_sites
):
    half_filled_list = [0, 1] * num_sites

    # all the ways in which half filled can be spread across lattice
    perms = list(itertools.permutations(half_filled_list))
    perms = [
        ''.join([str(j) for j in this_el])
        for this_el in perms
    ] # restore as strings instead of lists
    perms = list(set(perms)) # unique strings

    basis = [state_from_string(s) for s in perms]
    return basis


def random_superposition_occupation_basis():
    r"""
    Returns a random superposition over the occupation basis of a single site
        which can be singly or doubly occupied.
    #TODO equivalent to 2 qubit Harr-random?

    vacant = np.array([1,0])
    occupied = np.array([0,1])
    down = np.kron(occupied, vacant) #|10>
    up = np.kron(vacant, occupied) #|01>
    """

    down = np.array([0, 0, 1, 0])
    up = np.array([0, 1, 0, 0])
    unoccupied = np.array([1, 0, 0, 0])
    doubly_occupied = np.array([0, 0, 0, 1])

    alpha = np.random.randn() + 1j * np.random.randn()
    beta = np.random.randn() + 1j * np.random.randn()
    gamma = np.random.randn() + 1j * np.random.randn()
    delta = np.random.randn() + 1j * np.random.randn()

    state = (alpha * down) + (beta * up) + \
        (gamma * unoccupied) + (delta * doubly_occupied)
    state = state / np.linalg.norm(state) # normalise
    return state


def state_from_string(
    s,
    basis_states={
        '0': np.array([1, 0]),
        '1': np.array([0, 1])
    }
):
    # input s, form 1001, returns corresponding basis vector
    state = None
    for i in s:
        if state is None:
            state = basis_states[i]
        else:
            state = np.kron(
                state,
                basis_states[i]
            )
    state = np.array(state)
    return state


def fermi_hubbard_half_filled_superposition(
    max_num_qubits,
    **kwargs
):
    num_sites = max_num_qubits
    probe_dict = {}
    for N in range(1, 1 + num_sites):

        state = None
        for i in range(1, N + 1):
            for spin_type in ['up', 'down']:

                new_state = vector_from_fermion_state_description(
                    {
                        'num_sites': N,
                        'occupations': {
                            i: [spin_type]
                        }
                    }
                )

                if state is None:
                    state = new_state
                else:
                    state += new_state

        state = state / np.linalg.norm(state)

        probe_dict[(1, N)] = state
    print("[fermi_hubbard_half_filled_superposition] keys:", probe_dict.keys())
    return probe_dict


def vector_from_fermion_state_description(state):

    occupied = np.array([0, 1])
    vacant = np.array([1, 0])

    vector = 1 # so we can perform tensor product on it
    for i in range(1, 1 + state['num_sites']):
        try:
            occupation = state['occupations'][i]
        except BaseException:
            occupation = ['vacant']

        # in order: (i, down), (i, up)
        # i.e.  2 qubit encoding per site
        if 'down' in occupation:
            vector = np.kron(vector, occupied)
        else:
            vector = np.kron(vector, vacant)

        if 'up' in occupation:
            vector = np.kron(vector, occupied)
        else:
            vector = np.kron(vector, vacant)

    return vector

def fermi_hubbard_occupation_basis_down_in_first_site(
    max_num_qubits,
    num_probes = 1, 
    **kwargs
):
    r"""
    To test hopping out of first site
    """

    print("Fermi hubbard down in first site probes. max num qubits:",
        max_num_qubits
    )    
    probe_dict = {}    
    for j in range(num_probes):
        for n in range(1, max_num_qubits+1):
            num_occupation_locations = 2*n
            down_in_first_site = ['0']*num_occupation_locations
            down_in_first_site[0] = '1'
            down_in_first_site = ''.join( down_in_first_site )
            probe_dict[(j, n)] = state_from_string(down_in_first_site)
    
    return probe_dict

def fermi_hubbard_occupation_basis_up_in_first_site(
    max_num_qubits,
    num_probes = 1, 
    **kwargs
):
    r"""
    To test hopping out of first site
    """

    print("Fermi hubbard down in first site probes. max num qubits:",
        max_num_qubits
    )    
    probe_dict = {}    
    for j in range(num_probes):
        for n in range(1, max_num_qubits+1):
            num_occupation_locations =2*n
            down_in_first_site = ['0']*num_occupation_locations
            down_in_first_site[1] = '1'
            down_in_first_site = ''.join(  down_in_first_site )
            probe_dict[(j, n)] = state_from_string(down_in_first_site)
    
    return probe_dict

def fermi_hubbard_occupation_basis_down_in_all_sites(
    max_num_qubits,
    num_probes = 1, 
    **kwargs
):
    probe_dict = {}
    
    for j in range(num_probes):
        for n in range(1, max_num_qubits+1):
            down_in_all_sites = '10'*n
            probe_dict[(j, n)] = state_from_string(down_in_all_sites)
    
    return probe_dict
###################################
# Testing/development
###################################

# probes generated according to Pauli matrices' eigenvectors
core_operator_dict = {
#     'a': np.array([[0 + 0.j, 1 + 0.j], [0 + 0.j, 0 + 0.j]]),
#     's': np.array([[0 + 0.j, 0 + 0.j], [1 + 0.j, 0 + 0.j]])
    'i': np.array([[1 + 0.j, 0 + 0.j], [0 + 0.j, 1 + 0.j]]),
    'x': np.array([[0 + 0.j, 1 + 0.j], [1 + 0.j, 0 + 0.j]]),
    'y': np.array([[0 + 0.j, 0 - 1.j], [0 + 1.j, 0 + 0.j]]),
    'z': np.array([[1 + 0.j, 0 + 0.j], [0 + 0.j, -1 + 0.j]]),
}


eigvals = {
    k : linalg.eig(core_operator_dict[k])[0]
    for k in core_operator_dict
}
eigvecs = {
    k : linalg.eig(core_operator_dict[k])[1]
    for k in core_operator_dict
}
all_eigvecs = [eigvecs[k][l] for k in eigvecs for l in range(eigvecs[k].shape[0]) ]
eigvec_indices = range(len(all_eigvecs))
eigenvectors = {
    i : all_eigvecs[i]
    for i in eigvec_indices
}

def random_sum_eigenvectors():
#     num_to_sum = random.randrange(2, 5)
    num_to_sum = 1
    indices_to_include = []
    while len(indices_to_include) < num_to_sum:
        a = random.choice(eigvec_indices)
        if a not in indices_to_include: 
            indices_to_include.append(a)
    
    state = None
    for i in indices_to_include:
        if state is None: 
            state = eigenvectors[i]
        else: 
            state += eigenvectors[i]
        print("Including eig i={}: {}".format(i, eigenvectors[i]))
    return state/linalg.norm(state)

def pauli_eigenvector_based_probes(
    max_num_qubits,
    num_probes,
    **kwargs
):
    separable_probes = {}
    for i in range(num_probes):
#         separable_probes[i, 0] = random_probe(1)
        separable_probes[i, 0] = random_sum_eigenvectors()
        for j in range(1, 1 + max_num_qubits):
            if j == 1:
                separable_probes[i, j] = separable_probes[i, 0]
            else:
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_sum_eigenvectors(),
                        axes=0
                    ).flatten(order='c')
                )
            norm = np.linalg.norm(separable_probes[i, j])
            while (
                np.abs(norm - 1) >
                1e-13

            ):
                print(
                    "non-unit norm: ",
                    norm
                )
                # keep replacing until a unit-norm
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_sum_eigenvectors(),
#                         random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
                norm = np.linalg.norm(separable_probes[i, j])
            # print("unit norm:", np.abs(1-norm) )

    return separable_probes


# probes generated according to Pauli matrices' eigenvectors
# testing eigenvalue of Paulis probes - not in use
core_operator_dict = {
#     'a': np.array([[0 + 0.j, 1 + 0.j], [0 + 0.j, 0 + 0.j]]),
#     's': np.array([[0 + 0.j, 0 + 0.j], [1 + 0.j, 0 + 0.j]])
    'i': np.array([[1 + 0.j, 0 + 0.j], [0 + 0.j, 1 + 0.j]]),
    'x': np.array([[0 + 0.j, 1 + 0.j], [1 + 0.j, 0 + 0.j]]),
    'y': np.array([[0 + 0.j, 0 - 1.j], [0 + 1.j, 0 + 0.j]]),
    'z': np.array([[1 + 0.j, 0 + 0.j], [0 + 0.j, -1 + 0.j]]),
}


eigvals = {
    k : linalg.eig(core_operator_dict[k])[0]
    for k in core_operator_dict
}
eigvecs = {
    k : linalg.eig(core_operator_dict[k])[1]
    for k in core_operator_dict
}
all_eigvecs = [eigvecs[k][l] for k in eigvecs for l in range(eigvecs[k].shape[0]) ]
eigvec_indices = range(len(all_eigvecs))
eigenvectors = {
    i : all_eigvecs[i]
    for i in eigvec_indices
}

def random_sum_eigenvectors():
#     num_to_sum = random.randrange(2, 5)
    num_to_sum = 1
    indices_to_include = []
    while len(indices_to_include) < num_to_sum:
        a = random.choice(eigvec_indices)
        if a not in indices_to_include: 
            indices_to_include.append(a)
    
    state = None
    for i in indices_to_include:
        if state is None: 
            state = eigenvectors[i]
        else: 
            state += eigenvectors[i]
        print("Including eig i={}: {}".format(i, eigenvectors[i]))
    return state/linalg.norm(state)

def pauli_eigenvector_based_probes(
    max_num_qubits,
    num_probes,
    **kwargs
):
    separable_probes = {}
    for i in range(num_probes):
#         separable_probes[i, 0] = random_probe(1)
        separable_probes[i, 0] = random_sum_eigenvectors()
        for j in range(1, 1 + max_num_qubits):
            if j == 1:
                separable_probes[i, j] = separable_probes[i, 0]
            else:
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_sum_eigenvectors(),
                        axes=0
                    ).flatten(order='c')
                )
            norm = np.linalg.norm(separable_probes[i, j])
            while (
                np.abs(norm - 1) >
                1e-13

            ):
                print(
                    "non-unit norm: ",
                    norm
                )
                # keep replacing until a unit-norm
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_sum_eigenvectors(),
#                         random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
                norm = np.linalg.norm(separable_probes[i, j])
            # print("unit norm:", np.abs(1-norm) )

    return separable_probes
