import numpy as np
import itertools
from scipy import linalg
import random


# Default probe set

def separable_probe_dict(
    max_num_qubits,
    num_probes,
    **kwargs
):
    separable_probes = {}
    for i in range(num_probes):
        separable_probes[i, 0] = random_probe(1)
        for j in range(1, 1 + max_num_qubits):
            if j == 1:
                separable_probes[i, j] = separable_probes[i, 0]
            else:
                separable_probes[i, j] = (
                    np.tensordot(
                        separable_probes[i, j - 1],
                        random_probe(1),
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
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
                norm = np.linalg.norm(separable_probes[i, j])
            # print("unit norm:", np.abs(1-norm) )

    return separable_probes


def random_probe(num_qubits):
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


# Specific experimental probes

def NV_centre_ising_probes_plus(
    max_num_qubits=2,
    num_probes=40,
    noise_level=0.03,  # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    minimum_tolerable_noise=1e-6,
    # minimum_tolerable_noise needed
    # or else run the risk of having
    # exact eigenstate and no learning occurs, and crashes.
    # *args,
    **kwargs
):
    """
    Returns a dict of separable probes where the first qubit always acts on
    a plus state.
    """
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
    minimum_tolerable_noise=1e-6,
    # minimum_tolerable_noise needed
    # or else run the risk of having
    # exact eigenstate and no learning occurs, and crashes.
    # *args,
    **kwargs
):
    """
    1 qubit  : |+>
    2 qubits : |+>|+'>, where |+> = |0> + e^{iR}|1> (normalised, R = random phase)
    N qubits : |+> |+'> ... |+'>
    To be used for NV centre experimental data.
    """
    if minimum_tolerable_noise > noise_level:
        noise_level = minimum_tolerable_noise
    plus_state = np.array([1 + 0j, 1]) / np.sqrt(2)
    random_noise = noise_level * random_probe(1)
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus / norm_factor

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
    import random
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


# General purpose probe dictionaries for specific cases
## e.g. experimental method, generalised to multiple dimensions

def n_qubit_plus_state(num_qubits):
    one_qubit_plus = (1 / np.sqrt(2) + 0j) * np.array([1, 1])
    plus_n = one_qubit_plus
    for i in range(num_qubits - 1):
        plus_n = np.kron(plus_n, one_qubit_plus)
    return plus_n


def plus_probes_dict(
    max_num_qubits,
    noise_level=0.0,  # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    minimum_tolerable_noise=0,
    **kwargs
):
    num_probes = kwargs['num_probes']
    if minimum_tolerable_noise > noise_level:
        noise_level = minimum_tolerable_noise
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


# Fermi Hubbard model -- requires encoding via Jordan-Wigner transformation.
def separable_fermi_hubbard_half_filled(
    max_num_qubits,
    num_probes,
    **kwargs
):
    # generates separable probes in N sites;
    # then projects so that for each dimension
    # the probe is projected onto the subspace of n
    # fermions on the n dimensional space
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
        i: sum(get_half_filled_basis_vectors(i)) for i in range(1, max_num_qubits + 1)
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

    perms = list(itertools.permutations(half_filled_list))
    perms = [''.join([str(j) for j in this_el]) for this_el in perms]
    perms = list(set(perms))

    basis = [state_from_string(s) for s in perms]
    return basis


def random_superposition_occupation_basis():
    # vacant = np.array([1,0])
    # occupied = np.array([0,1])
    # down = np.kron(occupied, vacant) #|10>
    # up = np.kron(vacant, occupied) #|01>

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
    state = state / np.linalg.norm(state)
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
    print("[fermi_hubbard_half_filled_superposition] num q:", max_num_qubits)
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
