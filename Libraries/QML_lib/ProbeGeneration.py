import numpy as np
#import qutip
# import ExpectationValues
import DataBase


## Simluated Probes: random
def separable_probe_dict(
    max_num_qubits, 
    num_probes,
    **kwargs
):
    separable_probes = {}
    for i in range(num_probes):
        separable_probes[i,0] = random_probe(1)
        for j in range(1, 1+max_num_qubits):
            if j==1:
                separable_probes[i,j] = separable_probes[i,0]
            else: 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        random_probe(1), 
                        axes=0
                    ).flatten(order='c')
                )
            norm = np.linalg.norm(separable_probes[i,j])
            while (
                np.abs( norm -1) >
                1e-13

            ):
                print(
                    "non-unit norm: ", 
                    norm
                )
                # keep replacing until a unit-norm 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1], 
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
                norm = np.linalg.norm(separable_probes[i,j])
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
        complex_vectors.append(real[i] + 1j*imaginary[i])

    a=np.array(complex_vectors)
    norm_factor = np.linalg.norm(a)
    probe = complex_vectors/norm_factor
    # if np.isclose(1.0, np.linalg.norm(probe), atol=1e-14) is False:
    #     print("Probe not normalised. Norm factor=", np.linalg.norm(probe)-1)
    #     return random_probe(num_qubits)
    while (
        np.abs( np.linalg.norm(probe) ) - 1
        > 
        1e-14 
    ):
        print("generating new random probe..")
        probe = random_probe(num_qubits)

    # print("random probe generated with norm:", np.linalg.norm(probe))
    return probe



## Specific experimental probes

def NV_centre_ising_probes_plus(
    max_num_qubits=2, 
    num_probes=40,
    noise_level=0.03, #from 1000 counts - Poissonian noise = 1/sqrt(1000)
    minimum_tolerable_noise = 1e-6,
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
    if minimum_tolerable_noise  > noise_level:
        noise_level = minimum_tolerable_noise
        # print("using minimum_tolerable_noise")
    print("noise level in plus probe:", noise_level)
    plus_state = np.array([1+0j, 1])/np.sqrt(2)
    random_noise = noise_level * random_probe(1)    
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus/norm_factor
    print("\n\t noisy plus:", noisy_plus )
    # print("\n\t has type:", type(noisy_plus))
    
    separable_probes = {}
    for i in range(num_probes):
#        separable_probes[i,0] = plus_state
        separable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                separable_probes[i,j] = separable_probes[i,0]
            else: 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        # noisy_plus,
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0, 
                    np.linalg.norm(separable_probes[i,j]), 
                    atol=1e-14
                ) is  False
            ):
                print("non-unit norm: ", 
                    np.linalg.norm(separable_probes[i,j])
                )
                # keep replacing until a unit-norm 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
    return separable_probes
    
def restore_dec_13_probe_generation(
    max_num_qubits=2, 
    num_probes=40,
    noise_level=0.03, #from 1000 counts - Poissonian noise = 1/sqrt(1000)
    minimum_tolerable_noise = 1e-7,
    # minimum_tolerable_noise needed
    # or else run the risk of having 
    # exact eigenstate and no learning occurs, and crashes. 
    **kwargs    
):
    """
    Returns a dict of separable probes where the first qubit always acts on 
    a plus state. 
    """
    print(
        "[restore_dec_13_probe_generation] min tol noise:", 
        minimum_tolerable_noise, 
        "noise level:", noise_level
    )

    if minimum_tolerable_noise  > noise_level:
        noise_level = minimum_tolerable_noise
        # print("using minimum_tolerable_noise")
    plus_state = np.array([1+0j, 1])/np.sqrt(2)
    random_noise = noise_level * random_probe(1)    
    noisy_plus = plus_state + random_noise

    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus/norm_factor
    # print("\n\t noisy plus:", noisy_plus )
    # print("\n\t has type:", type(noisy_plus))
    
    separable_probes = {}
    for i in range(num_probes):
#        separable_probes[i,0] = plus_state
        separable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                separable_probes[i,j] = separable_probes[i,0]
            else: 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        noisy_plus, 
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0, 
                    np.linalg.norm(separable_probes[i,j]), 
                    atol=1e-14
                ) is  False
            ):
                print("non-unit norm: ", 
                    np.linalg.norm(separable_probes[i,j])
                )
                # keep replacing until a unit-norm 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
    return separable_probes
    


def experimental_NVcentre_ising_probes(
    max_num_qubits=2, 
    num_probes=40,
    **kwargs
):
    """
    Returns a dict of separable probes where the first qubit always acts on 
    a plus state. 
    """
    plus_state = np.array([1, 1])/np.sqrt(2)
    noise_level = 0.03 # from 1000 counts - Poissonian noise = 1/sqrt(1000)
    random_noise = noise_level * random_probe(1)    
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus/norm_factor
    
    separable_probes = {}
    for i in range(num_probes):
#        separable_probes[i,0] = plus_state
        separable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                separable_probes[i,j] = separable_probes[i,0]
            else: 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        random_probe(1), 
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0, 
                    np.linalg.norm(separable_probes[i,j]), 
                    atol=1e-14
                ) is  False
            ):
                print("non-unit norm: ", np.linalg.norm(separable_probes[i,j]))
                # keep replacing until a unit-norm 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1], 
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
    return separable_probes
  
def plus_plus_with_phase_difference(
    max_num_qubits=2, 
    num_probes=40,
    noise_level=0.03, #from 1000 counts - Poissonian noise = 1/sqrt(1000)
    minimum_tolerable_noise = 1e-6,
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
    if minimum_tolerable_noise  > noise_level:
        noise_level = minimum_tolerable_noise
    plus_state = np.array([1+0j, 1])/np.sqrt(2)
    random_noise = noise_level * random_probe(1)    
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus/norm_factor
   
    separable_probes = {}
    for i in range(num_probes):
        separable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                separable_probes[i,j] = separable_probes[i,0]
            else: 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        # noisy_plus,
                        random_phase_plus(noise_level=noise_level),
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0, 
                    np.linalg.norm(separable_probes[i,j]), 
                    atol=1e-14
                ) is  False
            ):
                print("non-unit norm: ", 
                    np.linalg.norm(separable_probes[i,j])
                )
                # keep replacing until a unit-norm 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        random_phase_plus(noise_level=noise_level),
                        axes=0
                    ).flatten(order='c')
                )
    return separable_probes

def random_phase_plus(
    noise_level = 1e-5
):
    import random
    random_phase = random.uniform(0, np.pi)
    rand_phase_plus = np.array(
        [
            1.0+0.j, 
            np.exp(1.0j*random_phase)
        ]
    )/np.sqrt(2)
    
    noisy_state = noise_level*random_probe(1)
    rand_phase_plus += noisy_state
    norm = np.linalg.norm(rand_phase_plus)
    rand_phase_plus = rand_phase_plus/norm
    return rand_phase_plus


# Hubbard encoding probes


def vector_from_fermion_state_description(state):

    occupied = np.array([0,1])
    vacant = np.array([1,0])

    vector = 1

    for i in range(1, 1+state['num_sites']):
        try:
            occupation = state['occupations'][i]
        except:
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

def fermi_hubbard_single_varied_spin_n_sites(
    max_num_qubits, 
    spin_type = 'down',
    num_probes=10,
    **kwargs
):
    probe_dict = {}

    down = vector_from_fermion_state_description(
        {
            'num_sites' : 1, 
            'occupations' : {
                1 : ['down']
            }
        }
    )
    up = vector_from_fermion_state_description(
        {
            'num_sites' : 1, 
            'occupations' : {
                1 : ['up']
            }
        }
    )
    vacuum = vector_from_fermion_state_description(
        {
            'num_sites' : 1, 
            'occupations' : {
            }
        }
    )


    spins = [down, up]
    for i in range(num_probes):
        probe_dict[i,0] = spins[i%len(spins)]
        
        for j in range(1, 1+max_num_qubits):
            # here max_num_qubits = num sites; need 2 qubits to encode each site
            probe_id = (i,j)
            if j == 1:
                probe = probe_dict[(i,0)]
            else:
                probe = np.kron(
                    probe_dict[(i, j-1)],
                    vacuum
                )
            probe_dict[(i,j)] = probe
            # site_dim = int(np.log2(np.shape(probe)[0])) # TODO fix probe dimensionality/dependence
            # probe_dict[(i,site_dim)] = probe
    
    return probe_dict
    
def random_superposition_half_filled_occupation_basis():
    # vacant = np.array([1,0])
    # occupied = np.array([0,1])
    # down = np.kron(occupied, vacant) #|10>
    # up = np.kron(vacant, occupied) #|01>
    
    down = np.array([0, 0, 1, 0])
    up = np.array([0, 1, 0, 0])

    alpha = np.random.randn() + 1j*np.random.randn()
    beta = np.random.randn() + 1j*np.random.randn()
    
    state = ( alpha * down ) + ( beta * up )
    state = state/np.linalg.norm(state)
    return state

def fermi_hubbard_separable_probes_half_filled(
    max_num_qubits, 
    num_probes,
    **kwargs
):
    separable_probes = {}
    for i in range(num_probes):
        separable_probes[i,0] = random_superposition_half_filled_occupation_basis()
        for j in range(1, 1+max_num_qubits):
            if j==1:
                separable_probes[i,j] = separable_probes[i,0]
            else: 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1],
                        random_superposition_half_filled_occupation_basis(), 
                        axes=0
                    ).flatten(order='c')
                )
            norm = np.linalg.norm(separable_probes[i,j])
            while (
                np.abs( norm -1) >
                1e-13

            ):
                print(
                    "non-unit norm: ", 
                    norm
                )
                # keep replacing until a unit-norm 
                separable_probes[i,j] = (
                    np.tensordot(
                        separable_probes[i,j-1], 
                        random_superposition_half_filled_occupation_basis(),
                        axes=0
                    ).flatten(order='c')
                )
                norm = np.linalg.norm(separable_probes[i,j])
            # print("unit norm:", np.abs(1-norm) )

    return separable_probes



def fermi_hubbard_single_spin_n_sites(
    max_num_qubits, 
    spin_type = 'up',
    num_probes=10,
    **kwargs
):
    probe_dict = {}

    initial_probe = vector_from_fermion_state_description(
        {
            'num_sites' : 1, 
            'occupations' : {
                1 : [spin_type]
            }
        }
    )
    vacuum = vector_from_fermion_state_description(
        {
            'num_sites' : 1, 
            'occupations' : {
            }
        }
    )
    for i in range(num_probes):
        probe_dict[i,0] = initial_probe
        
        for j in range(1, 1+max_num_qubits):
            # here max_num_qubits = num sites; need 2 qubits to encode each site
            probe_id = (i,j)
            if j == 1:
                probe = probe_dict[(i,0)]
            else:
                probe = np.kron(
                    probe_dict[(i, j-1)],
                    vacuum
                )
            probe_dict[(i,j)] = probe
            # site_dim = int(np.log2(np.shape(probe)[0])) # TODO fix probe dimensionality/dependence
            # probe_dict[(i,site_dim)] = probe
    
    return probe_dict


def get_binary_string(num, length=8):
    return format(num, '0{}b'.format(length ))
    
def fermi_hubbard_half_filled_from_binary(binary_string):
    """ 
    e.g. 101 -> up down up -> in occupation basis: |01 10 01> 
    """
    vacant = np.array([1,0]) # |0>
    occupied = np.array([0,1]) # |1>
    
    up = np.kron(vacant, occupied)
    down = np.kron(occupied, vacant)
    
    state = 1
    for b in binary_string:
        if int(b, 2) == 1:
            state = np.kron(state, up)
        else:
            state = np.kron(state, down)
    
    return state

def fermi_hubbard_half_filled_superposition(
    max_num_qubits, 
    **kwargs
):
    print("[fermi_hubbard_half_filled_superposition] num q:", max_num_qubits)
    num_sites = max_num_qubits
    probe_dict = {}
    for N in range(1, 1+num_sites):

        state = None
        for i in range(1, N+1):
            for spin_type in ['up', 'down']:

                new_state = vector_from_fermion_state_description(
                    {
                        'num_sites' : N, 
                        'occupations' : {
                            i : [spin_type]
                        }
                    }
                )

                if state is None:
                    state = new_state
                else:
                    state += new_state

        state = state/np.linalg.norm(state)

        probe_dict[(1, N)] = state
    print("[fermi_hubbard_half_filled_superposition] keys:", probe_dict.keys())
    return probe_dict

def fermi_hubbard_half_filled_pure_states(
    max_num_qubits, 
    **kwargs  
):
    num_sites = max_num_qubits
    probe_dict = {}
    for i in range(2**num_sites):
        for j in range(1, num_sites+1):
            binary_rep = get_binary_string(i, length = num_sites)[-j:]
            probe_dict[(i,j)] = fermi_hubbard_half_filled_from_binary(binary_rep)
    return probe_dict
   

def fermi_hubbard_even_superposition(
    max_num_qubits, 
    spin_type = 'down',
    num_probes=10,
    **kwargs
):
    probe_dict = {}

    superposition = np.array([0, 1, 1, 0])/np.sqrt(2) 
    
    for i in range(num_probes):
        probe_dict[i,0] = superposition
        
        for j in range(1, 1+max_num_qubits):
            # here max_num_qubits = num sites; need 2 qubits to encode each site
            probe = np.kron(
                probe_dict[(i, j-1)],
                superposition
            )
            probe_dict[(i,j)] = probe
            # site_dim = int(np.log2(np.shape(probe)[0])) # TODO fix probe dimensionality/dependence
            # probe_dict[(i,site_dim)] = probe
    
    return probe_dict
    


## TODO fermi hubbard below this using incorrect basis encoding (not consisten with Jordan Wigner)
# TODO remove when safe

def fermi_hubbard_encoding_fixed_spin(
    max_num_qubits,
    spin_type='up', 
    num_probes=10, 
    **kwargs
):
    # basis: (Vacuum, spin-up, spin-down, double-occupancy)
    # represent Fermions in a site
    # returns |spin_type> \otimes \Vac>^{n}, n+1 sites
    basis_vectors = {
        'vac' : np.array([1,0,0,0]),
        'down' : np.array([0,1,0,0]),
        'up' : np.array([0,0,1,0]),
        'double' : np.array([0,0,0,1])
    }

    probe_dict = {}
    
    for i in range(num_probes):
        probe_dict[i,0] = basis_vectors[spin_type] 
        for j in range(1, max_num_qubits):
            # here max_num_qubits = num sites; need 2 qubits to encode each site
            probe_id = (i,j)
            if j == 1:
                probe = probe_dict[(i,0)]
            else:
                probe = np.kron(
                    probe_dict[(i, j-1)],
                    basis_vectors['vac']
                )
            probe_dict[(i,j)] = probe
            # site_dim = int(np.log2(np.shape(probe)[0])) # TODO fix probe dimensionality/dependence
            # probe_dict[(i,site_dim)] = probe
    
    return probe_dict


def fermi_hubbard_encoding_pure_up_down_cycle(
    max_num_qubits,
    bases=['up', 'down'],
    num_probes=10, 
    **kwargs
):
    # basis: (Vacuum, spin-up, spin-down, double-occupancy)
    # represent Fermions in a site
    # returns |spin_type>^N, N sites
    basis_vectors = {
        'vac' : np.array([1,0,0,0]),
        'down' : np.array([0,1,0,0]),
        'up' : np.array([0,0,1,0]),
        'double' : np.array([0,0,0,1])
    }
    superposition = np.array([0,1,1,0])*(1/np.sqrt(2))
    probe_dict = {}
    
    for i in range(num_probes):
        if i%2 == 0:
            pure_basis = basis_vectors['up']
        else:
            pure_basis = basis_vectors['down']
        probe_dict[i,0] = pure_basis
        print("i={} \n pure basis = {}".format(i, pure_basis))
        for j in range(1, max_num_qubits):
            # here max_num_qubits = num sites; need 2 qubits to encode each site
            probe_id = (i,j)
            if j == 1:
                probe = probe_dict[(i,0)]
            else:
                probe = np.kron(
                    probe_dict[(i, j-1)],
                    basis_vectors['vac']
                )
            probe_dict[(i,j)] = probe
            # site_dim = int(np.log2(np.shape(probe)[0])) # TODO fix probe dimensionality/dependence
            # probe_dict[(i,site_dim)] = probe
    
    return probe_dict



def fermi_hubbard_encoding_even_superposition_up_down(
    max_num_qubits,
    num_probes=10, 
    **kwargs
):
    # basis: (Vacuum, spin-up, spin-down, double-occupancy)
    # represent Fermions in a site
    # returns |spin_type>^N, N sites
    basis_vectors = {
        'vac' : np.array([1,0,0,0]),
        'down' : np.array([0,1,0,0]),
        'up' : np.array([0,0,1,0]),
        'double' : np.array([0,0,0,1])
    }
    superposition = np.array([0,1,1,0])*(1/np.sqrt(2))
    probe_dict = {}
    
    for i in range(num_probes):
        probe_dict[i,0] = superposition
        for j in range(1, max_num_qubits):
            # here max_num_qubits = num sites; need 2 qubits to encode each site
            probe_id = (i,j)
            if j == 1:
                probe = probe_dict[(i,0)]
            else:
                probe = np.kron(
                    probe_dict[(i, j-1)],
                    superposition
                )
            probe_dict[(i,j)] = probe
            # site_dim = int(np.log2(np.shape(probe)[0])) # TODO fix probe dimensionality/dependence
            # probe_dict[(i,site_dim)] = probe
    
    return probe_dict


def fermi_hubbard_encoding_half_filled_random_probes(
    max_num_qubits, 
    num_probes=10, 
    **kwargs
):
    # basis: (Vacuum, spin-up, spin-down, double-occupancy)
    # represent Fermions in a site
    basis_vectors = {
        'vac' : np.array([1,0,0,0]),
        'down' : np.array([0,1,0,0]),
        'up' : np.array([0,0,1,0]),
        'double' : np.array([0,0,0,1])
    }

    probe_dict = {}
    
    for i in range(num_probes):
        probe_dict[i,0] = random_half_filled_fermi_site(
            basis_vectors = basis_vectors
        )
        for j in range(1, max_num_qubits):
            # here max_num_qubits = num sites; need 2 qubits to encode each site
            probe_id = (i,j)
            if j == 1:
                probe = probe_dict[(i,0)]
            else:
                probe = np.kron(
                    probe_dict[(i, j-1)],
                    random_half_filled_fermi_site(basis_vectors = basis_vectors)
                )
            probe_dict[(i,j)] = probe
            # site_dim = int(np.log2(np.shape(probe)[0])) # TODO fix probe dimensionality/dependence
            # probe_dict[(i,site_dim)] = probe
    
    return probe_dict


def random_half_filled_fermi_site(
    basis_vectors = {
        'vac' : np.array([1,0,0,0]),
        'down' : np.array([0,1,0,0]),
        'up' : np.array([0,0,1,0]),
        'double' : np.array([0,0,0,1])
    }    
):    
    """
    half-filled i.e. single spin in superposition of down + up
    encoded in basis ( vacuum, down, up, up-down)
    (vacuum and up-down bases not used in half filling)
    """
    vec = (
        np.random.rand() * basis_vectors['down']
        + np.random.rand() * basis_vectors['up']
    )
    norm= np.linalg.norm(vec)
    vec = vec/norm    
    return vec



# General purpose probe dictionaries
def n_qubit_plus_state(num_qubits):
    one_qubit_plus = (1/np.sqrt(2) + 0j) * np.array([1,1])
    plus_n = one_qubit_plus
    for i in range(num_qubits-1):
        plus_n = np.kron(plus_n, one_qubit_plus)
    return plus_n


def plus_probes_dict(
    max_num_qubits, 
    noise_level=0.0, #from 1000 counts - Poissonian noise = 1/sqrt(1000)
    minimum_tolerable_noise = 0,
    **kwargs
):
    print("[Plus probe dict] Locals:", locals())
    num_probes = kwargs['num_probes']

    print(
        "[plus_probes_dict] min tol noise:", 
        minimum_tolerable_noise, 
        "noise level:", noise_level,
        "num probes:", num_probes
    )

    if minimum_tolerable_noise  > noise_level:
        noise_level = minimum_tolerable_noise
        # print("using minimum_tolerable_noise")
    print("noise level in plus probe:", noise_level)
    probe_dict = {}
    # noise_level = 0.03
    for j in range(num_probes):
        for i in range(1,1+max_num_qubits):
            # dict key is tuple of form (0,i) for consistency with other probe dict generation functions. 
            new_probe =  n_qubit_plus_state(i)
            noisy_state = random_probe(i) * noise_level
            noisy_probe = new_probe + noisy_state
            norm = np.linalg.norm(noisy_probe)
            noisy_probe = noisy_probe/norm
            probe_dict[(j,i)] = noisy_probe
    return probe_dict 

def zero_state_probes(max_num_qubits=9, **kwargs):
    zero = np.array([1+0j, 0])
    num_probes = kwargs['num_probes']
    probes = {}
    
    for q in range(1, 1+max_num_qubits):
        for j in range(num_probes):
            state = zero
            for i in range(q-1):
                state = np.tensordot(state, zero, axes=0).flatten('c')
            probes[(j, q)] = state

    return probes

def ideal_probe_dict(
    true_operator, 
    max_num_qubits,
    **kwargs
):
    probe_dict = {}
    true_dim = DataBase.get_num_qubits(true_operator)
    ideal_probe = DataBase.ideal_probe(true_operator)
    qt_probe = qutip.Qobj(ideal_probe)
    density_mtx = qutip.ket2dm(qt_probe)
    
    for i in range(1, 1+max_num_qubits):
        dict_key = (0, i) # for consistency with other probe dict functions
        if i==true_dim:
            probe_dict[dict_key] = ideal_probe
        elif i < true_dim:
            # TODO trace out from density mtx and retrieve closest pure state for here. 
            probe_dict[dict_key] = random_probe(i)
        else:
            # TODO replace this with traced out probe for 
            # case where sim num qubits < true num qubits
            rand = random_probe(1)
            new_probe = np.kron(
                probe_dict[(0, i-1)], 
                rand,
            )
            probe_dict[dict_key] = new_probe
    return probe_dict

def PT_Effective_Hamiltonian_probe_dict(
    **kwargs
):
    # for development
    # TODO make this more robust
    import pickle
    try:
        probes = pickle.load(
            open(
                "/home/bf16951/Dropbox/QML_share_stateofart/QMD/Launch/Data/test_PT_probedict.p", 
                'rb'
            )
        )
    except:
        probes = pickle.load(
            open(
                "/panfs/panasas01/phys/bf16951/QMD/Launch/Data/test_PT_probedict.p", 
                'rb'
            )
        )

    return probes
    


