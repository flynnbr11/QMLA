import numpy as np

## Simluated Probes: random
def separable_probe_dict(
    max_num_qubits, 
    num_probes,
    **kwargs
):
    seperable_probes = {}
    for i in range(num_probes):
        seperable_probes[i,0] = random_probe(1)
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = (np.tensordot(seperable_probes[i,j-1],
                    random_probe(1), axes=0).flatten(order='c')
                )
            while (np.isclose(1.0, np.linalg.norm(seperable_probes[i,j]), 
                atol=1e-14) is  False
            ):
                print("non-unit norm: ", np.linalg.norm(seperable_probes[i,j]))
                # keep replacing until a unit-norm 
                seperable_probes[i,j] = (
                    np.tensordot(seperable_probes[i,j-1], random_probe(1),
                    axes=0).flatten(order='c')
                )
    return seperable_probes

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
    if np.isclose(1.0, np.linalg.norm(probe), atol=1e-14) is False:
        print("Probe not normalised. Norm factor=", np.linalg.norm(probe)-1)
        return random_probe(num_qubits)

    return probe



## Specific experimental probes

def NV_centre_ising_probes_plus(
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
    
    if minimum_tolerable_noise  > noise_level:
        noise_level = minimum_tolerable_noise
        print("using minimum_tolerable_noise")
    plus_state = np.array([1+0j, 1])/np.sqrt(2)
    random_noise = noise_level * random_probe(1)    
    noisy_plus = plus_state + random_noise

    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus/norm_factor
    print("\n\t noisy plus:", noisy_plus )
    print("\n\t has type:", type(noisy_plus))
    
    seperable_probes = {}
    for i in range(num_probes):
#        seperable_probes[i,0] = plus_state
        seperable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = (
                    np.tensordot(
                        seperable_probes[i,j-1],
                        noisy_plus, 
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0, 
                    np.linalg.norm(seperable_probes[i,j]), 
                    atol=1e-14
                ) is  False
            ):
                print("non-unit norm: ", 
                    np.linalg.norm(seperable_probes[i,j])
                )
                # keep replacing until a unit-norm 
                seperable_probes[i,j] = (
                    np.tensordot(
                        seperable_probes[i,j-1],
                        random_probe(1),
                        axes=0
                    ).flatten(order='c')
                )
    return seperable_probes
    
    


def experimental_NVcentre_ising_probes(
    max_num_qubits=2, 
    num_probes=40
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
    
    seperable_probes = {}
    for i in range(num_probes):
#        seperable_probes[i,0] = plus_state
        seperable_probes[i,0] = noisy_plus
        for j in range(1, 1+max_num_qubits):
            if j==1:
                seperable_probes[i,j] = seperable_probes[i,0]
            else: 
                seperable_probes[i,j] = (
                    np.tensordot(
                        seperable_probes[i,j-1],
                        random_probe(1), 
                        axes=0
                    ).flatten(order='c')
                )
            while (
                np.isclose(
                    1.0, 
                    np.linalg.norm(seperable_probes[i,j]), 
                    atol=1e-14
                ) is  False
            ):
                print("non-unit norm: ", np.linalg.norm(seperable_probes[i,j]))
                # keep replacing until a unit-norm 
                seperable_probes[i,j] = (
                    np.tensordot(seperable_probes[i,j-1], random_probe(1),
                    axes=0).flatten(order='c')
                )
    return seperable_probes
  