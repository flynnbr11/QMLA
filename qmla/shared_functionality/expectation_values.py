import numpy as np

from scipy import linalg
import qmla.logging

# use_linalg = False
# use_sparse = False

# try:
#     import hamiltonian_exponentiation as h
#     # TODO set to true after testing
#     ham_exp_installed = True

# except BaseException:
#     ham_exp_installed = False

# if (use_linalg):
#     # override and use linalg.expm even if hamiltonian_exponentiation is
#     # installed
#     ham_exp_installed = False


def log_print(
    to_print_list, 
    log_file, 
    log_identifier='ExpectationValue'
):
    qmla.logging.print_to_log(
        to_print_list = to_print_list, 
        log_file = log_file, 
        log_identifier = log_identifier
    )


# Default expectation value calculations

def default_expectation_value(
    ham,
    t,
    state,
    log_file='QMDLog.log',
    log_identifier='Expecation Value'
):
    """
    Default expectation value calculation: | <state.transpose | e^{-iHt} | state> |**2

    Returns the expectation value computed by evolving the input state with
    the provided Hamiltonian operator. NB: In this case, the assumption is that the 
    value measured is 1 and the expectation value corresponds to the probability of
    obtaining 1.

    :param np.array ham: Hamiltonian needed for the time-evolution
    :param float t: Evolution time
    :param np.array state: Initial state to evolve and measure on
    :param str log_file: (optional) path of the log file
    :param str log_identifier: (optional) identifier for the log

    :return: expectation value of the evolved state
    """

    unitary = linalg.expm(-1j * ham * t)
    probe_bra = state.conj().T
    u_psi = np.dot(unitary, state)
    psi_u_psi = np.dot(probe_bra, u_psi)
    expec_val = np.abs(psi_u_psi)**2

    # check that expectation value is reasonable (0 <= EV <= 1)
    ex_val_tol = 1e-9
    if (
        expec_val > (1 + ex_val_tol)
        or
        expec_val < (0 - ex_val_tol)
    ):
        log_print(
            [
                "Expectation value greater than 1 or less than 0: \t",
                expec_val
            ],
            log_file=log_file,
            log_identifier=log_identifier
        )
    return expec_val




# Expecactation value function using Hahn inversion gate:

def hahn_evolution(
    ham,
    t,
    state,
    precision=1e-10,
    log_file=None,
    log_identifier=None
):
    r"""
    Hahn echo evolution and expectation value.

    Returns the expectation value computed by evolving the input state with
    the Hamiltonian corresponding to the Hahn eco evolution. NB: In this case, the assumption is that the 
    value measured is 1 and the expectation value corresponds to the probability of
    obtaining 1.

    :param np.array ham: Hamiltonian needed for the time-evolution
    :param float t: Evolution time
    :param np.array state: Initial state to evolve and measure on
    :param float precision: precision required of the expectation value
        when using custom exponentiation function. TODO deprecated; remove
    :param str log_file: (optional) path of the log file
    :param str log_identifier: (optional) identifier for the log

    :return: expectation value of the evolved state

    """
    
    # NOTE this is always projecting onto |+>
    # okay for experimental data with spins in NV centre
    import numpy as np
    from scipy import linalg
    from qmla.shared_functionality.probe_set_generation import random_probe
    inversion_gate = np.array([
        [0. - 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]]
    )

    # TODO Hahn gate here does not include pi/2 term
    # is it meant to???? as below
    # inversion_gate = inversion_gate * np.pi/2
    even_time_split = False
    if even_time_split:
        # unitary_time_evolution = h.exp_ham(
        #     ham,
        #     t,
        #     precision=precision
        # )
        unitary_time_evolution = qutip.Qobj(-1j * ham * t).expm().full()

        total_evolution = np.dot(
            unitary_time_evolution,
            np.dot(
                inversion_gate,
                unitary_time_evolution
            )
        )
    else:
        # TODO revisit custom exponentiation function and match with Qutip.
        # first_unitary_time_evolution = h.exp_ham(
        #     ham,
        #     t,
        #     precision=precision
        # )
        # second_unitary_time_evolution = h.exp_ham(
        #     ham,
        #     2*t,
        #     precision=precision
        # )

        first_unitary_time_evolution = linalg.expm(-1j * ham * t)
        second_unitary_time_evolution = np.linalg.matrix_power(
            first_unitary_time_evolution, 2
        )
        # first_unitary_time_evolution = qutip.Qobj(-1j*ham*t).expm().full()
        # second_unitary_time_evolution = qutip.Qobj(-1j*ham*2*t).expm().full()
        # first_unitary_time_evolution = h.exp_ham(ham, t)
        # second_unitary_time_evolution = h.exp_ham(ham, 2*t)

        total_evolution = np.dot(
            second_unitary_time_evolution,
            np.dot(inversion_gate,
                   first_unitary_time_evolution
                   )
        )


#    print("total ev:\n", total_evolution)
    ev_state = np.dot(total_evolution, state)

    nm = np.linalg.norm(ev_state)
    if np.abs(1 - nm) > 1e-5:
        print("[hahn] norm ev state:", nm, "\t t=", t)
        raise NameError("Non-unit norm")

    density_matrix = np.kron(ev_state, (ev_state.T).conj())
    density_matrix = np.reshape(density_matrix, [4, 4])
    reduced_matrix = partial_trace_out_second_qubit(
        density_matrix,
        qubits_to_trace=[1]
    )

    plus_state = np.array([1, 1]) / np.sqrt(2)
    # from 1000 counts - Poissonian noise = 1/sqrt(1000) # should be ~0.03
    noise_level = 0.00
    
    random_noise = noise_level * random_probe(1)
    noisy_plus = plus_state + random_noise
    norm_factor = np.linalg.norm(noisy_plus)
    noisy_plus = noisy_plus / norm_factor
    bra = noisy_plus.conj().T

    rho_state = np.dot(reduced_matrix, noisy_plus)
    expect_value = np.abs(np.dot(bra, rho_state))
#    print("Hahn. Time=",t, "\t ex = ", expect_value)
    # print("[Hahn evolution] projecting onto:", repr(bra))
    return 1 - expect_value # because we actually project onto |-> in experiment
#    return expect_value



def make_inversion_gate(num_qubits):
    r"""
    returns the inversion gate for the Hahn eco evolution as numpy array

    :parameter num_qubits: size of the inversion gate required

    :output : inversion gate
    """
    from scipy import linalg
    sigma_z = np.array([[1 + 0.j, 0 + 0.j], [0 + 0.j, -1 + 0.j]])
    hahn_angle = np.pi / 2
    hahn_gate = np.kron(
        hahn_angle * sigma_z,
        np.eye(2**(num_qubits - 1))
    )
    # inversion_gate = qutip.Qobj(-1.0j * hahn_gate).expm().full()
    inversion_gate = linalg.expm(-1j * hahn_gate)

    return inversion_gate


def n_qubit_hahn_evolution(
    ham, t, state,
    second_time_evolution_factor = 2,
    precision=1e-10,
    log_file=None,
    log_identifier=None
):
    r"""
    n qubits time evolution for hahn-echo measurement returning expectation value

    :param ham: Hamiltonian needed for the time-evolution
    :type ham: np.array()
    :param t: Evolution time
    :type t: float()
    :param state: Initial state to evolve and measure on
    :type state: float()
    :param precision: (optional) chosen precision for expectation value, default 1e-10
    :type precision: float()
    :param log_file: (optional) path of the log file for logging errors
    :type log_file: str() 
    :param log_identifier: (optional) identifier for the log
    :type log_identifier: str()    

    :output: expectation value of the evolved state


    """

    # print("n qubit hahn")
    # import qutip
    import numpy as np
    import qmla.database_framework
    from scipy import linalg
    # print("n qubit hahn")
    num_qubits = int(np.log2(np.shape(ham)[0]))

    try:
        from qmla.shared_functionality.hahn_inversion_gates import precomputed_hahn_inversion_gates
        inversion_gate = precomputed_hahn_inversion_gates[num_qubits]
    except BaseException:
        inversion_gate = make_inversion_gate(num_qubits)

    # print("[expectation values] N qubit Hahn evolution. dimension {}".format(np.shape(ham)))
    # print("state:", state)
    # want to evolve for t, then han inversion gate, 
    # then again evolution for (S * t)
    # where S = 2 in standard Hahn evolution, 
    # S = 1 for long time dynamics study
    first_unitary_time_evolution = linalg.expm(-1j * ham * t)
    second_unitary_time_evolution = np.linalg.matrix_power(
        first_unitary_time_evolution,
        second_time_evolution_factor 
    )

    total_evolution = np.dot(
        second_unitary_time_evolution,
        np.dot(
            inversion_gate,
            first_unitary_time_evolution
        )
    )

    ev_state = np.dot(total_evolution, state)
    nm = np.linalg.norm(ev_state)
    if np.abs(1 - nm) > 1e-5:
        print("\n\n[n qubit Hahn]\n norm ev state:",
              nm, "\nt=", t, "\nprobe=", repr(state))
        print("\nev state:\n", repr(ev_state))
        print("\nham:\n", repr(ham))
        print("\nHam element[0,2]:\n", ham[0][2])
        print("\ntotal evolution:\n", repr(total_evolution))
        print("\nfirst unitary:\n", first_unitary_time_evolution)
        print("\nsecond unitary:\n", second_unitary_time_evolution)
        print("\ninversion_gate:\n", inversion_gate)

    density_matrix = np.kron(
        ev_state,
        (ev_state.T).conj()
    )
    dim_hilbert_space = 2**num_qubits
    density_matrix = np.reshape(
        density_matrix,
        [dim_hilbert_space, dim_hilbert_space]
    )

    # qdm = qutip.Qobj(density_matrix, dims=[[2],[2]])
    # reduced_matrix_qutip = qdm.ptrace(0).full()

    # below methods give different results for reduced_matrix
    # reduced_matrix = qdm.ptrace(0).full()

    to_trace = list(range(num_qubits - 1))
    reduced_matrix = partial_trace(
        density_matrix,
        to_trace
    )

    density_mtx_initial_state = np.kron(
        state,
        state.conj()
    )

    density_mtx_initial_state = np.reshape(
        density_mtx_initial_state,
        [dim_hilbert_space, dim_hilbert_space]
    )
    reduced_density_mtx_initial_state = partial_trace(
        density_mtx_initial_state,
        to_trace
    )
    projection_onto_initial_den_mtx = np.dot(
        reduced_density_mtx_initial_state,
        reduced_matrix
    )
    expect_value = np.abs(
        np.trace(
            projection_onto_initial_den_mtx
        )
    )

    # expect_value is projection onto |+>
    # for this case Pr(0) refers to projection onto |->
    # so return 1 - expect_value

    return 1 - expect_value
    # return expect_value



def partial_trace(
    mat,
    trace_systems,
    dimensions=None,
    reverse=True
):
    """
    Returns the partial trace of mat over subsystems of multi-partite matrix.
    

    loghish description:
    This function has been taken from https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/tools/qi/qi.py#L177
    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.
    
    :param mat: a matrix NxN.
    :type mat: np.array()
    :param trace_systems: a list of subsystems (starting from 0) to trace over.
    :type trace_system: list(int())
    :param dimensions : (optional) a list of the dimensions of the subsystems. Default=None.
    If this is not set it will assume all subsystems are qubits.
    :type dimensions: list(int())
    :param reverse: (optional) If True reverses the ordering of systems in operator. Default=True.
    If True system-0 is the right most system in tensor product.
    If False system-0 is the left most system in tensor product.
    :type reverse: bool 

    :output:  A density matrix with the appropriate subsystems traced over as ndarray.
    """
    if dimensions is None:  # compute dims if not specified
        length = np.shape(mat)[0]
        num_qubits = int(np.log2(length))
        dimensions = [2 for _ in range(num_qubits)]
        if length != 2 ** num_qubits:
            raise Exception("Input is not a multi-qubit state, "
                            "specify input state dims")
    else:
        dimensions = list(dimensions)

    trace_systems = sorted(trace_systems, reverse=True)
    for j in trace_systems:
        # Partition subsystem dimensions
        dimension_trace = int(dimensions[j])  # traced out system
        if reverse:
            left_dimensions = dimensions[j + 1:]
            right_dimensions = dimensions[:j]
            dimensions = right_dimensions + left_dimensions
        else:
            left_dimensions = dimensions[:j]
            right_dimensions = dimensions[j + 1:]
            dimensions = left_dimensions + right_dimensions
        # Contract remaining dimensions
        dimension_left = int(np.prod(left_dimensions))
        dimension_right = int(np.prod(right_dimensions))

        # Reshape input array into tri-partite system with system to be
        # traced as the middle index
        mat = mat.reshape([dimension_left, dimension_trace, dimension_right,
                           dimension_left, dimension_trace, dimension_right])
        # trace out the middle system and reshape back to a matrix
        mat = mat.trace(axis1=1, axis2=4).reshape(
            dimension_left * dimension_right,
            dimension_left * dimension_right)
    return mat


def partial_trace_out_second_qubit(
    global_rho,
    qubits_to_trace=[1]
):

    """
    2 qubit version of the partial trace function.

    :parameter global_rho: original full density matrix
    :type global_rho: np.array()

    :parameter qubits_to_trace: list containing the qubit index that we want to trace out from the full system
    this can be [0] or [1]. Default=[1]
    :type global_rho: list(int())

    :output: matrix containing the traced out state as np.array()
    """

    len_input_state = len(global_rho)
    input_num_qubits = int(np.log2(len_input_state))

    qubits_to_trace.reverse()

    num_qubits_to_trace = len(qubits_to_trace)
    output_matrix = []  # initialise the output reduced matrix

    for i in range(num_qubits_to_trace):
        k = qubits_to_trace[i]

        if k == num_qubits_to_trace:
            for p in range(0, int(len_input_state), 2):

                # pick odd positions in the original matrix
                odd_positions = global_rho[p][::2]
                # pick even positions in the original matrix
                even_positions = global_rho[p + 1][1::2]

                output_matrix.append(
                    odd_positions + even_positions
                )
    output_matrix = np.array(output_matrix)
    return output_matrix


# for easy access to plus states to plot against
def n_qubit_plus_state(num_qubits):
    """
    Returns the num_qubits pure quantum states where each of the qubits is in plus state as a np.array()

    :parameter num_qubits: number of qubits 
    :type num_qubits: int()

    :output: pure quantum state of num_qubits in plus state.
    """

    one_qubit_plus = (1 / np.sqrt(2) + 0j) * np.array([1, 1])
    plus_n = one_qubit_plus
    for i in range(num_qubits - 1):
        plus_n = np.kron(plus_n, one_qubit_plus)
    return plus_n
