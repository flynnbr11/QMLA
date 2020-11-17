import qmla
import qmla.construct_models

##########
# Section: Generic
##########

def basic_latex_name(
    name,
    **kwargs
):
    return "${}$".format(name)

def pauli_set_latex_name(
    name,
    **kwargs
):
    core_operators = list(sorted(qmla.construct_models.core_operator_dict.keys()))
    num_sites = qmla.construct_models.get_num_qubits(name)
    separate_terms = name.split('+')

    site_connections = {}

    transverse_axis = None
    for term in separate_terms:
        components = term.split('_')
        if 'pauliSet' in components:
            components.remove('pauliSet')

            for l in components:
                if l[0] == 'd':
                    dim = int(l.replace('d', ''))
                elif l[0] in core_operators:
                    operators = l.split('J')
                else:
                    sites = l.split('J')
            sites = tuple([int(a) for a in sites])
            # assumes like-like pauli terms like xx, yy, zz
            op = operators[0]
            try:
                site_connections[sites].append(op)
            except:
                site_connections[sites] = [op]
        elif 'transverse' in components:
            components.remove('transverse')
            for l in components:
                if l[0] == 'd':
                    transverse_dim = int(l.replace('d', ''))
                elif l in core_operators:
                    transverse_axis = l

    ordered_connections = list(sorted(site_connections.keys()))
    latex_term = ""

    for c in ordered_connections:
        if len(site_connections[c]) > 0:
            this_term = r"\sigma_{"
            this_term += str(c)
            this_term += "}"
            this_term += "^{"
            for t in site_connections[c]:
                this_term += "{}".format(t)
            this_term += "}"
            latex_term += this_term
    if transverse_axis is not None:
        latex_term += 'T^{}_{}'.format(transverse_axis, transverse_dim)
    latex_term = "${}$".format(latex_term)
    return latex_term

##########
# Section: Specific to an exploration strategy
##########

def nv_centre_SAT(
    name, 
    **kwargs
):
    if name == 'x' or name == 'y' or name == 'z':
        return '$' + name + '$'

    num_qubits = qmla.construct_models.get_num_qubits(name)
    # terms = name.split('PP')
    terms = name.split('+')
    rotations = ['xTi', 'yTi', 'zTi']
    hyperfine = ['xTx', 'yTy', 'zTz']
    transverse = ['xTy', 'xTz', 'yTz', 'yTx', 'zTx', 'zTy']

    present_r = []
    present_hf = []
    present_t = []

    for t in terms:
        if t in rotations:
            present_r.append(t[0])
        elif t in hyperfine:
            present_hf.append(t[0])
        elif t in transverse:
            string = t[0] + t[-1]
            present_t.append(string)
        # else:
        #     print("Term",t,"doesn't belong to rotations, Hartree-Fock or transverse.")
        #     print("Given name:", name)
    present_r.sort()
    present_hf.sort()
    present_t.sort()

    r_terms = ','.join(present_r)
    hf_terms = ','.join(present_hf)
    t_terms = ','.join(present_t)

    latex_term = ''
    if len(present_r) > 0:
        latex_term += r'\hat{S}_{' + r_terms + '}'
    if len(present_hf) > 0:
        latex_term += r'\hat{A}_{' + hf_terms + '}'
    if len(present_t) > 0:
        latex_term += r'\hat{T}_{' + t_terms + '}'

    final_term = '$' + latex_term + '$'
    if final_term != '$$':
        return final_term

    else:
        plus_string = ''
        for i in range(num_qubits):
            plus_string += 'P'
        individual_terms = name.split(plus_string)
        individual_terms = sorted(individual_terms)

        latex_term = '+'.join(individual_terms)
        final_term = '$' + latex_term + '$'
        return final_term

def nv_spin_interaction(
    name,
    **kwargs
):
    term = name
    num_qubits = qmla.construct_models.get_num_qubits(term)
    t_str = 'T' * (num_qubits - 1)
    p_str = 'P' * num_qubits
    separate_terms = term.split(p_str)

    spin_terms = []
    interaction_terms = []

    for t in separate_terms:
        components = t.split('_')
        components.remove('nv')
        components.remove(str('d' + str(num_qubits)))
        if 'spin' in components:
            components.remove('spin')
            spin_terms.append(components[0])
        elif 'interaction' in components:
            components.remove('interaction')
            interaction_terms.append(components[0])

    latex_name = '('
    if len(spin_terms) > 0:
        latex_name += 'S_{'
        for s in spin_terms:
            latex_name += str(s)
        latex_name += '}'
    if len(interaction_terms) > 0:
        latex_name += 'I_{'
        for s in interaction_terms:
            latex_name += str(s)
        latex_name += '}'

    latex_name += str(
        r')^{\otimes'
        + str(num_qubits)
        + '}'
    )

    return '$' + latex_name + '$'



def grouped_pauli_terms(
    name,
    **kwargs
):
    separate_terms = name.split('+')
    all_connections = []
    latex_term = ""
    connections_terms = {}
    for term in separate_terms:
        components = term.split('_')
        try:
            components.remove('pauliLikewise')
        except:
            print("Couldn't remove pauliLikewise from", name)
        this_term_connections = []
        for l in components:
            if l[0] == 'd':
                dim = int(l.replace('d', ''))
            elif l[0] == 'l':
                operator = str(l.replace('l', ''))
            else:
                sites = l.split('J')
                this_term_connections.append(sites)
        for s in this_term_connections:
            con = "({},{})".format(s[0], s[1])
            try:
                connections_terms[con].append(operator)
            except:
                connections_terms[con] = [operator]

        latex_term = ""
        for c in list(sorted(connections_terms.keys())):
            connection_string = str(
                "\sigma_{"
                + str(c)
                + "}^{"
                + str(",".join(connections_terms[c]))
                + "}"
            )
            latex_term += connection_string

    return "${}$".format(latex_term)


def lattice_set_grouped_pauli(name, **kwargs):
    separate_terms = name.split('+')
    latex_term = ""
    latex_terms = {}
    for term in separate_terms:
        components = term.split('_')
        try:
            components.remove('pauliLikewise')
        except:
            print("Couldn't remove pauliLikewise from", name)
        this_term_connections = []
        for l in components:
            if l[0] == 'd':
                dim = int(l.replace('d', ''))
            elif l[0] == 'l':
                operator = str(l.replace('l', ''))
            else:
                n_sites = len(l.split('J'))
                sites = l.replace('J', ',')
                # if n_sites > 1: sites = "(" + str(sites) + ")"
                this_term_connections.append(sites)

        # limits for sum
        lower_limit = str(
            "i \in "
            +",".join(this_term_connections)
        )
        operator_string = str("\sigma_{ i }^{" + str(operator) + "}")
        if n_sites == 1: 
            sites_not_present = list(
                set([int(i) for i in this_term_connections]) 
                - set(range(1, dim+1))
            )
            if len(sites_not_present) == 0:
                lower_limit = "i=1"
        elif n_sites == 2:
            nns = [(str(n), str(n+1) ) for n in range(1, dim)]
            nns = [','.join(list(nn)) for nn in nns]
            nns = set(nns)
            sites_not_present = list(
                set(this_term_connections)
                - nns
            )
            if len(sites_not_present) == 0:
                lower_limit = "i"
                operator_string = str(
                    "\hat{\sigma}_{i}^{" + str(operator) + "}"
                    + "\hat{\sigma}_{i+1}^{" + str(operator) + "}"
                )
            else: 
                this_term_connections = [
                    "({})".format(c) for c in this_term_connections
                ]
                lower_limit = str(
                    "i \in "
                    +",".join(this_term_connections)
                )

        upper_limit = str(
            "N={}".format(dim)
        )
        new_term = str(
            "\sum"
            + "_{"
                + lower_limit
            + "}"
            + "^{"
                + upper_limit
            + "}"
            + operator_string
            # + "\hat{\sigma}^{"
            # + "}_{" + "i" + "}"
        )
        
        if n_sites not in latex_terms:
            latex_terms[n_sites] = {}
        if operator not in latex_terms[n_sites]:
            latex_terms[n_sites][operator] = new_term


    site_numbers = sorted(latex_terms.keys())
    all_latex_terms = []
    latex_model = ""
    for n in site_numbers:
        for term in latex_terms[n]:
            all_latex_terms.append(latex_terms[n][term])
    latex_model = "+".join(all_latex_terms)
    return "${}$".format(latex_model)


def lattice_pauli_likewise_concise(name, **kwargs):
    r""" 
    Don't list every pair of connected sites; just sum over \mathcal{C}
    """
    separate_terms = name.split('+')
    latex_term = ""
    latex_terms = {}
    for term in separate_terms:
        components = term.split('_')
        try:
            components.remove('pauliLikewise')
        except:
            print("Couldn't remove pauliLikewise from", name)
        this_term_connections = []
        for l in components:
            if l[0] == 'd':
                dim = int(l.replace('d', ''))
            elif l[0] == 'l':
                operator = str(l.replace('l', ''))
            else:
                n_sites = len(l.split('J'))
                sites = l.replace('J', ',')
                # if n_sites > 1: sites = "(" + str(sites) + ")"
                this_term_connections.append(sites)

        # limits for sum
        lower_limit = str(
            "i \in "
            +",".join(this_term_connections)
        )
        if n_sites == 1: 
            operator_string = str("\sigma_{ k }^{" + str(operator) + "}")
            sites_not_present = list(
                set([int(i) for i in this_term_connections]) 
                - set(range(1, dim+1))
            )
            if len(sites_not_present) == 0:
                lower_limit = "i=1"
        elif n_sites == 2:
            operator_string = str("\sigma_{ \langle k, l \\rangle }^{" + str(operator) + "}")
            nns = [(str(n), str(n+1) ) for n in range(1, dim)]
            nns = [','.join(list(nn)) for nn in nns]
            nns = set(nns)
            sites_not_present = list(
                set(this_term_connections)
                - nns
            )
            if len(sites_not_present) == 0:
                lower_limit = "i"
                operator_string = str(
                    "\hat{\sigma}\limits_{i}^{" + str(operator) + "}"
                    + "\hat{\sigma}\limits_{i+1}^{" + str(operator) + "}"
                )
            else: 
                this_term_connections = [
                    "({})".format(c) for c in this_term_connections
                ]
                lower_limit = str(
                    "i \in \mathcal{C}"
#                     +",".join(this_term_connections)
                )

        upper_limit = str(
            "N={}".format(dim)
        )
        new_term = str(
            "\sum"
            + "\limits_{"
                + lower_limit
            + "}"
            + "^{"
                + upper_limit
            + "}"
            + operator_string
            # + "\hat{\sigma}^{"
            # + "}_{" + "i" + "}"
        )
        
        if n_sites not in latex_terms:
            latex_terms[n_sites] = {}
        if operator not in latex_terms[n_sites]:
            latex_terms[n_sites][operator] = new_term


    site_numbers = sorted(latex_terms.keys())
    all_latex_terms = []
    latex_model = ""
    for n in site_numbers:
        for term in latex_terms[n]:
            all_latex_terms.append(latex_terms[n][term])
    latex_model = "+".join(all_latex_terms)
    return r"${}$".format(latex_model)
    


def fermi_hubbard_latex(
    name,
    **kwargs
):
    # TODO put in qmla.shared_functionality.latex_model_names
    # TODO gather terms in list, sort alphabetically and combine for latex
    # str
    basis_vectors = {
        'vac': np.array([1, 0, 0, 0]),
        'down': np.array([0, 1, 0, 0]),
        'up': np.array([0, 0, 1, 0]),
        'double': np.array([0, 0, 0, 1])
    }

    basis_latex = {
        'vac': 'V',
        'up': r'\uparrow',
        'down': r'\downarrow',
        'double': r'\updownarrow'
    }

    number_counting_terms = []
    hopping_terms = []
    chemical_terms = []
    terms = name.split('+')
    term_type = None
    for term in terms:
        constituents = term.split('_')
        for c in constituents:
            if c[0:2] == 'FH':
                term_type = c[2:]
                continue  # do nothing - just registers what type of matrix to construct
            elif c in list(basis_vectors.keys()):
                spin_type = c
            elif c[0] == 'd':
                num_sites = int(c[1:])
            else:
                sites = [str(s) for s in c.split('h')]

        if term_type == 'onsite':
            term_latex = r"\hat{{N}}_{{{}}}".format(sites[0])
            number_counting_terms.append(term_latex)
        elif term_type == 'hop':
            term_latex = '\hat{{H}}_{{{}}}^{{{}}}'.format(
                ",".join(sites),  # subscript site indices
                basis_latex[spin_type]  # superscript which spin type
            )
            hopping_terms.append(term_latex)
        elif term_type == 'chemical':
            term_latex = r"\hat{{C}}_{{{}}}".format(sites[0])
            chemical_terms.append(term_latex)

    latex_str = ""
    for term_latex in (
        sorted(hopping_terms) + sorted(number_counting_terms) +
        sorted(chemical_terms)
    ):
        latex_str += term_latex
    latex_str = "${}$".format(latex_str)
    return latex_str



def lattice_set_fermi_hubbard(name, **kwargs):
    separate_terms = name.split('+')

    all_terms = []
    for term in separate_terms:
        components = term.split('_')
        if 'FH-hopping-sum' in components:
            components.remove('FH-hopping-sum')
            connected_sites = ""
            for c in components:
                if c in ['down', 'up']:
                    spin_type = c
                elif c[0] == 'd':
                    num_sites = int(c[1:])
                else:
                    sites = [int(s) for s in c.split('h')]
                    connected_sites += str(
                        "({},{})".format(sites[0], sites[1])
                    )
            
            if spin_type == 'up':
                spin_label = str("\\uparrow")
            elif spin_type == 'down':
                spin_label = str("\\downarrow")
            new_term = str(
                "\hat{H}^{" + spin_label + "}"
                + "_{"
                + connected_sites
                + "}"
            )
            all_terms.append(new_term)
        elif 'FH-onsite-sum' in components:
            components.remove('FH-onsite-sum')
            sites = []
            for c in components:
                if c[0] == 'd':
                    num_sites = int(c[1:])
                else:
                    sites.append(int(c))
            sites = sorted(sites)
            sites = ','.join([str(i) for i in sites])
            sites_not_present =  list(
                set(range(1, num_sites+1))
                - set(sites)
            )
            if len(sites_not_present) > 0:
                new_term = str(
                    "\hat{N" + "}^{" 
                    + str(num_sites)
                    + "}_{" + sites + "}"
                )
            else:
                new_term = str(
                    "\hat{N" + "}^{" 
                    + str(num_sites)
                    + "}"
                )
            all_terms.append(new_term)

    model_string = '+'.join(all_terms)
    return "${}$".format(model_string)
            