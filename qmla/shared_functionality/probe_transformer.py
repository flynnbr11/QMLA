import sys
import os
import numpy as np


class ProbeTransformation():
    r"""
    Transform probes between bases so that the same probes
    can be used on different physical representations. 

    """
    def __init__(self):
        self.transform_map = None
    
    def transform(self, probe):
        return probe
        

class FirstQuantisationToJordanWigner(ProbeTransformation):
    r"""
    
    This class is used to map from first transformation 
    to an occupation basis which can be acted on by operators
    which have undergone the Jordan Wigner transformation. 
    
    e.g. a spin system has 
    :math:`
    \| \psi^{(1)}\rangle = \alpha_{\downarrow} \| \downarrow \rangle + \alpha{\uparrow} \| \uparrow \rangle
     = \alpha_{\downarrow} \| \downarrow \rangle + \alpha{\uparrow} \| \uparrow \rangle`
     
     An equivelent representation which can be used by second quantised operators is
     a binary encoding: the first element is whether a down spin is present; 
     the second element is wehther an up spin is present:
     :math:` \| \psi^{(2)}\rangle = \alpha_{\downarrow} \| 10 \rangle + \alpha{\uparrow} \| 01 \rangle`
     
    We can therefore move between bases by mapping a probe :math:`\| \psi \rangle`, 
    from the first quantisation picture to the second (Jordan Wigner) picture, via
    
    :math:`\| \psi \rangle = \sum_{i} \| \psi^{(2)_{i}} \rangle \langle \psi^(1)_{i} \| \psi_{i} \rangle`
    
    """
    def __init__(self, max_num_qubits):
        self.transform_map = {
            q : self.paired_probe_bases(q)
            for q in range(1, 1+max_num_qubits)
        }
        

    def transform(self, probe):
        r"""
        Transform ``probe`` to the new basis. binary_to_first_quantisation_probe
        """
        
        num_qubits = np.log2(probe.shape[0])
        transformer = self.transform_map[num_qubits]
        new_probe = None
        
        for basis in transformer:
            
            value = np.dot(basis[0], probe)
            if new_probe is None: 
                new_probe = value * basis[1]
            else:
                new_probe += value * basis[1]
        
        return new_probe

    
    def paired_probe_bases(
        self, 
        num_qubits
    ):
        r"""
        Produce pairs of states corresponding to the same basis, 
        where one element is the basis in first quantisation, 
        and the second is the basis through Jordan Wigner transformation. 
        """

        bases = []
        for i in range(2**num_qubits):
            binary_string = bin(i)[2:].zfill(num_qubits)
            first_q = self.binary_to_first_quantisation_probe(binary_string)
            jw = self.binary_string_to_jordan_wigner_probe(binary_string)

            bases.append(
                (first_q, jw)
            )
        return bases
    
    
    def binary_to_first_quantisation_probe(self, binary_string):
        r"""
        |0> -> (1 0)
        |1> -> (0 1)
        """

        bases = {
            0 : np.array([1,0]),
            1 : np.array([0,1])
        }

        probe = None
        for s in binary_string:
            b = bases[int(s)]
            if probe is None: 
                probe = b
            else:
                probe = np.kron(probe, b)

        return probe


    def binary_string_to_jordan_wigner_probe(self, binary_string):
        """
        First map string to occupation basis string, i.e.
        down: 0 -> 10
        up: 1 -> 01
        e.g. going from first quantisation to occupation basis:
        |001> --> '101001' 

        then get state from new binary string
        """

        occupation_blocks = {
            '0' : '10', # down
            '1' : '01' # up
        }

        occupation_string = ''

        for s in binary_string:
            occupation_string += occupation_blocks[s]

        probe = self.binary_to_first_quantisation_probe(occupation_string)
        return probe
