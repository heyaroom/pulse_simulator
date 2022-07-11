import numpy as np
import scipy.sparse as sp
from .util.transform import qubitize
from .util.tensor_product import TensorProduct
from .util.frame import get_system_dressed_frame
from .util.leakage import get_computational_basis

class Qubit:
    """Class for standard harmonic oscillator"""
    
    def __init__(self, idx, dim, frequency, anharmonicity=None):
        """initialize the parameters of the system
        Args:
            idx (int) : index of the system
            dim (int) : dimension of the system
            frequency (float) : eigenfrequency of the system
            anharmonicity (float) : anharmonicity of the system
        """
        
        if dim < 2:
            raise ValueError(f'Qubit dimension must be set >= 2')
        if dim >= 3:
            if anharmonicity is None:
                anharmonicity = 0
                
        self.idx = idx
        self.dim = dim
        self.frequency = frequency
        self.anharmonicity = anharmonicity

        self.E = np.zeros([self.dim, self.dim])
        self.I = np.identity(self.dim)
        self.D = sp.spdiags(np.sqrt(range(dim)), 1, dim, dim, format='csr').toarray()
        self.C = self.D.T.conj()
        self.X = self.C + self.D
        self.Y = 1j*(self.C - self.D)
        # self.Z = self.I - 2*self.C@self.D
        self.Z = 2*self.C@self.D
        self.A = self.C@self.C@self.D@self.D
        
        self.Qi = qubitize(self.I)
        self.Qx = qubitize(self.X)
        self.Qy = qubitize(self.Y)
#         self.Qz = qubitize(self.Z)
        self.Qz = qubitize(self.I) - qubitize(self.Z)
        self.Ql = self.I - self.Qi

        self.S0 = np.diag([1,0]+[0]*(self.dim-2))
        self.S1 = np.diag([0,1]+[0]*(self.dim-2))
        if dim >= 3:
            self.S2 = np.diag([0,0,1]+[0]*(self.dim-3))
        self.Sp = 0.5*np.einsum("i,j->ij", [+1,+1]+[0]*(self.dim-2), [+1,+1]+[0]*(self.dim-2))
        self.Sm = 0.5*np.einsum("i,j->ij", [+1,-1]+[0]*(self.dim-2), [+1,-1]+[0]*(self.dim-2))
        self.Sc = 0.5*(self.S0 + self.S1)

    def __repr__(self):
        print_str = "-"*50 + "\n"
        print_str += f"Qubit ({self.idx}) \n"
        print_str += "*" + f" Dimension     = {self.dim} \n"
        print_str += "*" + f" Frequency     = {self.frequency} \n"
        print_str += "*" + f" Anharmonicity = {self.anharmonicity} \n"
        print_str += "-"*50
        return print_str

    def __str__(self):
        return self.__repr__()
        
    def hamiltonian(self, frame_frequency):
        """return the hamiltonian on the rotating frame
        Args:
            frame_frequency (float) : rotation frequency of the system simulating the time evolution
        Returns:
            H (np.array) : hamiltonian of the system on the rotating frame
        """
        
        H = 0.5*(self.frequency-frame_frequency)*self.Z + 0.5*self.anharmonicity*self.A
        return H
    
class Coupling:
    """Class for exchange interaction"""
    
    def __init__(self, coupling, qubits):
        """initialize the parameters of the system
        Args:
            coupling (float) : coupling strength
            qubits (tuple) : tuple of the index of the qubit to be coupled such as (0,1)
        """
        
        if len(qubits) != 2:
            raise ValueError(f'Only 2-Qubits can be selected.')
        self.qubits = qubits
        self.coupling = coupling
        self.detuning = self.qubits[0].frequency - self.qubits[1].frequency
        
        self.XX = np.kron(self.qubits[0].X, self.qubits[1].X)
        self.YY = np.kron(self.qubits[0].Y, self.qubits[1].Y)
        self.H = 0.5*self.coupling*(self.XX + self.YY)
        
    def __repr__(self):
        print_str = "-"*50 + "\n"
        print_str += f"Coupling \n"
        print_str += "*" + f" Qubits        = Q{self.qubits[0].idx}-Q{self.qubits[1].idx} \n"
        print_str += "*" + f" Coupling      = {self.coupling} \n"
        print_str += "*" + f" Detuning      = {self.detuning} \n"
        print_str += "-"*50
        return print_str

    def __str__(self):
        return self.__repr__()
    
    def hamiltonian(self):
        """return the hamiltonian on the rotating frame
        Returns:
            H (np.array) : hamiltonian of the system on the rotating frame
        """
        return self.H

class Drive:
    """Class for microwave irradiation"""
    
    def __init__(self, idx, qubit, amplitude, frequency):
        """initialize the parameters of the system
        Args:
            idx (int) : index of the microwave drive (corresponding to the Port index in the sequence_parser)
            qubit (int) : index of the target qubit to be irraddiated
            amplitude (float) : microwave drive amplitude
            frequency (float) : microwave drice frequency
        """
        
        self.idx = idx
        self.qubit = qubit
        self.amplitude = amplitude
        self.frequency = frequency
        
        self.Or = 0.5*self.amplitude*self.qubit.X
        self.Oi = 0.5*self.amplitude*self.qubit.Y
        
    def __repr__(self):
        print_str = "-"*50 + "\n"
        print_str += f"Drive ({self.idx}) \n"
        print_str += "*" + f" Target qubit   = Q{self.qubit.idx} \n"
        print_str += "*" + f" Amplitude      = {self.amplitude} \n"
        print_str += "*" + f" Frequency      = {self.frequency} \n"
        print_str += "-"*50
        return print_str

    def __str__(self):
        return self.__repr__()
    
    def operator_real(self):
        """return the hamiltonian corresponds to real part of the waveform
        Returns:
            self.Or (np.array) : hamiltonian corresponds to real part of the waveform
        """
        return self.Or
    
    def operator_imag(self):
        """return the hamiltonian corresponds to imaginary part of the waveform
        Returns:
            self.Oi (np.array) : hamiltonian corresponds to imaginary part of the waveform
        """
        return self.Oi
    
class System:
    """Class for quantum systems containing coupled standard harmonic oscillators and microwave irradiation"""
    
    def __init__(self):
        self.qubits = {}
        self.coupls = {}
        self.drives = {}
              
    def __repr__(self):
        print_str  = "#"*25 + "Pulse Simulator".center(20) + "#"*25 + "\n"
        print_str += "#"*25 + "Qubit".center(20) + "#"*25 + "\n"
        for idx, qubit in self.qubits.items():
            print_str += str(qubit) + "\n"
            
        print_str += "#"*25 + "Coupling".center(20) + "#"*25 + "\n"
        for idx, coupling in self.coupls.items():
            print_str += str(coupling) + "\n"
        
        print_str += "#"*25 + "Drive".center(20) + "#"*25 + "\n"
        for idx, drive in self.drives.items():
            print_str += str(drive) + "\n"
        print_str += "#"*70 + "\n"
        
        return print_str

    def __str__(self):
        return self.__repr__()
        
    def add_qubit(self, idx, dim, frequency, anharmonicity=None):
        """add a new standard harmonic oscillator
        Args:
            idx (idx) : index of the new qubit
            dim (idx) : dimension of the new qubit
            frequency (float) : eigenfrequency of the new qubit
            anharmonicity (float) : anharmonicity of the new qubit
        """
        
        if idx in self.qubits.keys():
            raise ValueError(f'Qubit index {idx} is already used.')
        self.qubits[idx] = Qubit(idx, dim, frequency, anharmonicity)
        
    def add_coupling(self, idxs, coupling):
        """add a new exchange interaction
        Args:
            idxs (tuple) : tuple of the index of the coupled qubits
            coupling (float) : coupling strength
        """
        
        for idx in idxs:
            if idx not in self.qubits.keys():
                raise ValueError(f'Qubit {idx} is not found.')
        if idxs in self.coupls:
            raise ValueError(f'Coupling index {idxs} is already used.')
        qubits = [self.qubits[idxs[0]], self.qubits[idxs[1]]]
        self.coupls[idxs] = Coupling(coupling, qubits)
        
    def add_drive(self, idx, qubit, amplitude, frequency):
        """add a new microwave drive
        Args:
            idx (int) : index of the new microwave drive
            qubit (int) : index of the qubit to be irradiated
            amplitude (float) : microwave drive amplitude
            frequency (float) : microwave drive frequency
        """
        
        if idx in self.drives.keys():
            raise ValueError(f'Drive index {idx} is already used.')
        if qubit not in self.qubits.keys():
            raise ValueError(f'Qubit {qubit} is not found.')
        self.drives[idx] = Drive(idx, self.qubits[qubit], amplitude, frequency)

    def compile(self, frame_frequency=None):
        """compute the system time-evolution
        Args:
            frame_frequency (float) : rotation frequency of the system simulating the time evolution
        """
        
        if frame_frequency is None:
            frame_frequency = np.mean([q.frequency for q in self.qubits.values()])
        self.frame_frequency = frame_frequency
        
        self.dims = [q.dim for q in self.qubits.values()]
        self.dim = np.prod(self.dims)
        
        self.static_hamiltonian = 0
        for idx, q in self.qubits.items():
            tp = TensorProduct(*self.dims)
            tp.prod(q.hamiltonian(frame_frequency), idx)
            self.static_hamiltonian += tp.get_operator()
        for idxs, c in self.coupls.items():
            tp = TensorProduct(*self.dims)
            tp.prod(c.hamiltonian(), idxs)
            self.static_hamiltonian += tp.get_operator()
            
        self.dynamic_operators = {}
        self.dynamic_detunings = {}
        for idx, d in self.drives.items():
            tp = TensorProduct(*self.dims)
            tp.prod(d.operator_real(), d.qubit.idx)
            operator_real = tp.get_operator()
            tp = TensorProduct(*self.dims)
            tp.prod(d.operator_imag(), d.qubit.idx)
            operator_imag = tp.get_operator()
            self.dynamic_operators[idx] = (operator_real, operator_imag)
            self.dynamic_detunings[idx] = d.frequency - frame_frequency
            
        self.conv, self.frame = get_system_dressed_frame(self)
        self.comp = get_computational_basis(self)
        
        # operator conversion onto the system dressed frame
        self.static_hamiltonian_on_frame = self.conv.T.conj()@self.static_hamiltonian@self.conv
        self.dynamic_operators_on_frame = {}
        for key, val in self.dynamic_operators.items():
            self.dynamic_operators_on_frame[key] = self.conv.T.conj()@val@self.conv
        self.frame_on_frame = self.conv.T.conj()@self.frame@self.conv