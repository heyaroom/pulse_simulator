import numpy as np
import scipy.sparse as sp
from .util.transform import qubitize
from .util.tensor_product import TensorProduct

class Qubit:
    def __init__(self, idx, dim, frequency, anharmonicity=None):
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
        self.Qz = qubitize(self.Z)
        self.Ql = self.I - self.Qi

        self.S0 = np.diag([1,0]+[0]*(self.dim-2))
        self.S1 = np.diag([0,1]+[0]*(self.dim-2))
        if dim >= 3:
            self.S2 = np.diag([0,0,1]+[0]*(self.dim-3))
        self.Sp = 2**(-0.5)*np.diag([+1,+1]+[0]*(self.dim-2))
        self.Sm = 2**(-0.5)*np.diag([+1,-1]+[0]*(self.dim-2))
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
        H = 0.5*(self.frequency-frame_frequency)*self.Z + 0.5*self.anharmonicity*self.A
        return H
    
class Coupling:
    def __init__(self, coupling, qubits):
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
        return self.H

class Drive:
    def __init__(self, idx, qubit, amplitude, frequency):
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
        return self.Or
    
    def operator_imag(self):
        return self.Oi
    
class System:
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
        if idx in self.qubits.keys():
            raise ValueError(f'Qubit index {idx} is already used.')
        self.qubits[idx] = Qubit(idx, dim, frequency, anharmonicity)
        
    def add_coupling(self, idxs, coupling):
        for idx in idxs:
            if idx not in self.qubits.keys():
                raise ValueError(f'Qubit {idx} is not found.')
        if idxs in self.coupls:
            raise ValueError(f'Coupling index {idxs} is already used.')
        qubits = [self.qubits[idxs[0]], self.qubits[idxs[1]]]
        self.coupls[idxs] = Coupling(coupling, qubits)
        
    def add_drive(self, idx, qubit, amplitude, frequency):
        if idx in self.drives.keys():
            raise ValueError(f'Drive index {idx} is already used.')
        if qubit not in self.qubits.keys():
            raise ValueError(f'Qubit {qubit} is not found.')
        self.drives[idx] = Drive(idx, self.qubits[qubit], amplitude, frequency)

    def compile(self, frame_frequency):
        self.dims = [q.dim for q in self.qubits.values()]
        
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