import itertools
import numpy as np
from .tensor_product import TensorProduct

# def get_system_dressed_frame(sys): # perturbation
#     qubit_freqs = {}
#     for idx, q in sys.qubits.items():
#         qubit_freqs[idx] = q.frequency
#     for idxs, c in sys.coupls.items():
#         w0 = sys.qubits[idxs[0]].frequency
#         w1 = sys.qubits[idxs[1]].frequency
#         a0 = sys.qubits[idxs[0]].anharmonicity
#         a1 = sys.qubits[idxs[1]].anharmonicity
#         g01 = c.coupling
#         qubit_freqs[idxs[0]] += g01**2/(w0-w1) + g01**2*(a0+a1)/(((w0+a0)-w1)*(w0-(w1+a1))) # 3rd order
#         qubit_freqs[idxs[1]] += g01**2/(w1-w0) + g01**2*(a0+a1)/(((w0+a0)-w1)*(w0-(w1+a1))) # 3rd order
        
#     system_label = list(itertools.product(*[range(d) for d in sys.dims]))
    
#     system_energy = {}
#     for l in system_label:
#         tmp = 0
#         for q in sys.qubits.keys():
#             tmp += l[q]*(qubit_freqs[q] - sys.frame_frequency)
#         system_energy[l] = tmp
    
#     system_basis = {}
#     for l in system_label:
        
        
#     system_frame = 0
#     for l in system_label:
#         system_frame += system_energy[l] * np.einsum("i,j->ij", system_basis[l], system_basis[l])
        
#     return system_basis, system_frame

def get_system_dressed_frame(sys): # numerical
    val, vec = np.linalg.eig(sys.static_hamiltonian)

    idx_list = []
    for i in vec.T:
        idx = np.argmax(abs(i)**2)
        idx_list.append(idx)
    idx_list = np.array(idx_list)

    vec = vec.T[np.argsort(idx_list)]
    val = val[np.argsort(idx_list)]
    
    # V = vec
    # E = np.diag(val)
    # H = V.T@E@V.conj()
    # np.allclose(H, sys.static_hamiltonian) = True

    label = list(itertools.product(*[range(d) for d in sys.dims]))

    qfreqs = {} # on rotating frame
    for q in sys.qubits.keys():
        qfreqs[q] = 0
        for i,j in zip(label, val):
            if i[q] == 0 and (2 not in i):
                qfreqs[q] -= j
            if i[q] == 1 and (2 not in i):
                qfreqs[q] += j
        qfreqs[q] *= 0.5

    qenergy = []
    for l in label:
        tmp = 0
        for q in sys.qubits.keys():
            tmp += l[q]*qfreqs[q]
        qenergy.append(tmp)
    
    conv = vec.T
    frame = conv@np.diag(qenergy)@conv.T.conj()
    
    return conv, frame
    