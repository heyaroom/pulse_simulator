import itertools
import numpy as np

def get_computational_basis(sys):
    label = list(itertools.product(*[range(d) for d in sys.dims]))
    
    computational_basis = []
    for idx, l in enumerate(label):
        if 2 not in l:
            computational_basis.append(idx)
    return computational_basis
            
def get_leakage(u, sys):
    leakage = 1 - abs(np.linalg.det(u[sys.comp][:,sys.comp]))
    return leakage
    