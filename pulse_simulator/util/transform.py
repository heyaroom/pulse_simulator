import numpy as np
import scipy.linalg as lin

def hermitize(X):
    Y = 0.5*(X + X.T.conj())
    return Y

def unitarize(X):
    Y = X@lin.inv(lin.sqrtm(X.T.conj()@X))
    return Y

def qubitize(operator):
    operator_2 = 0j*operator
    operator_2[np.ix_([0,1],[0,1])] = operator[np.ix_([0,1],[0,1])]
    return operator_2