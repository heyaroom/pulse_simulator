import numpy as np
import scipy.linalg as lin

def hermitize(X):
    """approximate the input matrix as a hermitian matrix
    Args:
        X (np.array) : input matrix (should be near to a hermitian)
    Returns:
        Y (np.array) : output matrix approximated to be a hermitian
    """
    Y = 0.5*(X + X.T.conj())
    return Y

def unitarize(X):
    """approximate the input matrix as a unitary matrix
    Args:
        X (np.array) : input matrix (should be near to a unitary)
    Returns:
        Y (np.array) : output matrix approximated to be a unitary
    """
    Y = X@lin.inv(lin.sqrtm(X.T.conj()@X))
    return Y

def qubitize(X):
    """truncate the higher order levels from the input matrix
    Args:
        X (np.array) : input matrix
    Returns:
        Y (np.array) : output matrix truncated the higher order levels
    """
    Y = 0j*X
    Y[np.ix_([0,1],[0,1])] = X[np.ix_([0,1],[0,1])]
    return Y