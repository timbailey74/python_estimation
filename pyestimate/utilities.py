#
# Basic algebraic utilities. Ported from matlab_utilities. Tim Bailey 2015.
#

import numpy as np

#
#

def reprow(x, N):
    # Alternatives: return np.tile(x, (N,1))
    # Alternatives: return np.outer(np.ones(N), x)
    return np.tile(x, (N,1))

def repcol(x, N):
    return np.outer(x, np.ones(N))

def repcol_(x, N):
    if len(x.shape) == 1:
        return np.tile(x, (N,1)).T # for [] arrays
    elif x.shape[1] == 1:
        return np.tile(x, (1,N)) # for [[]] column arrays
    else:
        raise ValueError('Must be an array or single-column matrix')


# Given a 2-D array, eliminate all duplicate rows
def unique_rows(a):
    # from: http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array/8567929#8567929
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))



# Generate indices for the locations in range [0,N) that are not in idx
def index_other(N, idx):
    i = np.ones(N)
    i[idx] = 0
    idy = np.arange(N)
    return idy[i!=0]


# Generate a single index array from a list of slices
def slice2index(args):
    return np.hstack([np.arange(a.start, a.stop, a.step) for a in args])

# Generate a single index array from a list of (start, end) or (start, end, incr) tuples
def tuple2index(args):
    return np.hstack([np.arange(*a) for a in args])


# Multiplication of the form F*P*G
def triprod(F, P, G, order_F_PG=True):
    if order_F_PG:
        return np.dot(F, np.dot(P, G))
    else:
        return np.dot(np.dot(F, P), G)


# Symmetric pos-def product of the form F*P*F.T
def symprod(F, P, decompose=True):
    if decompose:  # Cholesky decomposition; guarantees symmetric pos-def
        L = sci.cholesky(P, lower=True)  # P = L*L.T
        FL = np.dot(F, L)
        return np.dot(FL, FL.T)
    else:  # Simple matrix multiple; guarantee symmetry only
        return force_symmetry(triprod(F, P, F.T))


# Product of diagonal matrix D and full matrix A, where d is diagonal of D.
# If left is True: returns dot(D,A), otherwise, dot(A,D)
def diagprod(d, A, left=True):
    assert len(d.shape) == 1
    if left:  # this works because python broadcasts row-wise
        return (d*A.T).T
    else:
        return d*A

# Symmetric product D*A*D, where d is diagonal of D.
def symdiagprod(d, A):
    return diagprod(d, diagprod(d, A, True), False)
#    return diagprod(d, diagprod(d, A, False).T, False)  # if A symmetric, this is a tiny bit cheaper (ie., one less transpose)


# Force matrix A to be symmetrical by averaging it with its transpose
def force_symmetry(A):
    return 0.5*(A + A.T)


# Compute log(sum(w)) given set of log(w), with improved numerics 
def log_sum(logw):
    c = np.max(logw, 0) 
    shifted_exp = np.exp(logw - c)
    shifted_sum = np.sum(shifted_exp, 0)
    return c + np.log(shifted_sum) 

#
def pi2pi(angle):
    """
    Normalised angle calculation; bound angle(s) between +/- pi.
    :param angle: Scalar angle or array of angles
    :return: normalised angles
    """
    return (angle + np.pi) % (2*np.pi) - np.pi


#
def deg2rad(degrees, normalise=True):
    radians = degrees * np.pi / 180.0
    return pi2pi(radians) if normalise else radians

def rad2deg(radians):
    return radians * 180.0 / np.pi


#
def dist_sqr(x1, x2):
    """
    Compute the square distances of each vector in x1 to each vector in x2. To
    compute the set of Euclidean distances, simply compute sqrt(d2). This
    equation is adapted from Netlab, dist2.m, by Ian T Nabney.
    :param x1: matrix of M column vectors
    :param x2: matrix of N column vectors
    :return: d2 - M x N matrix of square distances
    """
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('Vectors must have the same dimension.')

    N1 = x1.shape[1] if len(x1.shape) > 1 else 1
    N2 = x2.shape[1] if len(x2.shape) > 1 else 1
    d2 =  np.tile(np.sum(x1*x1,0), (N2,1)).T \
        + np.tile(np.sum(x2*x2,0), (N1,1))   \
        - 2 * np.dot(x1.T, x2)
    d2[d2<0] = 0 # Ensure rounding errors do not give negative values
    return d2


# Define static variables for a function
# See: http://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


# Scale a vector to approximate zero-mean and unit-variance
class Scale:
    def __init__(self, mean, sigma):
        self.mean = mean.copy()
        self.sigma = sigma.copy()
    def scale(self, x, idx=None):
        if idx is not None:
            return (x - self.mean[idx]) / self.sigma[idx]
        return (x - self.mean) / self.sigma
    def unscale(self, sx, idx=None):
        if idx is not None:
            return self.mean[idx] + sx*self.sigma[idx]
        return self.mean + sx*self.sigma

