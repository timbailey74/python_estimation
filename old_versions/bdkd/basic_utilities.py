#
# Basic Gaussian manipulation and inference utilities. Ported from
# matlab_utilities. Tim Bailey 2015.
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slin
import scipy.optimize as sopt
import scipy.special as sp

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

#
def chi_square_mass(x, n):
    """
    Compute n-dof Chi-square mass function at points x
    Reference: Press, "Numerical Recipes in C", 2nd Ed., 1992, page 221.
    """
    return sp.gammainc(n/2, x/2)

# Modified from my Matlab code
def chi_square_bound(prob, n):
    assert 0 < prob < 1, 'Probability must be in interval (0, 1)'
    # Hunt for xu that gives upper bound on prob
    xu, prob_xu = 3, 0
    while prob_xu < prob:
        xu *= 2
        prob_xu = chi_square_mass(xu, n)
    # Solve for root
    f = lambda x: chi_square_mass(x, n) - prob
    return sopt.brentq(f, 0, xu)

# Draw samples from a multivariate Gaussian
def gauss_samples(x, P, n):
    """
    :param x: mean vector
    :param P: covariance matrix
    :param n: number of samples
    :return: samples from Gauss(x,P); a matrix of n vectors
    """
    L = slin.cholesky(P, lower=True)
    X = np.random.standard_normal((len(x), n))
    return np.dot(L, X) + repcol(x, n)

# Draw samples from Gaussian, but keeping only those samples that fall within a given probability bound
def gauss_samples_bounded(x, P, N, prob):
    # Solution 1: draw uniformly from unit circle, and transform according to {x,P}
    # Solution 2: draw from {x,P} and reject outliers
    pass

# http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
def sample_uniform_unit_hypersphere(dim, n):
    u = np.random.rand(n)
    if dim == 1:
        return 2*(0.5 - u)
    x = np.random.randn(dim, n)
    scale = u**(1/dim) / np.sqrt(sum(x**2))
    return scale*x  # note, python broadcasting does the correct replication of scale

# Compute mean and covariance from set of weighted samples
def sample_mean_weighted(x, w, normalised_weights=True):
    if normalised_weights:
        if abs(1 - sum(w)) > 1e-12:
            raise ArithmeticError('Weights should be normalised')
        if sum(w!=0) <= len(x):
            raise ArithmeticError('Samples form a hyperplane, covariance rank deficient')
    w = reprow(w, x.shape[0])
    xm = np.sum(w*x, 1)
    xc = x - repcol(xm, x.shape[1])
    P = np.dot(w*xc, xc.T)
    return xm, P

# Evaluate a Gaussian with covariance S at distance v from the mean
def gauss_evaluate(v, S, logflag=False):
    """
    gauss_evaluate() - evaluate multivariate Gaussian function with covariance S at offsets v
    """
    D = v.size
    L = slin.cholesky(S, lower=True)
    f = slin.solve_triangular(L, v, lower=True) # 'normalised' innovation; f = inv(L)*v
    E = -0.5 * np.sum(f*f, axis=0)
    if logflag:
        C = 0.5*D*np.log(2*np.pi) + np.sum(np.log(L.diagonal()))
        w = E - C
    else:
        C = (2*np.pi)**(D/2.) * np.prod(L.diagonal())
        w = np.exp(E) / C
    return w

def pi2pi(angle):
    """
    Normalised angle calculation; bound angle(s) between +/- pi.
    :param angle: Scalar angle or array of angles
    :return: normalised angles
    """
    return (angle + np.pi) % (2*np.pi) - np.pi

def dist_sqr_old(x1, x2):
    """
    Compute the square distances of each vector in x1 to each vector in x2. To
    compute the set of Euclidean distances, simply compute sqrt(d2). This
    equation is adapted from Netlab, dist2.m, by Ian T Nabney.
    :param x1: matrix of N column vectors
    :param x2: matrix of M column vectors
    :return: d2 - M x N matrix of square distances
    """
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('Vectors must have the same dimension.')

    N1 = x1.shape[1] if len(x1.shape) > 1 else 1
    N2 = x2.shape[1] if len(x2.shape) > 1 else 1
    d2 =  np.tile(np.sum(x2*x2,0), (N1,1)).T \
        + np.tile(np.sum(x1*x1,0), (N2,1))   \
        - 2 * np.dot(x2.T, x1)
    d2[d2<0] = 0 # Ensure rounding errors do not give negative values
    return d2

def dist_sqr(x1, x2):
    """
    Same as dist_sqr_old() but with x1 as M-columns and x2 as N columns.
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

def dist_mahalanobis(v, S):
    """
     INPUTS:
       v - a set of innovation vectors.
       S - the covariance matrix for the innovations.

     OUTPUT:
       M - set of Mahalanobis distances for each v(:,i).
    """
    L = slin.cholesky(S, lower=True)
    f = slin.solve_triangular(L, v, lower=True) # 'normalised' innovation; f = inv(L)*v
    return np.sum(f*f, axis=0)

def dist_mahalanobis_all_pairs(x1, x2, S):
    """
    Equivalent to dist_sqr(x1, x2) but using Mahalanobis rather than Euclidean distance.
    """
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('Vectors must have the same dimension.')
    dim, n1, n2 = x1.shape[0], x1.shape[1], x2.shape[1]
    v = np.tile(x1, (1,n2))
    v -= np.tile(x2, (n1,1)).reshape(dim, v.shape[1], order='F')
    return dist_mahalanobis(v, S).reshape(n1, n2, order='F')

# FIXME: expects x to be a column-vector (2,1) or flattened (2,); will not accept row-vector (1,2)
def ellipse_mass(x, P, prob, N=60, test=False):
    if test:  # use matrix square-root
        R = slin.sqrtm(P)
    else:  # use eigen-vectors
        d, v = slin.eigh(P)
        R = v * np.sqrt(d)  # broadcasts to equal: R = np.dot(v, np.diag(np.sqrt(d)))
    c = np.sqrt(chi_square_bound(prob, 2))
    phi = np.linspace(0, 2*np.pi, N)
    unit_circle = np.vstack((np.cos(phi), np.sin(phi)))
    if len(x.shape) == 1:  # ensure x is a column vector (required for broadcast below)
        x = x[:, np.newaxis]
    return x + c*np.dot(R, unit_circle)

#
def line_plot_conversion(a, b):
    # Assumes points are matrix of row-vectors
    M, N = a.shape
    nans = np.tile(np.nan*np.zeros(N), (M,1))
    block = np.hstack((a, b, nans))
    flat = block.flatten('C')
    return flat.reshape(M*3, N)

# Define static variables for a function
# See: http://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

#
# TEST CODE ------------------------------------------------------------------
#

@static_vars(a=0, b='hello')
def foo(c):
    foo.a += c
    print('a: {0}, b: {1}, c: {2}'.format(foo.a, foo.b, c))

def test_ellipse_mass():
    x = np.random.randn(2,1)
    P = np.random.randn(2,2)
    P = np.dot(P, P.T)
    e1 = ellipse_mass(x, P, 0.95, test=True)
    e2 = ellipse_mass(x, P, 0.95, test=False)
    plt.plot(e1[0,:], e1[1,:], '.-', e2[0,:], e2[1,:], '.-')

if __name__ == "__main__":
    foo(3)
    foo(7)
    test_ellipse_mass()
    plt.show()
