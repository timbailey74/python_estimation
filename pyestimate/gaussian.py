# Convention: if x is a matrix of d-dimensional row-vectors
#   Exceptions: linalg.eigh gives e-vecs as col-vectors
# FIXME: change operations to either row-array form, or user-specified via axis

import numpy as np
import scipy.linalg as sl
from scipy.special import gammainc
from scipy.optimize import brentq, fminbound


# Compute mean and covariance from set of samples
def sample_mean(x):
    N = x.shape[1]
    xm = np.sum(x, 1) / N
    xc = x - repcol(xm, N)
    P = np.dot(xc, xc.T) / N
    return xm, P


# Compute sample mean and covariance, same as sample_mean() above, 
# but accounting for missing data
def sample_mean_cov(x, axis=0):
    """
    If no missing data, the same result can be obtained via:
        xm = np.mean(x, axis)
        N = x.shape[axis]
        Pm = np.cov(x, rowvar=axis)*(N-1)/N
    where (N-1)/N converts "unbiased" covariance to 2nd moment; the conversion
    is not important and makes negligible difference when N is large.
    """
    try:  # assume is masked array
        mask = 1 - x.mask.astype(int)  # want False -> 1, True -> 0
        x = x.data.copy()
        x[mask==0] = 0
    except:  # otherwise, ordinary array, no missing data
        mask = np.ones(x.shape)
    # Calculate mean
    Nx = np.sum(mask, axis)  # per-element counter for mean
    xm = np.sum(x, axis) / Nx  
    # Calculate covariance
    if axis:
        xc = x.T - xm
        mask = mask.T
    else:
        xc = x - xm
    xc[mask==0] = 0
    P = np.dot(xc.T, xc) / np.dot(mask.T, mask)
    return xm, P


# Mean and covariance of weighted samples
def sample_mean_weighted(x, w, axis=0, check=False):
    if check:
        if abs(1 - sum(w)) > 1e-12:
            raise ArithmeticError('Weights should sum to 1')
        if sum(w!=0) <= len(x):
            raise ArithmeticError('Samples form a hyperplane, covariance rank deficient')
    if axis:
        x = x.T
    xm = np.sum(w*x.T, 1)  # mean
    xc = x - xm  # zero-centred samples
    P = np.dot(w*xc.T, xc)  # covariance
    return xm, P


#
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


# Draw gaussian samples from an array of independent scalar pdfs
def gauss_samples_sigma(x, sigma, n=1):
    try: # each x has its own sigma
        X = [xi + si*np.random.standard_normal(n) for (xi,si) in zip(x, sigma)]
    except: # each x has same sigma
        X = [xi + sigma*np.random.standard_normal(n) for xi in x]
    return np.array(X)
# FIXME: instead of try-except block, above might be faster to write:
#   if np.size(sigma) == 1:  # best option, I think
# or:
#   if isinstance(sigma, float) ...
# or:
#   if not isinstance(sigma, (list, tuple, np.ndarray))
# or:
#   from collections import Iterable
#   if not isinstance(sigma, Iterable)


# Evaluate Gaussian pdf N(x,P) given L*L.T = P and L*f = x
# FIXME: expects f to be matrix of col-vectors
def gauss_evaluate_sqrtform(f, L, logw=False):
    D = f.shape[0]
    assert L.shape[0] == D
    E = -0.5 * np.sum(f*f, axis=0)
    if logw:
        C = 0.5*D*np.log(2*np.pi) + np.sum(np.log(L.diagonal()))
        w = E - C
    else:
        C = (2*np.pi)**(D/2.) * np.prod(L.diagonal())
        w = np.exp(E) / C
        assert np.all(w >= 0)
    return w


# Evaluate a Gaussian with covariance S at distance v from the mean
# FIXME: expects v to be matrix of col-vectors
def gauss_evaluate(v, S, logw=False):
    # FIXME: what if v or S are scalars?
    L = sl.cholesky(S, lower=True)
    f = sl.solve_triangular(L, v, lower=True)
    return gauss_evaluate_sqrtform(f, L, logw)


# Fast implementation of 1-D case
def gauss_evaluate_1d(x, P, logw=False):
    E = -0.5 * x**2 / P
    C = 2 * np.pi * P
    return E - np.log(C)/2 if logw else np.exp(E) / np.sqrt(C)


#
def gauss_power(P, r):
    """
    Compute the covariance and weight of Gauss(x,P)**r
    :param P: covariance matrix
    :param r: exponent, such that we want
    :return (w, P/r): weight and covariance of Gauss(x,P)**r == w*Gauss(x,P/r)
    """
    s = r - 1
    d = P.shape[0]
    e = sci.eigh(P, eigvals_only = True) # real-symmetric eigen-vals
    w = 1 / np.sqrt((2*np.pi)**(d*s) * r**d * np.prod(e**s))
    return w, P/r


# Vectorised version of gaussian conditional.
#   x is a single (D,) vector, or a (N,D) matrix of row vectors
#   vals is either (len(i),), or (N,len(i)), or (N,) in 1-D case
def gaussian_conditional(x, P, vals, i, logw=None):
    D, Di = len(P), len(i)
    assert len(P.shape) == 2 and not np.diff(P.shape)  # P matrix must be square
    assert x.shape[0] == D or x.shape[1] == D  # x is (D,) or (N,D)
    assert Di == 1 or vals.shape[0] == Di or vals.shape[1] == Di  
    # Index for variables not in "i"
    j = index_other(D, i)
    # Compute Pc
    Pii = P[np.ix_(i, i)]
    Pji = P[np.ix_(j, i)]
    Pjj = P[np.ix_(j, j)]
    Uii = sl.cholesky(Pii)  # upper-triangular Cholesky factor
    Uinv = sl.solve_triangular(Uii, np.eye(Di), check_finite=False)
    Wn = np.dot(Pji, Uinv)
    Pc = Pjj - np.dot(Wn, Wn.T)
    # Compute xc
    if len(vals.shape) == 1 and len(vals) != Di:
        vals = vals[:, np.newaxis]  # special case required for Di==1; convert to (N,1)
    xi, xj = (x[:,i], x[:,j]) if len(x.shape)==2 else (x[i], x[j])
    # The following two lines are vectorised for multiple values
    vn = np.dot(vals - xi, Uinv).T
    xc = xj + np.dot(Wn, vn).T
    # Compute w, only if required
    if logw is None:
        return xc, Pc
    w = gauss_evaluate_sqrtform(vn, Uii, logw)
    return xc, Pc, w


# Fast implementation of 2-D case
#   x is (2,) or (N,2); and vals is (N,)
def gaussian_conditional_2d(x, P, vals, i=1, logw=None):
    x0, x1 = x.T  # note: not flattened, because x may be a matrix of arrays
    P00, Pij, _, P11 = P.flatten()
    if i:  # condition on x[1]
        assert i == 1
        xi, xj, Pii, Pjj = x1, x0, P11, P00
    else:  # condition on x[0]
        xi, xj, Pii, Pjj = x0, x1, P00, P11
    Pc = Pjj - Pij**2 / Pii
    v = vals - xi
    xc = xj + v * Pij/Pii
    Pc = np.array([Pc])[:,np.newaxis]
    if len(vals)>1:  # FIXME: this check is required to match format of gaussian_conditional()
        xc = np.array(xc)[:,np.newaxis]
    if logw is None:
        return xc, Pc
    w = gauss_evaluate_1d(v, Pii, logw)
    return xc, Pc, w


#
def gaussian_product(x1, P1, x2, P2):
    Sc  = sl.cholesky(P1 + P2)  # upper triangular factor of convolved covariances
    Sci = sl.inv(Sc)
    Wc = np.dot(P1, Sci)
    vc = np.dot(Sci.T, x2-x1)
    return x1 + np.dot(Wc, vc), P1 - np.dot(Wc, Wc.T)


# Linear Kalman update; where v = z - H*x[idx] is innovation vector
def gaussian_product_projected(x, P, v, R, H, idx=None):
    P1, P2 = (P[np.ix_(idx,idx)], P[:,idx]) if idx is not None else (P,P)
    S = symprod(H, P1) + R
    U = sl.cholesky(S)  # upper triangular factor
    Ui = sl.inv(U)
    vc = np.dot(Ui.T, v)
    Wc = triprod(P2, H.T, Ui)
    return x + np.dot(Wc, vc), P - np.dot(Wc, Wc.T)


#
def covariance_intersection(x1, P1, x2, P2, w=None):
    Y1 = sl.inv(P1)
    Y2 = sl.inv(P2)
    if w is None:  # choose w to minimise determinant of fused covariance
        f = lambda w: 1 / sl.det(w*Y1 + (1-w)*Y2)
        w = fminbound(f, 0, 1)
    Y = w*Y1 + (1-w)*Y2
    y = w*np.dot(Y1, x1) + (1-w)*np.dot(Y2, x2)
    P = sl.inv(Y)
    x = np.dot(P, y)
    return x, P, w


#
def chi_square_density(x, n):
    """
    Compute n-dof Chi-square density function at points x
    Reference: Papoulis, "Probability, Random Variables and Stochastic Processes", 4th Ed., 2002, p89.
    :param x: x-axis coordinates
    :param n: degrees of freedom
    :return: f(x) - Chi-square probability density function
    """
    k = n/2
    C = 2**k * sp.gamma(k)
    return x**(k-1) * np.exp(-x/2) / C


#
def chi_square_mass(x, n):
    """
    Compute n-dof Chi-square mass function at points x
    Reference: Press, "Numerical Recipes in C", 2nd Ed., 1992, page 221.
    """
    return gammainc(n/2, x/2)


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
    return brentq(f, 0, xu)


# Compute n-sigma ellipse for 2-D Gaussian
def ellipse_sigma(x, P, n=2, Nseg=60):
    L = sl.cholesky(P, lower=True)
    phi = np.linspace(0, 2*np.pi, Nseg)
    return x + n * np.dot(L, [np.cos(phi), np.sin(phi)]).T


#
def ellipse_mass(x, P, prob=0.95, Nseg=60):
    c = np.sqrt(chi_square_bound(prob, 2));
    L = sl.cholesky(P, lower=True)
    phi = np.linspace(0, 2*np.pi, Nseg)
    return x + c * np.dot(L, [np.cos(phi), np.sin(phi)]).T


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
    Equivalent to utilities.dist_sqr(x1, x2) but using Mahalanobis rather than Euclidean distance.
    """
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('Vectors must have the same dimension.')
    dim, n1, n2 = x1.shape[0], x1.shape[1], x2.shape[1]
    v = np.tile(x1, (1,n2))
    v -= np.tile(x2, (n1,1)).reshape(dim, v.shape[1], order='F')
    return dist_mahalanobis(v, S).reshape(n1, n2, order='F')

