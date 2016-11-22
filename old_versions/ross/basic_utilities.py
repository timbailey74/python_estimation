#
# Basic utilites
#   - Simple file-name stuff
#   - Gaussian manipulation and inference utilities. Ported from matlab_utilities. Tim Bailey 2015.
#   - etc
#

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as slin


def join_path(base, d):
    return os.path.normpath(os.path.join(base, d))

def get_all_filenames(dir):
    return [n for n in os.listdir(dir) if os.path.isfile(join_path(dir, n))]

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

# Compute mean and covariance from set of samples
def sample_mean(x):
    N = x.shape[1]
    xm = np.sum(x, 1) / N
    xc = x - repcol(xm, N)
    P = np.dot(xc, xc.T) / N
    return xm, P

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


#
def deg2rad(degrees, normalise=True):
    radians = degrees * np.pi / 180.0
    return pi2pi(radians) if normalise else radians

def rad2deg(radians):
    return radians * 180.0 / np.pi

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

# Sliding window polynomial fit; deals with end-effects by truncating number of points in polynomial
def poly_smooth_old(t, x, deg, window):
    assert deg < 1 + 2*window
    N = len(t)
    xs = np.zeros(N)
    for i in range(N):
        if i < window: idx = slice(0, 1+max(deg, i+window))
        elif i >= N-window: idx = slice(min(N-1-deg, i-window), N)
        else: idx = slice(i-window, i+window+1)
        #print(i,idx)
        p = np.polyfit(t[idx], x[idx], deg=deg)
        xs_idx = np.polyval(p, t[idx])
        if i < window: xs[i] = xs_idx[i]
        elif i >= N-window: xs[i] = xs_idx[i-N]
        else: xs[i] = xs_idx[window]
    return xs

# Sliding window polynomial fit. If a point appears in multiple polynomials, average them
def poly_smooth_av(t, x, deg, window):
    assert deg < 1 + 2*window
    N = len(t)
    xs = np.zeros(N)
    ns = np.zeros(N)
    for i in range(window, N-window):
        idx = slice(i-window, i+window+1)
        p = np.polyfit(t[idx], x[idx], deg=deg)
        xs[idx] += np.polyval(p, t[idx])
        ns[idx] += 1
    xs /= ns  # average entries that had multiple polys
    return xs

# Sliding window polynomial fit; Savitzky–Golay filter
# FIXME: perhaps return gradient for each point also (if, requested)
# FIXME: perhaps allow for evaluation at different points (ie., interpolation)
def poly_smooth(t, x, deg, window):
    N = len(t)
    assert deg < 1 + 2*window
    assert N > 2*window
    xs = np.zeros(N)
    for i in range(window, N-window):
        idx = slice(i-window, i+window+1)
        p = np.polyfit(t[idx], x[idx], deg=deg)
        xpoly = np.polyval(p, t[idx])
        xs[i] = xpoly[window]
        if i == window:  # start condition; use first poly to fit the initial points
            xs[:i] = xpoly[:window]
        if i == N-window-1:  # end condition; use last poly to fit the final points
            xs[i:] = xpoly[window:]
    return xs

# Mean and std-dev with removal of outliers greater than s-stddevs from mean. Can substitute median for mean.
def robust_mean(x, s, mid_func=np.mean):
    i = x==x  # true for everything except nans
    while True:
        mid, stddev = mid_func(x[i]), np.std(x[i])
        j = abs(x - mid) <= s*stddev  # use <= to account for std==0
        if np.all(i==j):
            break
        i = j
    return mid, stddev, i

# 3D plots
def plot3(x, y, z, form='-', title = '', ax=None):
    if not ax:
        #ax = plt.gcf().add_subplot(111, projection='3d')
        ax = plt.gca(projection='3d')
    ax.plot(x, y, z, form)
    return ax

def set_axes_equal(ax=None):
    # Set equal-axes for 3D plots. Adapted from:
    #http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    if ax is None:
        ax = plt.gca(projection='3d')
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    radius = 0.5 * max(np.diff((xlim, ylim, zlim)))
    bound = [-radius, radius]
    ax.set_xlim3d(bound + np.mean(xlim))
    ax.set_ylim3d(bound + np.mean(ylim))
    ax.set_zlim3d(bound + np.mean(zlim))

def line_plot_conversion(a, b):
    # Assumes points are matrix of row-vectors; FIXME: do I prefer column-vectors?
    M, N = a.shape
    nans = np.tile(np.nan*np.zeros(N), (M,1))
    block = np.hstack((a, b, nans))
    flat = block.flatten('C')
    return flat.reshape(M*3, N)

def annotate_angled(text, xy, angle=0, ax=None):
    if ax is None:
        ax = plt
    an = ax.annotate(text, xy=xy, horizontalalignment='left', verticalalignment='bottom')
    an.set_rotation(angle)
    return an


#
# TEST CODE ------------------------------------------------------------------
#

def test_polysmooth():
    t = np.arange(10)
    x = np.sin(t)
    xs2 = poly_smooth(t, x, 1, 1)
    xs3 = poly_smooth(t, x, 3, 2)
    plt.plot(t, x, t, xs2, t, xs3)

if __name__ == "__main__":
    test_polysmooth()

plt.show()
