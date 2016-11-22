# Convention: if x is a matrix of d-dimensional row-vectors
#   Exceptions: linalg.eigh gives e-vecs as col-vectors
# FIXME: change operations to either row-array form, or user-specified via axis

import numpy as np
import scipy.linalg as sl
from scipy.special import gammainc
from scipy.optimize import brentq, fminbound


# Compute sample mean and covariance, accounting for missing data
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


# Old incorrect version....
def sample_mean_cov_(x, axis=0):
    try:  # assume is masked array
        mask = 1 - x.mask.astype(int)  # want False -> 1, True -> 0
        x = x.data.copy()
        x[mask==0] = 0
    except:  # otherwise, ordinary array, no missing data
        mask = np.ones(x.shape)
    Nx = np.sum(mask, axis)  # per-element counter for mean
    xm = np.sum(x, axis) / Nx  # mean
    # Calculating covariance, the sum of outer-products depends on axis
    if axis:  # column vectors
        P = np.dot(x, x.T) / np.dot(mask, mask.T)
    else:  # row vectors
        P = np.dot(x.T, x) / np.dot(mask.T, mask)
    P -= np.outer(xm, xm)
    return xm, P


# Mean and covariance of weighted samples
def sample_mean_weighted(x, w, axis=0):
    if axis:
        x = x.T
    xm = np.sum(w*x.T, 1)  # mean
    xc = x - xm  # zero-centred samples
    P = np.dot(w*xc.T, xc)  # covariance
    return xm, P


# Generate indices for the locations in range [0,N) that are not in idx
def index_other(N, idx):
    i = np.ones(N)
    i[idx] = 0
    idy = np.arange(N)
    return idy[i!=0]


# Multiplication of the form F*P*G
def triprod(F, P, G):
    return np.dot(np.dot(F, P), G)


# Multiplication of the form F*P*F.T
def symprod(F, P):
    return triprod(F, P, F.T)


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
    #import IPython; IPython.embed()
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


# Compute log(sum(w)) given set of log(w), with improved numerics 
def log_sum(logw):
    c = np.max(logw, 0) 
    shifted_exp = np.exp(logw - c)
    shifted_sum = np.sum(shifted_exp, 0)
    return c + np.log(shifted_sum) 


#
class pca:
    """
    Note: if diagonalise is not whitened, the covariance is np.diag(d[-M:]).

    Attributes
    ----------
    xm : 
    P : np array
    d :
    E :
    """
    def __init__(self, x, axis=0, whiten=False):
        self.xm, self.P = sample_mean_cov(x, axis)
        self.d, self.E = sl.eigh(self.P)  # d is in ascending order
        self.axis = axis        
        self.whiten = whiten
    def diagonalise(self, x, M):
        """Project to decorrelated subspace 
        """
        Es = self.E[:, -M:]
        if self.whiten:
            Es = Es / np.sqrt(self.d[-M:])  # don't use /= because that would mutate self.E
        if self.axis==0:
            return np.dot(x - self.xm, Es)
        else:
            return np.dot(Es.T, (x.T - self.xm).T)  # or np.dot(x.T - self.xm, Es).T
    def undiagonalise(self, x):
        """Reproject to original space, but flattened onto hyper-plane
        """
        M = x.shape[1 - self.axis]
        Es = self.E[:, -M:]
        if self.whiten:
            Es = Es * np.sqrt(self.d[-M:])  # don't use *= because that would mutate self.E
        if self.axis==0:
            return np.dot(x, Es.T) + self.xm
        else:
            return (np.dot(Es, x).T + self.xm).T
    def project_hyperplane(self, x, M):
        return self.undiagonalise(self.diagonalise(x, M))


#
class impute:
    def __init__(self, x, axis=0, min_eval=1e-8):
        self.xm, self.P = sample_mean_cov(x, axis)
        # Determine projection to full-rank subspace
        d, E = sl.eigh(self.P)
        i = next(i for i,di in enumerate(d) if di>min_eval)
        self.E = E[:, i:]  # projection operator
        self.Psub = np.diag(d[i:])  # subspace covariance
        self.xsub = np.zeros(self.E.shape[1])  # subspace mean
    def impute(self, x):
        assert len(x.shape) == 1  # must be single vector
        i = np.arange(len(x))[x.mask==False]  # index of non-missing values
        try:  # attempt full rank solve
            xresult, Presult = gaussian_conditional(self.xm, self.P, x[i], i)
        except:  # rank-deficient case, attempt pca projection
            # Compute mean and covariance of conditioning term
            xc = x.data.copy()
            xc[x.mask==True] = self.xm[x.mask==True]  # optional step (nominal missing values)
            Pc = np.ones(len(x)) * 1e8  # approximate infinite variance
            Pc[x.mask==False] = 0  # non-missing terms are perfectly known
            Pc = np.diag(Pc)
            # Project conditioning term to subspace
            xcs = np.dot(xc - self.xm, self.E)
            Pcs = symprod(self.E.T, Pc)
            # Multiply to fuse information
            xp, Pp = gaussian_product(self.xsub, self.Psub, xcs, Pcs)
            # Reproject to original space
            xpr = np.dot(xp, self.E.T) + self.xm
            Ppr = symprod(self.E, Pp)
            # Take marginal over missing-values
            xresult = xpr[x.mask==True]
            Presult = Ppr[np.ix_(x.mask==True, x.mask==True)]
            #import IPython; IPython.embed()
        return xresult, Presult


#
# Tests ----
#

def test_generate_data(D=3, N=20):
    # Generate 'truth' pdf
    xm = np.zeros(D)
    P = 2*(np.random.rand(D,D)-1)
    P = np.dot(P, P.T)
    # Generate samples, with random masked bits
    x = np.random.multivariate_normal(xm, P, N)
    return np.ma.masked_where(x>2, x) 

# If this test crashes, run it again...
def test_full_rank_impute(x = test_generate_data()):
    # Fit Gaussian to ensemble, accounting for missing data
    xm_e, P_e = sample_mean_cov(x)  
    # Select a sample with missing data
    xs = next(xi for xi in x if np.any(xi.mask) and not np.all(xi.mask))
    # Do impute
    idx = np.arange(D)[xs.mask==False]  # index of non-missing values
    xc, Pc = gaussian_conditional(xm_e, P_e, xs[idx], idx)
    # Print results
    print('Estimated mean: \n{}'.format(xm_e))
    print('Estimated covariance: \n{}'.format(P_e))
    print('Sample with missing data: \n{}'.format(xs))
    print('Mean fill value: \n{}'.format(xc))
    print('Covariance fill value: \n{}'.format(Pc))
    print('Samples of fill values:\n{}'.format(np.random.multivariate_normal(xc, Pc, 5)))
    
    #import IPython; IPython.embed()


#
def test_general_impute(x = test_generate_data()):
    imputer = impute(x)
    # Select a sample with missing data
    xs = next(xi for xi in x if np.any(xi.mask) and not np.all(xi.mask))
    # Do impute
    xc, Pc = imputer.impute(xs)    
    # Print results
    print('Sample with missing data: \n{}'.format(xs))
    print('Mean fill value: \n{}'.format(xc))
    print('Covariance fill value: \n{}'.format(Pc))
    print('Samples of fill values:\n{}'.format(np.random.multivariate_normal(xc, Pc, 5)))


def is_equal(a, b):
    return np.all(np.abs(a-b) < 1e-10)

# x is a single (D,) vector, or a (N,D) matrix of row vectors
# vals is either (len(i),), or (N,len(i)), or (N,) in 1-D case
# FIXME: remove duplication from test code
def test_gaussian_conditional(N, D, i=[1], logw=True):
    x = np.random.rand(N, D)
    P = np.random.randn(D, D)
    P = np.dot(P, P.T)
    vals = np.random.rand(N, len(i))
    #
    xc,Pc,wc = gaussian_conditional(x, P, vals, i, logw)
    for j, (xj, vj) in enumerate(zip(x, vals)):
        a,b,c = gaussian_conditional(xj, P, vj, i, logw)
        assert is_equal(a,xc[j,:]) and is_equal(b,Pc) and is_equal(c,wc[j])
    #   
    xsingle = x[0,:]
    xs,Ps,ws = gaussian_conditional(xsingle, P, vals, i, logw)
    for j, vj in enumerate(vals):
        a,b,c = gaussian_conditional(xsingle, P, vj, i, logw)
        assert is_equal(a,xs[j,:]) and is_equal(b,Ps) and is_equal(c,ws[j])
    #
    valsingle = vals[0,:]
    xv,Pv,wv = gaussian_conditional(x, P, valsingle, i, logw)
    for j, xj in enumerate(x):
        a,b,c = gaussian_conditional(xj, P, valsingle, i, logw)
        assert is_equal(a,xv[j,:]) and is_equal(b,Pv) and is_equal(c,wv[j])
    #
    if len(i) == 1:
        val1 = np.random.rand(N)
        xx,PP,ww = gaussian_conditional(x, P, val1, i, logw)
        for j, (xj, vj) in enumerate(zip(x, val1)):
            a,b,c = gaussian_conditional(xj, P, vj, i, logw)
            assert is_equal(a,xx[j,:]) and is_equal(b,PP) and is_equal(c,ww[j])
        xy,Py,wy = gaussian_conditional(xsingle, P, val1, i, logw)
        for j, vj in enumerate(val1):
            a,b,c = gaussian_conditional(xsingle, P, vj, i, logw)
            assert is_equal(a,xy[j,:]) and is_equal(b,Py) and is_equal(c,wy[j])
    #import IPython; IPython.embed()
    print('Success')

#   x is (2,) or (N,2); and vals is (N,)
def test_gaussian_conditional_2d(N, i=1, logw=True):
    D = 2
    x = np.random.rand(N, D)
    P = np.random.randn(D, D)
    P = np.dot(P, P.T)
    vals = np.random.rand(N)
    #
    a,b,c = gaussian_conditional(x, P, vals, [i], logw)
    d,e,f = gaussian_conditional_2d(x, P, vals, i, logw)
    assert is_equal(a,d) and is_equal(b,e) and is_equal(c,f)
    #
    xsingle = x[0,:]
    a,b,c = gaussian_conditional(xsingle, P, vals, [i], logw)
    d,e,f = gaussian_conditional_2d(xsingle, P, vals, i, logw)
    assert is_equal(a,d) and is_equal(b,e) and is_equal(c,f)
    print('Success')
    #import IPython; IPython.embed()
