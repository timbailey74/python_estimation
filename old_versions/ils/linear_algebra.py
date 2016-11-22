#

import numpy as np
import scipy.linalg as sci

def reprow(x, N):
    # Alternatives: return np.tile(x, (N,1))
    # Alternatives: return np.outer(np.ones(N), x)
    return np.tile(x, (N,1))

def repcol_(x, N):
    return np.outer(x, np.ones(N))

def repcol(x, N):
    if len(x.shape) == 1:
        return np.tile(x, (N,1)).T # for [] arrays
    elif x.shape[1] == 1:
        return np.tile(x, (1,N)) # for [[]] column arrays
    else:
        raise ValueError('Must be an array or single-column matrix')


# Normalised angle calculation; bound angle within +/- pi
def pi2pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

# Multiplication of the form F*P*G
def triprod(F, P, G, order_F_PG=True):
    if order_F_PG:
        return np.dot(F, np.dot(P, G))
    else:
        return np.dot(np.dot(F, P), G)

# Product of diagonal matrix D and full matrix A, where d is diagonal of D.
# If left is True: returns dot(D,A), otherwise, dot(A,D)
def diagprod(d, A, left=True):
    assert len(d.shape) == 1
    if left:  # this works because python broadcasts row-wise
        return (d*A.T).T
    else:
        return d*A

# Symmetric pos-def product of the form F*P*F.T
def symprod(F, P, decompose=True):
    if decompose:  # Cholesky decomposition; guarantees SPD
        L = sci.cholesky(P, lower=True)  # P = L*L.T
        FL = np.dot(F, L)
        return np.dot(FL, FL.T)
    else:  # Simple matrix multiple; guarantee symmetry only
        return force_symmetry(triprod(F, P, F.T))

# Symmetric product D*A*D, where d is diagonal of D.
def symdiagprod(d, A):
    return diagprod(d, diagprod(d, A, True), False)
#    return diagprod(d, diagprod(d, A, False).T, False)  # if A symmetric, this is a tiny bit cheaper (ie., one less transpose)

# Generate indices for the locations in range [0,N) that are not in idx
def index_other(N, idx):
    i = np.ones(N)
    i[idx] = 0
    idy = np.array(range(N))
    return idy[i>0]

# Generate a single index array from a list of slices
def slice2index(args):
    return np.hstack([np.arange(a.start, a.stop, a.step) for a in args])

# Generate a single index array from a list of (start, end) or (start, end, incr) tuples
def tuple2index(args):
    return np.hstack([np.arange(*a) for a in args])

# Force matrix A to be symmetrical by averaging it with its transpose
def force_symmetry(A):
    return 0.5*(A + A.T)

# Obsolete, use sci.block_diag() directly
def blkdiag(*args):
    return sci.block_diag(args)

# Older version, even more obsolete than above
def blkdiag_(*args):
    # From: http://comments.gmane.org/gmane.comp.python.scientific.user/20664
    arrs = [np.asarray(a) for a in args]
    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0))
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r+rr, c:c+cc] = arrs[i]
        r += rr
        c += cc

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


# Use finite differencing to compute the Jacobian of the i-th array in list (x)
def jac_finite_diff(x, model, dmodel=None, i=0, backward=False, offset=1e-7):
    if backward:
        offset = -offset
    if not dmodel:
        dmodel = lambda dy: dy # nominal diff-model
    y = model(*x)
    xi = x[i] # beware, shallow copy
    lenx, leny = len(xi), len(y) # alternatively, len = np.size(x[i]) or len = x[i].size
    J = np.zeros((leny, lenx))
    for j in range(lenx):
        xj = xi[j]
        xo = xj + offset
        xi[j] = xo
        yo = model(*x)
        xi[j] = xj
        J[:,j] = dmodel(y-yo) / (xj-xo)
    return J, y

# Use central differencing scheme to compute the Jacobian of the i-th array in list (x)
def jac_central_diff(x, model, dmodel=None, i=0, offset=1e-7):
    if not dmodel:
        dmodel = lambda dy: dy # nominal diff-model
    y = model(*x)
    xi = x[i]               # beware, shallow copy
    lenx, leny = len(xi), len(y)
    J = np.zeros((leny, lenx))  # note, tuple input
    for j in range(lenx):
        xj = xi[j]          # record old value
        xu = xj + offset
        xl = xj - offset
        xi[j] = xu          # compute upper
        yu = model(*x)
        xi[j] = xl          # compute lower
        yl = model(*x)
        xi[j] = xj          # recover old value
        J[:,j] = dmodel(yu-yl) / (xu-xl)
    return J, y

# Class: Simplified Jacobian interface that behaves like autograd.jacobian
class jacobian_class:
    def __init__(self, model, i=0, central_diff=False, dmodel=None, *args, **kwargs):
        self.model = model
        self.dmodel = dmodel
        self.i = i
        self.numjac = jac_central_diff if central_diff else jac_finite_diff
        self.args, self.kwargs = args, kwargs
    def __call__(self, *args):
        J, self.value = self.numjac(args, self.model, self.dmodel, self.i, *self.args, **self.kwargs)
        return J
    # Notes:
    # 1. Can access function value as: my_jacobian.value
    # 2. Can change my_jacobian.i to differentiate wrt different terms

# Closure: Simplified Jacobian interface that behaves like autograd.jacobian
def jacobian_function(model, i=0, central_diff=False, dmodel=None, *args, **kwargs):
    numjac = jac_central_diff if central_diff else jac_finite_diff
    def evaluate(*args_f):
        nonlocal numjac
        J, _ = numjac(args_f, model, dmodel, i, *args, **kwargs)
        return J
    return evaluate

# Alias to make the closure version the default (ie., to have same name as autograd version)
jacobian = jacobian_function


#
# TESTS ------------------------------------------------------
#

def test_numerical_jacobians():
    def rbmodel(xv, xt):
        # range-bearing model, with xv=(x,y,p), xt=(x,y)
        dx = xt[0]-xv[0]
        dy = xt[1]-xv[1]
        return np.array(
            [np.sqrt(dx**2 + dy**2),        # range
             np.arctan2(dy, dx) - xv[2]])   # bearing

    def rbdiff(e):
        # model diff for range-bearing; Caution: pass-by-reference
        e[1] = pi2pi(e[1])
        return e

    xv = np.array([3, -5., 2])  # note, must be floating-point array
    xt = np.array([2., 1])
    x = [xv, xt]
    Hv, z = jac_central_diff(x, rbmodel, rbdiff, i=0)
    Ht, _ = jac_central_diff(x, rbmodel, rbdiff, i=1)
    #print("z = ", z, "\nHv =\n", Hv, "\nHt =\n", Ht)
    print('z = \n{0}\nHv = \n{1}\nHt = \n{2}'.format(z, Hv, Ht))

def test_numerical_jacobians_2():
    #x = np.array([0., 1, 2])
    x = [0, 1, 2]
    J, y = jac_finite_diff([x], np.sin)
    print('\n\nx = {0}\ny = {1}\nJ = \n{2}'.format(x, y, J))

def test_pi2pi():
    x = 6.7
    xl = [-32.1, -6.7, -1.2, 1.2, 6.7, 32.1]
    xa = 30*(np.random.rand(1,5)-0.5)
    print("x = ", x, "\np = ", pi2pi(x),\
          "\nxa = ", xa, "\npa = ", pi2pi(xa), "\nda = ", xa-pi2pi(xa), "\n")

#
#
#

if __name__ == "__main__":
    flags = 0xff # test all
    #flags = 0x1
    if flags & 0x01:
        print('test_numerical_jacobians()')
        test_numerical_jacobians()
        test_numerical_jacobians_2()
    if flags & 0x02:
        print('test_pi2pi()')
        test_pi2pi()
