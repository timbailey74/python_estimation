# Basic iterative least-squares operations.
# Tim Bailey, 2015

import numpy as np
import scipy.linalg as sci

# Normalised angle calculation; bound angle within +/- pi
def pi2pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

# Multiplication of the form F*P*G
def triprod(F, P, G):
    return np.dot(F, np.dot(P, G))

# Generate indices for the locations in range [0,N) that are not in idx
def index_other(N, idx):
    i = np.ones(N)
    i[idx] = 0
    idy = np.array(range(N))
    return idy[i>0]

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

# Use finite differencing to compute the Jacobian of the i-th array in list (x)
def jac_finite_diff(x, model, dmodel=None, i=0, backward = False, offset = 1e-9):
    if backward: offset = -offset
    if not dmodel: dmodel = lambda dy: dy # nominal diff-model
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
def jac_central_diff(x, model, dmodel=None, i=0, offset = 1e-9):
    if not dmodel: dmodel = lambda dy: dy # nominal diff-model
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

# Simple information-form update step (ie., fusion step).
# WARNING: Does not cater for discontinuous functions.
def information_form_update(y, Y, z, R, xs, zs, Hs, idx, logflag=-1):
    if logflag != -1:
        (v,S) = information_form_innovation(y, Y, z, R, xs, zs, Hs, idx)
        w = gauss_evaluate(v, S, logflag)
    Ri = np.linalg.inv(R) # FIXME: Expensive if R is diagonal
    HtRi = np.dot(Hs.T, Ri)
    Y[np.ix_(idx,idx)] += force_symmetry(np.dot(HtRi, Hs)) # Note: requires np.ix_() to work correctly
    y[idx] += np.dot(HtRi, (z - zs + np.dot(Hs,xs)))
    if logflag != -1:
        return w

# Prediction is augment and marginalise simultaneously
def information_form_prediction(y, Y, q, Q, F, idx):
    pass

def information_form_prediction_linearised(y, Y, q, Q, fs, xs, qs, F, G, idx):
    pass

# Constrain and marginalise simultaneously; we assume the constraint is z = h(x) + r = 0
def information_form_constrain_and_marginalise(y, Y, R, xs, zs, Hs, iremove):
    """
    R - additive noise
    """
    # Split state into parts we want to keep and parts we want to remove (ie., marginalise away)
    # Note: requires np.ix_() to work correctly
    ikeep = index_other(y.size, iremove)
    Hr = Hs[:, iremove]
    Hk = Hs[:, ikeep]
    Ykr = Y[np.ix_(ikeep, iremove)]
    Yri = np.linalg.inv(Y[np.ix_(iremove,iremove)])
    # Compute values that will be used several times
    HrYri = np.dot(Hr, Yri)
    Gi = np.linalg.inv(np.dot(HrYri, Hr.T) + R)
    YkrYri = np.dot(Ykr, Yri)
    S = Hk.T - np.dot(YkrYri, Hr.T)
    SGi = np.dot(S, Gi)
    vv = np.dot(Hs, xs) - zs - np.dot(HrYri, y[iremove])
    # Constrain and marginalise
    yu = y[ikeep] - np.dot(YkrYri, y[iremove]) + np.dot(SGi, vv)
    Yu = Y[np.ix_(ikeep,ikeep)] - np.dot(YkrYri, Ykr.T) + np.dot(SGi, S.T)
    return (yu, Yu)

# Information-form augment; linear models
def information_form_augment(y, Y, q, Q, F, idx):
    """
    idx indexes the subset of existing states, such that xnew = F*x[idx] + q
    """
    Qi = np.linalg.inv(Q)
    FtQi = np.dot(F.T, Qi)
    ya = np.append(y, np.dot(Qi, q))
    ya[idx] -= np.dot(FtQi, q)
    Yxn = np.zeros((y.size, q.size))
    Yxn[idx,:] = -FtQi
    Ya = np.vstack((np.hstack((Y, Yxn)), np.hstack((Yxn.T, Qi))))
    Ya[np.ix_(idx,idx)] += force_symmetry(np.dot(FtQi, F.T))
    return (ya, Ya)

# Information-form augment; nonlinear models
def information_form_augment_linearised(y, Y, q, Q, fs, xs, qs, F, G, idx):
    """
    xs is a linearisation vector for the states x[idx]
    fs = f(xs,qs)
    F is df/dx and G is df/dq
    """
    # FIXME: Perform tests for discontinuities in x-xs (expensive) and q-qs (cheap)
    qlin = fs - np.dot(F, xs) + np.dot(G, q-qs)
    Qlin = triprod(G, Q, G.T)
    return information_form_augment(y, Y, qlin, Qlin, F, idx)

# Information-form marginalisation
def information_form_marginalise(y, Y, idx):
    idy = index_other(y.size, idx)
    F = Y[np.ix_(idx, idy)]
    B = np.linalg.inv(Y[np.ix_(idy, idy)])
    FB = np.dot(F, B)
    Ym = Y[np.ix_(idx,idx)] - force_symmetry(np.dot(FB, F.T))
    ym = y[idx] - np.dot(FB, y[idy])
    return (ym, Ym)

# Information-form to moment-form conversion
def information2moment(y, Y, idx=None):
    # TODO: Expensive. Only need P[idx,idx]. Implement sparse solution.
    L = sci.cho_factor(Y)
    x = sci.cho_solve(L, y)
    P = np.linalg.inv(Y)
    return (x[idx], P[np.ix_(idx,idx)]) if idx else (x, P)

# Information-form innovation calculation
def information_form_innovation(y, Y, z, R, xs, zs, Hs, idx):
    (x, P) = information2moment(y, Y, idx)
    S = triprod(Hs, P, Hs.T) + R
    zhat = zs + np.dot(Hs, x - xs)
    return (z-zhat, S)

# Evaluate Gaussian pdf N(x,P) given L*L.T = P and L*f = x
def gauss_evaluate_sqrtform(f, L, logw=False):
    D = f.size
    E = -0.5 * np.sum(f*f, axis=0)
    if logw:
        C = 0.5*D*np.log(2*np.pi) + np.sum(np.log(L.diagonal()))
        w = E - C
    else:
        C = (2*np.pi)**(D/2.) * np.prod(L.diagonal())
        w = np.exp(E) / C
    return w

# v is an array and S is either scalar or an array (representing matrix diagonal)
def gauss_evaluate_scalar(v, S, logw=False):
    pass

# Evaluate a Gaussian with covariance S at distance v from the mean
def gauss_evaluate(v, S, logw=False):
    # FIXME: what if v or S are scalars?
    L = sci.cholesky(S, lower=True)
    f = sci.solve_triangular(L, v, lower=True)
    return gauss_evaluate_sqrtform(f, L, logw)

# Linearised innovation with compensation for discontinuities in z-zpred (but not in x-xs)
def moment_form_innovation(x, model, norm, z, xs, Hs, idx, args):
    zpred = model(xs, *args) + np.dot(Hs, x[idx] - xs)
    return norm(z - zpred)

# Kalman update
def moment_form_update(x, P, v, R, Hs, idx=None, logflag=None):
    P1, P2 = P, P
    if idx != None:
        P1 = P[np.ix_(idx,idx)] # square block (idx x idx)
        P2 = P[:, idx] # rectangular block (idx columns)
    # P1, P2 = (P[np.ix_(idx,idx)], P[:, idx]) if idx != None else (P, P)
    S = triprod(Hs, P1, Hs.T) + R
    L = sci.cholesky(S, lower=True)
    Li = np.linalg.inv(L)
    #Li = sci.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    vc = np.dot(Li, v)
    Wc = triprod(P2, Hs.T, Li.T)
    x += np.dot(Wc, vc)
    P -= np.dot(Wc, Wc.T)
    if logflag != None:
        w = gauss_evaluate_sqrtform(vc, L, logflag)
        return x, P, w
    else:
        return x, P

# Obsolete function. Using sci.solve(A, b) is much more accurate.
# Solve A*x = b using the normal equations. Assumes input of At = A.T instead of A
def normal_equation_solve_linear(At, b):
    from scipy.linalg import cho_factor, cho_solve
    Y = np.dot(At, At.T)
    #c = np.linalg.cond(Y)
    L = cho_factor(Y)
    return cho_solve(L, np.dot(At, b))

# Simple Least-squares solver.
# WARNING: This is NOT solving H*x = z.
# It assumes the form: z = h(x) = 0, such that x = inv(Ht*H)*Ht(0 - h(xs) + H*xs)
# = inv(Ht*H)*Ht(H*xs - zs). It assumes implicitly that R = I
def normal_equation_solve(xs, zs, H, type=0):
    """Least-squares solve using the normal equations"""
    v = np.dot(H, xs) - zs
    y = np.dot(H.T, v)
    Y = np.dot(H.T, H)
    if type == 0:
        L = sci.cho_factor(Y) # CAUTION, not triangular (off-triangle side filled with random values, see SciPy docs)
        x = sci.cho_solve(L, y)
    elif type == 1:
        L = np.linalg.cholesky(Y)
        x = sci.cho_solve((L, True), y)
    elif type == 2:
        x = sci.solve(Y, y)
    else:
        L = sci.cholesky(Y, lower=True)              # or L = np.linalg.cholesky(Y)
        f = sci.solve_triangular(L, y, lower=True)   # or f = sp.solve(L, y), using lower = False or True makes no difference
        x = sci.solve_triangular(L.T, f, lower=False) # lower=False is default
    return x

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

def test_gauss_evaluate():
    S = np.random.uniform(-3,3,(3,3))
    S = np.dot(S.T, S)
    v1 = np.random.uniform(-1,1,(3,4))
    v2 = v1[:,0]
    print(gauss_evaluate(v1, S))
    print(gauss_evaluate(v1, S, True))
    print(gauss_evaluate(v2, S, False))
    print(gauss_evaluate(v2, S, True))

def test_information_form_ops():
    pass

def test_constrain_marginalise():
    # Check that simultaneous constrain-marginalise algorithm produces same result as
    # sequential constrain then marginalise.
    def randm(M,N,K):
        x = np.random.uniform(-K, K, (M,N))
        if M == N:
            x = np.dot(x, x.T)
        return x
    Nx = 40
    Nz = 20
    Nrem = 35

    # Make random data
    iremove = list(set(np.random.randint(0, Nx, Nrem)))
    xs = randm(Nx,1,5)
    y = randm(Nx,1,5)
    Y = randm(Nx,Nx,3)
    R = randm(Nz,Nz,2)
    zs = randm(Nz,1,2)
    Hs = np.random.uniform(-3,3, (Nz,Nx))

    # Simultaneous constrain-marginalise
    (yu, Yu) = information_form_constrain_and_marginalise(y, Y, R, xs, zs, Hs, iremove)

    # Compare against sequence of constrain then marginalise
    yc = y.copy()
    Yc = Y.copy()
    zz = np.zeros((zs.size, 1))
    information_form_update(yc, Yc, zz, R, xs, zs, Hs, np.array(range(yc.size)))
    (ym, Ym) = information_form_marginalise(yc, Yc, index_other(yc.size, iremove))
    print('Yu: ', 1 / np.linalg.cond(Yu))
    print('Ym: ', 1 / np.linalg.cond(Ym))
    print('Yc: ', 1 / np.linalg.cond(Yc))
    import matplotlib.pyplot as plt
    plt.plot(ym)
    plt.plot(yu)
    plt.figure()
    plt.plot(ym-yu)
    plt.figure()
    plt.plot(Ym.flatten()-Yu.flatten())
    plt.show()

#
#
#

if __name__ == "__main__":
    flags = 0xff # test all
    flags = 0x1
    if flags & 0x01:
        print('test_numerical_jacobians()')
        test_numerical_jacobians()
        test_numerical_jacobians_2()
    if flags & 0x02:
        print('test_pi2pi()')
        test_pi2pi()
    if flags & 0x04:
        print('test_gauss_evaluate()')
        test_gauss_evaluate()
    if flags & 0x08:
        print('test_information_form_ops()')
        test_information_form_ops()
    if flags & 0x10:
        print('test_constrain_marginalise()')
        test_constrain_marginalise()
