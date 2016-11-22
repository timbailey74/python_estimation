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

# Symmetric pos-def product of the form F*P*F.T
def symprod(F, P):
    L = sci.cholesky(P, lower=True)  # P = L*L.T
    FL = np.dot(F, L)
    return np.dot(FL, FL.T)

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

# Use finite differencing to compute the Jacobian of the i-th array in list (x)
def jac_finite_diff(x, model, dmodel=None, i=0, backward=False, offset=1e-9):
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
def jac_central_diff(x, model, dmodel=None, i=0, offset=1e-9):
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

# Simple information-form update step (ie., fusion step). Modification of
# (y, Y) performed in-place, so they are not returned from function.
# WARNING: Does not cater for discontinuous functions.
def information_form_update_old(y, Y, z, R, xs, zs, Hs, idx, logflag=-1):
    if logflag != -1:
        (v,S) = information_form_innovation(y, Y, z, R, xs, zs, Hs, idx)
        w = gauss_evaluate(v, S, logflag)
    Ri = np.linalg.inv(R) # FIXME: Expensive if R is diagonal
    HtRi = np.dot(Hs.T, Ri)
    Y[np.ix_(idx,idx)] += force_symmetry(np.dot(HtRi, Hs)) # Note: requires np.ix_() to work correctly
    y[idx] += np.dot(HtRi, (z - zs + np.dot(Hs,xs)))
    if logflag != -1:
        return w

# WARNING: Does not cater for discontinuous functions.
def information_form_update(y, Y, z, R, xs, zs, rs, Hx, Hr, idx, logflag=-1):
    Ra = triprod(Hr, R, Hr.T)
    if logflag != -1:  # FIXME: expensive information2moment conversion
        x, P = information2moment(y, Y, idx)
        v = moment_form_innovation(x, z, xs, rs, zs, Hx, Hr)
        S = triprod(Hx, P, Hx.T) + Ra
        w = gauss_evaluate(v, S, logflag)
    Ri = np.linalg.inv(Ra) # FIXME: Expensive if Ra is diagonal
    HtRi = np.dot(Hx.T, Ri)
    Y[np.ix_(idx,idx)] += force_symmetry(np.dot(HtRi, Hx)) # Note: requires np.ix_() to work correctly
    y[idx] += np.dot(HtRi, (z - zs + np.dot(Hx,xs) + np.dot(Hr,rs)))
    if logflag != -1:
        return w


# Prediction is augment and marginalise simultaneously
# For formulae, see Maybeck Vol1, Section , pp.
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
    Ya[np.ix_(idx,idx)] += force_symmetry(np.dot(FtQi, F))
    return ya, Ya

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
    return ym, Ym

# Information-form to moment-form conversion
def information2moment(y, Y, idx=None):
    # FIXME: Expensive. Only need P[idx,idx]. Implement sparse solution.
    L = sci.cho_factor(Y)
    x = sci.cho_solve(L, y)  # FIXME: Is this implementation numerically accurate?
    P = np.linalg.inv(Y)
    return (x, P) if idx is None else (x[idx], P[np.ix_(idx,idx)])

# Information-form innovation calculation; Expensive: converts to moment-form and uses its innovation
# FIXME: this interface doesn't account for rs or Hr
def information_form_innovation(y, Y, z, R, xs, zs, Hs, idx=None):
    (x, P) = information2moment(y, Y, idx)
    S = triprod(Hs, P, Hs.T) + R
    zhat = zs + np.dot(Hs, x - xs)
    return (z-zhat, S)

# Evaluate mean of (marginalised) noise estimate (r_hat), given mean of state (x) and measurement (z).
# This is possible because all uncertainty is due to (x, r), so their joint state is fully correlated
# (ie., rank deficient) such that z_measured - z_predicted = 0.
def evaluate_noise_mean(x, z, xs, zs, rs, Hx, Hr):
    # Note: if Hr is diagonal, pass as an array; eg, np.diag(Hr)
    # Note2: if applying to a process model, let z=x_(k+1), rs=qs, Hr=Fq
    # Note3: if Hr is singular, then part or all of r_hat is unobservable
    numerator = z - zs - np.dot(Hx, x-xs)
    if len(Hr.shape) == 1:  # Hr is diagonal
        rdiff = numerator / Hr
    else:  # Hr is square
        rdiff = np.dot(sci.inv(Hr), numerator)  # FIXME: perhaps better numerics to compute by decomposition and solve, rather than inversion
    return rdiff + rs

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
#def moment_form_innovation(x, model, norm, z, xs, Hs, idx, args):
#    zpred = model(xs, *args) + np.dot(Hs, x[idx] - xs)
#    return norm(z - zpred)

# Linearised innovation with compensation for discontinuities in z-zpred (but not in x-xs)
def moment_form_innovation(x, z, xs, rs, zs, Hx, Hr, idx=None, norm=None):
    # Note: zs = h(xs, rs), and we assume r-prior is zero-mean
    if idx is not None:
        x = x[idx]
    zpred = zs + np.dot(Hx, x - xs) - np.dot(Hr, rs)  # note: Hr*(r - rs) == -Hr*rs
    v = z - zpred
    if norm is not None:
        v = norm(v)
    return v

# Kalman update; changes (x, P) in-place, so not returned by function
def moment_form_update(x, P, v, R, Hs, idx=None, logflag=None):
    P1, P2 = (P[np.ix_(idx,idx)], P[:,idx]) if idx is not None else (P,P)
    S = triprod(Hs, P1, Hs.T) + R
    L = sci.cholesky(S, lower=True)
    Li = np.linalg.inv(L)
    #Li = sci.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    vc = np.dot(Li, v)
    Wc = triprod(P2, Hs.T, Li.T)
    x += np.dot(Wc, vc)
    P -= np.dot(Wc, Wc.T)
    if logflag is not None:
        return gauss_evaluate_sqrtform(vc, L, logflag)

# TODO: We can do efficient sqrt-root update when R is diagonal
def moment_form_update_diagonal_noise():
    pass

# Marginalisation; trivial
def moment_form_marginalise(x, P, idx):
    return x[idx].copy(), P[np.ix_(idx,idx)].copy()

# Augmentation
def moment_form_augment(x, P, q, Q, F, idx=None):
    if idx is None:
        P1, P2, x1 = P, P, x
    else:
        P1 = P[np.ix_(idx,idx)]  # square block (idx x idx)
        P2 = P[idx, :]  # rectangular block (idx columns)
        x1 = x[idx]
    xn = np.dot(F, x1) + q  # new augmented sub-vector
    xa = np.append(x, xn)
    Pnn = triprod(F, P1, F.T) + Q  # new block-diagonal
    Pnn = force_symmetry(Pnn)  # FIXME: should properly ensure Pnn is symmetric pos-def
    Pno = np.dot(F, P2)  # new-old off-diagonal block
    Pa = np.vstack((np.hstack((P, Pno.T)), np.hstack((Pno, Pnn))))
    return xa, Pa

# Linearised augmentation
def moment_form_augment_linearised(x, P, q, Q, fs, xs, qs, F, G, idx=None):
    # fs = f(xs, qs, ...)
    # FIXME: check x-xs and q-qs do not have discontinuities
    # FIXME: qlin, Qlin calculation same as for info-form; refactor
    qlin = fs - np.dot(F, xs) + np.dot(G, q-qs)
    Qlin = triprod(G, Q, G.T)
    return moment_form_augment(x, P, qlin, Qlin, F, idx)

# Partial implementation of the two prediction functions below
def _augment2predict(xa, Pa, x, iremove):
    if iremove is None:  # remove all of previous state, keep only new states
        ikeep = np.arange(len(x), len(xa))
    else:  # keep some of old states
        ikeep = index_other(len(xa), iremove)
    return moment_form_marginalise(xa, Pa, ikeep)

def moment_form_predict(x, P, q, Q, F, ifunc=None, iremove=None):
    # Slow (but simple) version; augment then marginalise
    xa, Pa = moment_form_augment(x, P, q, Q, F, ifunc)
    return _augment2predict(xa, Pa, x, iremove)

def moment_form_predict_linearised(x, P, q, Q, fs, xs, qs, F, G, ifunc=None, iremove=None):
    # Slow (but simple) version; augment then marginalise
    xa, Pa = moment_form_augment_linearised(x, P, q, Q, fs, xs, qs, F, G, ifunc)
    return _augment2predict(xa, Pa, x, iremove)

# Prediction step; simultaneous augment and marginalise
def moment_form_predict_fast(x, P, Q, ifunc, imarg):
    pass
# efficient implementation does clever in-place arrangement of new values

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
