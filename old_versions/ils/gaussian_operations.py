# Moment-form and information-form operations on Gaussian pdfs.

import numpy as np
import scipy.linalg as sci
from scipy.special import gammainc
from scipy.optimize import brentq
import ils.linear_algebra as la

# FIXME: try replacing all triangular solve operations (eg., sci.cho_solve()
# etc) with sci.solve_triangular() with check_finite=False. What, if any, are
# the speed gains?
# FIXME: When given matrices representing a collection of vectors, most 
# operations expect col-vectors; should extend to allow row-vectors

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

# Evaluate Gaussian pdf N(x,P) given L*L.T = P and L*f = x
def gauss_evaluate_sqrtform(f, L, logw=False):
    D = L.shape[0]
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

# Draw samples from a multivariate Gaussian
def gauss_samples(x, P, n):
    L = sci.cholesky(P, lower=True)
    X = np.random.standard_normal((len(x), n))
    return np.dot(L, X) + la.repcol(x, n)  # alternatively, to broadcast x (if an array), use x[:,newaxis]


# Simple information-form update step (ie., fusion step). Modification of
# (y, Y) performed in-place, so they are not returned from function.
# WARNING: Does not cater for discontinuous functions.
def information_form_update_old(y, Y, z, R, xs, zs, Hs, idx, logflag=-1):
    if logflag != -1:
        v, S = information_form_innovation(y, Y, z, R, xs, zs, Hs, idx)
        w = gauss_evaluate(v, S, logflag)
    Ri = np.linalg.inv(R) # FIXME: Expensive if R is diagonal
    HtRi = np.dot(Hs.T, Ri)
    Y[np.ix_(idx,idx)] += la.force_symmetry(np.dot(HtRi, Hs)) # Note: requires np.ix_() to work correctly
    y[idx] += np.dot(HtRi, (z - zs + np.dot(Hs,xs)))
    if logflag != -1:
        return w

# WARNING: Does not cater for discontinuous functions.
def information_form_update(y, Y, z, R, xs, zs, rs, Hx, Hr, idx, logflag=-1):
    Ra = la.triprod(Hr, R, Hr.T)
    if logflag != -1:  # FIXME: expensive information2moment conversion
        x, P = information2moment(y, Y, idx)
        v = moment_form_innovation(x, z, xs, rs, zs, Hx, Hr)
        S = la.triprod(Hx, P, Hx.T) + Ra
        w = gauss_evaluate(v, S, logflag)
    Ri = np.linalg.inv(Ra) # FIXME: Expensive if Ra is diagonal
    HtRi = np.dot(Hx.T, Ri)
    Y[np.ix_(idx,idx)] += la.force_symmetry(np.dot(HtRi, Hx)) # Note: requires np.ix_() to work correctly
    y[idx] += np.dot(HtRi, (z - zs + np.dot(Hx,xs) + np.dot(Hr,rs)))
    if logflag != -1:
        return w


# Prediction is augment and marginalise simultaneously
# For formulae, see Maybeck Vol1, Section 5.7, pp. 238--241
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
    ikeep = la.index_other(y.size, iremove)
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
    Ya[np.ix_(idx,idx)] += la.force_symmetry(np.dot(FtQi, F))
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
    Qlin = la.triprod(G, Q, G.T)
    return information_form_augment(y, Y, qlin, Qlin, F, idx)

# Information-form marginalisation
def information_form_marginalise(y, Y, idx):
    idy = la.index_other(y.size, idx)
    F = Y[np.ix_(idx, idy)]
    B = np.linalg.inv(Y[np.ix_(idy, idy)])
    FB = np.dot(F, B)
    Ym = Y[np.ix_(idx,idx)] - la.force_symmetry(np.dot(FB, F.T))
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
    S = la.triprod(Hs, P, Hs.T) + R
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
        rdiff = np.dot(sci.inv(Hr), numerator)
        #rdiff = sci.solve(Hr, numerator)  # FIXME: check that this produces same result as line above
    return rdiff + rs


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
    S = la.triprod(Hs, P1, Hs.T) + R
    L = sci.cholesky(S, lower=True)
    Li = np.linalg.inv(L)
    #Li = sci.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    vc = np.dot(Li, v)
    Wc = la.triprod(P2, Hs.T, Li.T)
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
    Pnn = la.triprod(F, P1, F.T) + Q  # new block-diagonal
    Pnn = la.force_symmetry(Pnn)  # FIXME: should properly ensure Pnn is symmetric pos-def
    Pno = np.dot(F, P2)  # new-old off-diagonal block
    Pa = np.vstack((np.hstack((P, Pno.T)), np.hstack((Pno, Pnn))))
    return xa, Pa

# Linearised augmentation
def moment_form_augment_linearised(x, P, q, Q, fs, xs, qs, F, G, idx=None):
    # fs = f(xs, qs, ...)
    # FIXME: check x-xs and q-qs do not have discontinuities
    # FIXME: qlin, Qlin calculation same as for info-form; refactor
    qlin = fs - np.dot(F, xs) + np.dot(G, q-qs)
    Qlin = la.triprod(G, Q, G.T)
    return moment_form_augment(x, P, qlin, Qlin, F, idx)

# Partial implementation of the two prediction functions below
def _augment2predict(xa, Pa, x, iremove):
    if iremove is None:  # remove all of previous state, keep only new states
        ikeep = np.arange(len(x), len(xa))
    else:  # keep some of old states
        ikeep = la.index_other(len(xa), iremove)
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


#
# TESTS ------------------------------------------------------
#

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
    (ym, Ym) = information_form_marginalise(yc, Yc, la.index_other(yc.size, iremove))
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
    flags = 0xff  # test all
    flags = 0x01
    if flags & 0x01:
        print('test_gauss_evaluate()')
        test_gauss_evaluate()
    if flags & 0x02:
        print('test_information_form_ops()')
        test_information_form_ops()
    if flags & 0x04:
        print('test_constrain_marginalise()')
        test_constrain_marginalise()
