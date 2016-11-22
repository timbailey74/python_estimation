#
#

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

