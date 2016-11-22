#
#

# Simple information-form update step (ie., fusion step). Modification of
# (y, Y) performed in-place, so they are not returned from function.
# WARNING: Does not cater for discontinuous functions.
def information_form_update_old(y, Y, z, R, xs, zs, Hs, idx, logflag=-1):
    if logflag != -1:
        v, S = information_form_innovation(y, Y, z, R, xs, zs, Hs, idx)
        w = gauss_evaluate(v, S, logflag)
    Ri = np.linalg.inv(R) # FIXME: Expensive if R is diagonal
    HtRi = np.dot(Hs.T, Ri)
    Y[np.ix_(idx,idx)] += la.force_symmetry(np.dot(HtRi, Hs)) # Note: requires np.ix_() 
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

