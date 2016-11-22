# Simple optimiser functions.

import numpy as np
import scipy.optimize as sopt
import scipy.linalg as slin
import lasers.timstuff.basic_utilities as bu

# Verbose output
verbose_level = 1 # FIXME: use this to control diagnostic output

def verbose_print(level, *args):
    global verbose_level
    if verbose_level >= level:
        print(*args)

def verbose_wrapper(fun, x, args):
    w = fun(x, *args)
    verbose_print(1, 'Probe weight: ', w)
    return w

# Scale a state-vector to approximate zero-mean and unit-variance
class Scaler:
    def __init__(self, mean, sigma):
        self.mean = np.array(mean)
        self.sigma = np.array(sigma)
    def scale(self, x, idx=None):
        if idx is not None:
            return (x - self.mean[idx]) / self.sigma[idx]
        return (x - self.mean) / self.sigma
    def unscale(self, sx, idx=None):
        if idx is not None:
            return self.mean[idx] + sx*self.sigma[idx]
        return self.mean + sx*self.sigma

def line_search_evaluation(fun, xhigh, xlow, t, args):
    """ Evaluate fun() at points spaced according to t, where t=0 is xhigh and
     t=1 is xlow, and t>1 probes beyond xlow. """
    n = len(t)
    xp = np.outer(xlow-xhigh, t) + bu.repcol(xhigh, n)
    wp = [fun(xp[:,i], *args) for i in range(n)]
    i = np.argmin(wp)
    return xp, wp, i

def line_search_optimise_once(fun, xhigh, xlow, wlow, n, args):
    """ Search for a lower-error point along a line via grid-search. Make single search only.
     Deprecated... Prefer line_search_optimise(). """
    t = np.linspace(0, 1, n + 2)[1:]    # probe spacing between (0, 1]
    t = np.concatenate((t[:-1], t + 1)) # probe spacing (0, 1) and (1, 2]
    xp, wp, i = line_search_evaluation(fun, xhigh, xlow, t, args)
    return (xp[:,i], wp[i]) if wp[i] < wlow else (xlow, wlow)

def line_grid_interpolate(fun, xhigh, xlow, n, args):
    """ Compute n equi-spaced points between xhigh and xlow """
    t = np.linspace(0, 1, n + 2)[1:-1] # probe spacing between (0, 1)
    xp, wp, i = line_search_evaluation(fun, xhigh, xlow, t, args)
    return xp, wp, i

def line_grid_extend(fun, xhigh, xlow, n, args):
    """ Compute n+1 equi-spaced points from xlow away from xhigh """
    t = 1 + np.linspace(0, 1, n + 2)[1:] # probe spacing between (1, 2]
    xp, wp, i = line_search_evaluation(fun, xhigh, xlow, t, args)
    return xp, wp, i

def line_search_optimise(fun, xhigh, xlow, wlow, n, args, nits=3):
    """
    Driver function for adaptive grid line-search optimisation. Interpolates to reduce
    grid spacing, and extrapolates to extend search area.
    """
    while nits:
        nits -= 1
        # Start with interpolate between xhigh and xlow
        xp, wp, i = line_grid_interpolate(fun, xhigh, xlow, n, args)
        verbose_print(1, 'Interpolating: w = ', wp[i], ', dx = ', slin.norm(xhigh-xlow)/(n+1))
        if wp[i] < wlow: # We have new low, set up for refined interpolation
            xhigh = xp[:, i+1] if i+1 < n else xlow # shift xhigh to neighbour of new low
            xlow, wlow = xp[:, i], wp[i]
        else:
            # Old xlow is still lowest, need to extend search beyond xlow
            while True:
                xp, wp, i = line_grid_extend(fun, xhigh, xlow, n, args)
                verbose_print(1, 'Extending: w = ', wp[i], ', dx = ', slin.norm(xhigh-xlow)/(n+1))
                if wp[i] <= wlow: # New low found, record it (and shift xhigh in case we extend further)
                    xhigh, xlow, wlow = xlow, xp[:, i], wp[i]
                if i != n or wlow < wp[i]: # No further extension unless wp[n] is new low
                    break
            # Set up next interpolation, shift xhigh to be adjacent to xlow
            xhigh = xp[:, i+1] if wp[i] <= wlow else xp[:, 0]
    return xlow, wlow, xhigh

def line_bracket(fun, xhigh, xlow, wlow, args):
    """ Search along line connecting points xhigh-xlow to find a point that brackets xlow """
    dx = xlow - xhigh
    xp, wp = xlow, wlow
    t = 1
    while wp <= wlow: # search for first xp with rising weight
        wlow = wp
        xp += t * dx
        wp = fun(xp, *args)
        t *= 2
    return xp, wp

def line_bracket3(fun, xhigh, xlow, wlow, args):
    """ Bracket line at three points (0, tlow, 1) """
    # FIXME: Function untested, probably broken
    dx = xlow - xhigh
    xp, wr = xlow, wlow
    t, tlow = 1, 0
    while wr <= wlow: # search for first xp with rising weight
        wlow = wr
        tlow += t
        xp += t * dx
        wp = fun(xp, *args)
        t *= 2
    tlow /= t/2
    return xp, wp, tlow

def line_bracket3points(fun, xhigh, xlow, wlow, args):
    """ Bracket line at three points (xhigh, xlow, xp) """
    # FIXME: Function untested, probably broken
    dx = xlow - xhigh
    xp = xlow + dx
    wp = fun(xp, *args)
    t = 1
    while wp <= wlow: # search for first xp with rising weight
        xhigh, xlow, wlow = xlow, xp, wp
        t *= 2
        xp += t * dx
        wp = fun(xp, *args)
    return xhigh, xlow, xp

def bracket_and_contract_search(fun, xhigh, xlow, wlow, args, tol=10):
    """ Bracket and refine to get local minima on line connecting points xhigh-xlow """
    xbrack, wbrack = line_bracket(fun, xhigh, xlow, wlow, args)
    dx = xbrack - xhigh
    f = lambda t: fun(dx*t + xhigh, *args)
    tb = sopt.minimize_scalar(f, (0,1), method='Golden', tol=tol).x
    xb = dx*tb + xhigh
    wb = fun(xb, *args)
    return (xb, wb) if wb < wlow else (xlow, wlow)

def bracket_and_minimise(fun, xhigh, xlow, wlow, args, tol=10):
    """ Same as bracket_and_contract_search() but using scipy bracketing. """
    dx = xlow - xhigh
    f = lambda t: fun(dx*t + xhigh, *args)
    tb = sopt.bracket(f, 0, 1)
    # FIXME: use tb[:3] or tb[:3:2] for 'Golden'?
    topt = sopt.minimize_scalar(f, tb[:3:2], method='Golden', tol=tol).x
    xopt = dx*topt + xhigh
    wopt = fun(xopt, *args)
    return (xopt, wopt) if wopt < wlow else (xlow, wlow)

def bracket_and_contract_3(fun, xhigh, xlow, wlow, args, tol=10):
    """ Bracket and refine to get local minima on line connecting points xhigh-xlow """
    # FIXME: Function untested, probably broken
    xbrack, wbrack, tlow = line_bracket3(fun, xhigh, xlow, wlow, args)
    dx = xbrack - xhigh
    f = lambda t: fun(dx*t + xhigh, *args)
    tb = sopt.minimize_scalar(f, (0,tlow,1), method='Golden', tol=tol).x
    xb = dx*tb + xhigh
    wb = fun(xb, *args)
    return (xb, wb) if wb < wlow else (xlow, wlow)

def random_walk_optimise(fun, x0, n, M=1, args=()):
    # Assumes x has been scaled to have unit variance
    w0 = fun(x0, *args)
    for i in range(n):
        xtest = x0 + M*np.random.randn(len(x0))
        wtest = fun(xtest, *args)
        if wtest < w0:
            verbose_print(1, 'New best: ', wtest, ' at iteration ', i+1, ' of ', n)
            x0, w0 = xtest, wtest
    return x0, w0

def random_walk_mcmc_old(fun, x0, n, M=1, args=()):
    # Same as random_walk_optimise() but allowing for downhill exploration
    modify = np.exp
    w0 = modify(fun(x0, *args))
    xb, wb = x0, w0
    for i in range(n):
        xtest = x0 + M*np.random.randn(len(x0))
        wtest = modify(fun(xtest, *args))
        if wtest < w0:
            verbose_print(1, 'Downhill jump: ', w0, ' to ', wtest)
            x0, w0 = xtest, wtest
            if w0 < wb:
                verbose_print(1, 'New best: ', w0, ' at iteration ', i+1, ' of ', n)
                xb, wb = x0, w0
        elif w0/wtest > np.random.rand(): # wtest >= w0, do MCMC move
            verbose_print(1, 'Uphill jump: ', w0, ' to ', wtest)
            x0, w0 = xtest, wtest
        else:
            verbose_print(1, 'No change: ', w0, ' at iteration ', i+1, ' of ', n)
    return xb, wb

def random_walk_mcmc(fun, x0, n, M=1, args=()):
    # We assume fun() returns negative-log-likelihoods
    w0 = fun(x0, *args)
    xb, wb = x0, w0
    for i in range(n):
        xtest = x0 + M*np.random.randn(len(x0))
        wtest = fun(xtest, *args)
        if wtest < w0:
            verbose_print(1, 'Downhill jump: ', w0, ' to ', wtest)
            #xtest, wtest, _ = line_search_optimise(fun, x0, xtest, wtest, 3, args)
            x0, w0 = xtest, wtest
            if w0 < wb:
                verbose_print(1, 'New best: ', w0, ' at iteration ', i+1, ' of ', n)
                xb, wb = x0, w0
        elif np.exp(w0 - wtest) > np.random.rand(): # wtest >= w0, do MCMC move
            verbose_print(1, 'Uphill jump: ', w0, ' to ', wtest)
            x0, w0 = xtest, wtest
        else:
            verbose_print(1, 'No change: ', w0, ' at iteration ', i+1, ' of ', n)
    return xb, wb

def stochastic_gradient(fun, x, P, n, args, includex=True):
    xs = np.hstack((x[:,np.newaxis], bu.gauss_samples(x, P, n-1))) if includex else bu.gauss_samples(x, P, n)
    w = [verbose_wrapper(fun, xs[:,i], args) for i in range(n)]
    imin = np.argmin(w)
    imax = np.argmax(w)
    return xs[:,imin], xs[:,imax], w[imin], w[imax]

def rand_line_search(fun, x, P, n, args):
    xs = np.hstack((x[:,np.newaxis], bu.gauss_samples(x, P, n-1)))
    w = [verbose_wrapper(fun, xs[:,i], args) for i in range(n)]
    i = np.argpartition(w, 2) # two lowest weights, ordered
    imin, inext = i[0], i[1]
    return xs[:,imin], xs[:,inext], w[imin], w[inext]

def stochastic_gradient_line_search(fun, x, P, n, nits, args):
    for i in range(nits):
        #xlow, xhigh, wlow, _ = stochastic_gradient(fun, x, P, n, args)
        xlow, xhigh, wlow, _ = rand_line_search(fun, x, P, n, args)
        verbose_print(1, 'Stochastic gradient, w = ', wlow)
        x, wlow, xhigh = line_search_optimise(fun, xhigh, xlow, wlow, 5, args, 6) # FIXME: make configurable
        verbose_print(1, 'Line search, w = ', wlow)
    return x, wlow

def numerical_gradient_line_search(fun, x, offset, args):
    pass

def importance_sampler_optimise(fun, x0, n=100, nits=5, args=()):
    global verbose_level
    xall, wall = np.zeros((len(x0),0)), []
    P = np.eye(len(x0)) # prior is N(x0, I)
    for it in range(nits):
        verbose_print(1, 'Iteration: ', it+1, ' of ', nits)
        xs = bu.gauss_samples(x0, P, n) # draw samples from prior
        ws = [fun(xs[:,i], *args) for i in range(n)]  # compute weights (or log-weights)
        wlike = max(ws) - ws + 1
        wlike = wlike**3 # FIXME: HACK; wlike too flat, exp(wlike) too steep
        # FIXME: do we want to exponentiate wlike (if ws is log-likelihood)?
        wlike /= sum(wlike)
        # FIXME: adjust weights according to proposal...
        x0, P = bu.sample_mean_weighted(xs, wlike) # sample mean and covariance
        xall = np.hstack((xall, xs))
        wall = np.concatenate((wall, ws))
    return xall, wall, x0, P

