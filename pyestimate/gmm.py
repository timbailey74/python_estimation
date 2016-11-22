# FIXME: assumes x is list of row-arrays
# However, assumes ss.gauss_evaluate() takes a matrix of column-vectors


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import operator
import sample_stats as ss


class GMM:
    def __init__(self, x, P, w, logw=False, normalise=True):
        self.x = list(x)  # ensure is list, not tuple
        self.P = list(P)
        self.w = np.array(w).flatten()
        self._logw = logw
        if normalise:
            if logw:
                self.w = self.w - ss.log_sum(self.w)
            else:
                self.w = self.w / sum(self.w)  # FIXME: why does /= sometimes fail??
    def copy(self):
        x = [x.copy() for x in self.x]
        P = [P.copy() for P in self.P]
        return type(self)(x, P, self.w.copy(), self.logw)
    @property
    def logw(self):
        return self._logw
    @logw.setter
    def logw(self, logw):
        if logw != self._logw:
            assert type(logw) is bool
            self.w = np.log(self.w) if logw else np.exp(self.w)
            self._logw = logw


# Basic k-component GMM construction for a unit-variance system
def default_gmm(D, k=1, random_mean=True):
    x, P = np.zeros(D), np.eye(D)
    if random_mean:
        x = [np.random.multivariate_normal(x, P, 1)[0] for _ in range(k)]
    else:
        x = [x] * k
    return GMM(x, [P]*k, [1/k]*k)  # FIXME: while P is correct for all x=0, what should it be if x random?


# Basic k-component GMM construction given a set of samples
def nominal_gmm(x, k):
    xm, P = ss.sample_mean_cov(x)
    xs = [np.random.multivariate_normal(xm, P, 1)[0] for _ in range(k)]
    return GMM(xs, [P/k]*k, [1/k]*k)  # FIXME: is P/k a good choice kernel?


# Evaluate likelihood of samples for each separate component of GMM
def gmm_evaluate_componentwise(g, x, logw=False):
    Nc = len(g.w)
    op1 = [np.exp, np.log][logw]
    op2 = [operator.mul, operator.add][logw]
    gw = g.w if g.logw == logw else op1(g.w)
    return [op2(gw[i], ss.gauss_evaluate((x - g.x[i]).T, g.P[i], logw)) for i in range(Nc)]


# Evaluate GMM likelihood at each location x
def gmm_evaluate(g, x, logw=False):
    w = gmm_evaluate_componentwise(g, x, logw)
    return np.sum(w, 0) if not logw else ss.log_sum(w)


# Deprecated
def gmm_conditional_old(g, val, idx):
    k = len(g.w)
    op = operator.mul if not g.logw else operator.add
    xc, Pc, wc = [], [], []
    for i in range(k):
        x, P, w = ss.gaussian_conditional(g.x[i], g.P[i], val, idx, g.logw)
        xc.append(x)
        Pc.append(P)
        wc.append(op(g.w[i], w))
    return GMM(xc, Pc, wc, g.logw)


# Vectorised conditional for multiple values (all over same idx)
#   vals is (len(idx),), (N,(len(idx)) or (N,) for 1D case
def gmm_conditional(g, vals, idx):
    k = len(g.w)
    op = operator.mul if not g.logw else operator.add
    xc, Pc, wc = [], [], []
    for i in range(k):
        x, P, w = ss.gaussian_conditional(g.x[i], g.P[i], vals, idx, g.logw)
        xc.append(x)
        Pc.append(P)
        wc.append(op(g.w[i], w))
    if len(vals.shape) == 1 and vals.shape[0] == len(idx):
        return GMM(xc, Pc, wc, g.logw)
    xc = np.array(xc)
    wc = np.array(wc)
    return [GMM(list(xc[:,i,:]), Pc, wc[:,i], g.logw) for i in range(len(vals))]



def gmm_conditional_2d(g, vals, idx=1):
    k = len(g.w)
    op = operator.mul if not g.logw else operator.add
    xc, Pc, wc = [], [], []
    for i in range(k):
        x, P, w = ss.gaussian_conditional_2d(g.x[i], g.P[i], vals, idx, g.logw)
        xc.append(x)
        Pc.append(P)
        wc.append(op(g.w[i], w))
    if len(vals) == 1:
        return GMM(xc, Pc, wc, g.logw)
    xc = np.array(xc)
    wc = np.array(wc)
    return [GMM(list(xc[:,i,:]), Pc, wc[:,i], g.logw) for i in range(len(vals))]


#
def gmm_marginal(g, i):
    x = [x[i] for x in g.x]
    P = [P[np.ix_(i,i)] for P in g.P]
    return GMM(x, P, g.w.copy(), g.logw)


# Compute 1st and 2nd moments of (non-normalised) GMM 
def gmm_to_gaussian(g):
    # Record non-normalised total weight
    assert g.logw == False
    wt = sum(g.w)
    # Normalise (ie., make sum to 1)
    w = g.w / wt
    # Compute sample mean and covariance of component means
    x, P = ss.sample_mean_weighted(np.array(g.x), w)
    # Add component covariances to sample covariance
    for wi, Pi in zip(w, g.P):
        P += wi * Pi 
    return x, P, wt


#
def gmm_samples(g, n):
    assert g.logw == False
    wc = np.cumsum(g.w)
    uc = np.cumsum(np.random.rand(n+1))
    uc *= wc[-1] / uc[-1]
    uc[-1] = wc[-1]  # ensure exact, ie., no numerical imperfection
    x, s, t = [], 0, 0
    for i in range(len(wc)):
        while uc[t] < wc[i]:
            t += 1
        if t > s:
            x.append(np.random.multivariate_normal(g.x[i], g.P[i], t-s))
        s = t
    assert t == n
    return np.vstack(x)


# Bound on GMM likelihood that encloses prob-percent of the probability mass
def gmm_likelihood_bound(g, prob=0.95, N=1000):
    assert 0 < prob < 1
    #if N is None:  # make N to order of number of significant-figures
    #    val = prob * 10**np.arange(5)
    #    N = 10 ** next(i for i,v in enumerate(val) if not v-int(v))
    x = gmm_samples(g, N)
    w = gmm_evaluate(g, x, g.logw)
    i = int((1-prob) * N)
    return np.sort(w)[i]


# Use kmeans to generate a GMM based on hard nearest-neighbour assignment
def kmeans_gmm(x, k, nkm=(5,50), check_degeneracy=True):
    km = KMeans(k, init='random', n_init=nkm[0], max_iter=nkm[1])
    idx = km.fit_predict(x)
    assert max(idx) < k
    w = [sum(idx==i) for i in range(k)]
    meancov = [ss.sample_mean_cov(x[idx==i,:], 0) for i in range(k)]
    xm, P = zip(*meancov)
    if check_degeneracy:  # will raise an exception if any P is degenerate
        [np.linalg.cholesky(Pi) for Pi in P]
    return GMM(xm, P, w)


#
def kmeans_gmm_trials(x, k, nkm, trials):
    for _ in range(trials):
        try:
            return kmeans_gmm(x, k, nkm, True)
        except:
            pass


# FIXME: this function mutates g, do we want this?? I think we do.
# FIXME: Check correctness of wsr, wsc, g.w, and sample_mean_weighted weights
# FIXME: Account for missing data in x
def gmm_em(g, x, n_iterations, outliers=0):
    def compute_fit(wsr, outliers, logw):
        if outliers:
            assert 0 <= outliers < 1
            keep = np.ceil((1-outliers) * len(wsr))
            wsr_inlier = np.sort(wsr)[-keep:]
        else:
            wsr_inlier = wsr
        if not logw:
            fit = -sum(np.log(wsr_inlier)) if np.all(wsr_inlier) else np.inf
        else: 
            fit = -sum(wsr_inlier)
        return fit
    # Fix degenerate covariances
    def mitigate_degeneracy(g, Pnominal):
        Nc = len(g.w)
        for i in range(Nc):
            if np.linalg.det(g.P[i]) < 1e-9:  # check for collapsed covariance
                #print('Warning: Degenerate covariance.')
                g.P[i] = Pnominal.copy()
    # E-step: compute assignment likelihood
    def estep(g, x, outliers):
        w = gmm_evaluate_componentwise(g, x, g.logw)
        wsr = np.sum(w, 0) if not g.logw else ss.log_sum(w) 
        fit = compute_fit(wsr, outliers, g.logw)
        if not g.logw:
            wsr[wsr==0] = 1  # to avoid divide-by-zero errors
            w /= wsr
        else: 
            w -= wsr
        return w, fit
    # M-step: compute new (x,P,w) values for gmm
    def mstep(g, x, w):
        if g.logw:  # need non-log weights for the M-step
            g.w = np.exp(g.w)
            w = np.exp(w)
        Nc = len(g.w)
        wsc = np.sum(w, 1)  
        for i in range(Nc):
            g.x[i], g.P[i] = ss.sample_mean_weighted(x, w[i]/wsc[i])
        g.w = wsc / sum(wsc)
        if g.logw:  # convert back again
            g.w = np.log(g.w)
        return g
    #
    # EM Algorithm 
    #
    # Compute a nominal covariance for correcting degenerate cases
    _, P = ss.sample_mean_cov(x)
    Pnominal = P / len(g.w)  # FIXME: is this a reasonable nominal value
    # EM iterations    
    for _ in range(n_iterations):
        mitigate_degeneracy(g, Pnominal)
        w, fit = estep(g, x, outliers)
        g = mstep(g, x, w)
    mitigate_degeneracy(g, Pnominal)  # FIXME: should not have to do this; look for a better fix
    _, fit = estep(g, x, outliers)  # FIXME too
    return g, fit


#
def gmm_em_rnd(x, k=5, n_em=(50,200), nkm=(5,5,50), n_rnd=5):
    wbest = np.inf
    for _ in range(n_rnd):
        # Initialise GMM
        if nkm is not None:  # using kmeans
            g = kmeans_gmm_trials(x, k, nkm[1:], nkm[0]) 
        else:  # using random second-moment approx
            g = nominal_gmm(x, k)
        # Do EM fit
        try:
            gs = g.copy()  # preserve original, for debugging
            g, w = gmm_em(g, x, n_em[0])
        except:  # FIXME: how might gmm_em fail?
            #import IPython; IPython.embed()
            continue
        if w < wbest:
            wbest = w
            gbest = g
    return gmm_em(gbest, x, n_em[1])


# 3D plots
def plot3(x, y, z, form='-', title = '', ax=None):
    if not ax:
        #ax = plt.gcf().add_subplot(111, projection='3d')
        ax = plt.gca(projection='3d')
    ax.plot(x, y, z, form)
    ax.set_title(title)
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


#
def plot_2d_components(g, x=None, prob=0.95):
    e = []
    for i in range(len(g.w)):
        e.append(ss.ellipse_mass(g.x[i], g.P[i], prob))
        e.append([np.nan]*2)
    e = np.vstack(e)
    plt.plot(e[:,0], e[:,1])
    if x is not None:
        plt.plot(x[:,0], x[:,1], '.')


#
def plot_2d_pdf(g, x=None, prob=0.95):
    assert g.logw == False
    e, h = [], []
    for i in range(len(g.w)):
        e.append(ss.ellipse_mass(g.x[i], g.P[i], prob))
        e.append([np.nan]*2)
        h.append([*g.x[i], 0])
        h.append([*g.x[i], g.w[i]*ss.gauss_evaluate(np.zeros(2), g.P[i], False)])
        h.append([np.nan]*3)
    e = np.vstack(e)
    h = np.vstack(h)
    if x is not None:
        w = gmm_evaluate(g, x, False)
        t = []
        for i in range(len(x)):
            t.append([*x[i], 0])
            t.append([*x[i], w[i]])
            t.append([np.nan]*3)
        t = np.vstack(t)
    ax = plot3(h[:,0], h[:,1], h[:,2], 'r-', '2D GMM')
    ax.plot(e[:,0], e[:,1])
    if x is not None:
        plot3(t[:,0], t[:,1], t[:,2], 'g-', ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


# http://matplotlib.org/examples/pylab_examples/contour_demo.html
# http://www.python-course.eu/matplotlib_contour_plot.php
def plot_2d_contours(g, w, x=None):
    pass


#
#
#


def test_generate_data():
    s1 = np.random.multivariate_normal([0,0], [[1,0], [0,2]], 500)
    s2 = np.random.multivariate_normal([3,4], [[3,0.5], [0.5,2]], 1000)
    return np.vstack((s1, s2))


def test_kmeans(N=2, x=test_generate_data()):
    k = KMeans(N)
    idx = k.fit_predict(x)
    col = 'bgrkcmy'
    for i in range(N):
        xi = x[idx==i, :]
        xim, Pi = ss.sample_mean_cov(xi, 0)
        wi = xi.shape[0] / x.shape[0]
        e = ss.ellipse_sigma(xim, Pi)
        c = col[i%len(col)]
        plt.plot(xi[:,0], xi[:,1], c+'.', e[:,0], e[:,1], c)


def test_gmm_fit(k=2, x=test_generate_data()):
    assert type(k) is int and k > 0
    g, f = gmm_em_rnd(x, k)
    plt.plot(x[:,0], x[:,1], '.')
    for x, P in zip(g.x, g.P):
        e = ss.ellipse_sigma(x, P)
        plt.plot(e[:,0], e[:,1])
        