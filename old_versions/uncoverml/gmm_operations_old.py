# FIXME: assumes x is list of row-arrays
# However, assumes ss.gauss_evaluate() takes a matrix of column-vectors
# FIXME: GMM assumes weights are likelihoods not log-likelihoods



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sample_stats as ss


class GMM:
    def __init__(self, x, P, w, normalise=True):
        self.x = list(x)  # ensure is list, not tuple
        self.P = list(P)
        self.w = np.array(w).flatten()
        if normalise:
            self.w = self.w / sum(self.w)  # FIXME: why does /= sometimes fail??
    def copy(self):
        x = [x.copy() for x in self.x]
        P = [P.copy() for P in self.P]
        return type(self)(x, P, self.w.copy())


# Basic k-component GMM construction for a unit-variance system
def default_gmm(D, k=1, random_mean=True):
    x, P = np.zeros(D), np.eye(D)
    if random_mean:
        x = [np.random.multivariate_normal(x, P, 1) for _ in range(k)]
    else:
        x = [x] * k
    return GMM(x, [P]*k, [1/k]*k)  # FIXME: while P is correct for all x=0, what should it be if x random?


# Basic k-component GMM construction given a set of samples
def nominal_gmm(x, k):
    xm, P = ss.sample_mean_cov(x)
    xs = [np.random.multivariate_normal(xm, P, 1) for _ in range(k)]
    return GMM(xs, [P/k]*k, [1/k]*k)  # FIXME: is P/k a good choice kernel?


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


# FIXME: this function mutates g, do we want this??
# FIXME: Check correctness of wsr, wsc, g.w, and sample_mean_weighted weights
# FIXME: Is there a way to do this with log-weights for better numerics?
# FIXME: Do we want a version of fit that accomodates a percentage of outlier points?
# FIXME: Account for missing data in x
def gmm_em(g, x, n_iterations):
    Nc = len(g.w)  # number of mixture components
    # Compute a nominal covariance for correcting degenerate cases
    _, P = ss.sample_mean_cov(x)
    Pnominal = P / Nc  # FIXME: is this a reasonable nominal value
    # EM interations    
    for _ in range(n_iterations):
        # Fix degenerate covariances
        for i in range(Nc):
            if np.linalg.det(g.P[i]) < 1e-10:  # check for collapsed covariance
                print('Warning: Degenerate covariance.')
                #import IPython; IPython.embed()
                g.P[i] = Pnominal.copy()
        # E-step: compute assignment likelihood
        w = [g.w[i] * ss.gauss_evaluate((x - g.x[i]).T, g.P[i], logw=False) 
                for i in range(Nc)]
        wsr = np.sum(w, 0)
        fit = -sum(np.log(wsr)) if np.all(wsr) else np.inf
        wsr[wsr==0] = 1  # avoid divide-by-zero errors
        w /= wsr
        # M-step: compute new (x,P,w) values for gmm
        wsc = np.sum(w, 1)
        g.w = wsc / sum(wsc)
        for i in range(Nc):
            g.x[i], g.P[i] = ss.sample_mean_weighted(x, w[i]/wsc[i])
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
            import IPython; IPython.embed()
            continue
        if w < wbest:
            wbest = w
            gbest = g
    return gmm_em(gbest, x, n_em[1])


# FIXME: allow for g.w to be logs, and thus direct calculation of logw
def gmm_evaluate(g, x, logw=False):
    Nc = len(g.w)
    w = [g.w[i] * ss.gauss_evaluate((x - g.x[i]).T, g.P[i], logw=False) 
            for i in range(Nc)]
    return np.sum(w, 0) if not logw else ss.log_sum(log(w))


#
def gmm_conditional(g, val, idx):
    k = len(g.w)
    xc, Pc, wc = [], [], []
    for i in range(k):
        x, P, w = ss.gaussian_conditional(g.x[i], g.P[i], val, idx, logw=False)
        xc.append(x)
        Pc.append(P)
        wc.append(g.w[i] * w)
    return GMM(xc, Pc, wc)


#
def gmm_marginal(g, i):
    x = [x[i] for x in g.x]
    j = np.ix_(i) 
    P = [P[j,j] for P in g.P]
    return GMM(x, P, g.w.copy())


# Compute 1st and 2nd moments of (non-normalised) GMM 
def gmm_to_gaussian(g):
    # Record non-normalised total weight
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
    g, f = gmm_em_rnd(x, k)
    plt.plot(x[:,0], x[:,1], '.')
    for x, P in zip(g.x, g.P):
        e = ss.ellipse_sigma(x, P)
        plt.plot(e[:,0], e[:,1])
