
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


# Imputation based on linear-Gaussian (PCA) projection
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
    import IPython; IPython.embed()

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


