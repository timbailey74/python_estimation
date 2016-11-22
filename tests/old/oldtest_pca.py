#
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
    #import IPython; IPython.embed()

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

