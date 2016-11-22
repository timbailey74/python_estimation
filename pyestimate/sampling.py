# Samples from various distributions.
# Maybe MCMC stuff...

# See gaussian.py for Gaussian samples.

# http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
def sample_uniform_unit_hypersphere(dim, n):
    u = np.random.rand(n)
    if dim == 1:
        return 2*(0.5 - u)
    x = np.random.randn(dim, n)
    scale = u**(1/dim) / np.sqrt(sum(x**2))
    return scale*x  # note, python broadcasting does the correct replication of scale

# Draw samples from Gaussian, but keeping only those samples that fall within a given probability bound
def gauss_samples_bounded(x, P, N, prob):
    # Solution 1: draw uniformly from unit circle, and transform according to {x,P}
    # Solution 2: draw from {x,P} and reject outliers
    pass

