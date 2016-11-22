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

