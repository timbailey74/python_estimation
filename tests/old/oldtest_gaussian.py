#
#

def test_ellipse_mass():
    x = np.random.randn(2,1)
    P = np.random.randn(2,2)
    P = np.dot(P, P.T)
    e1 = ellipse_mass(x, P, 0.95, test=True)
    e2 = ellipse_mass(x, P, 0.95, test=False)
    plt.plot(e1[0,:], e1[1,:], '.-', e2[0,:], e2[1,:], '.-')
    plt.show()
    

def test_gauss_evaluate():
    S = np.random.uniform(-3,3,(3,3))
    S = np.dot(S.T, S)
    v1 = np.random.uniform(-1,1,(3,4))
    v2 = v1[:,0]
    print(gauss_evaluate(v1, S))
    print(gauss_evaluate(v1, S, True))
    print(gauss_evaluate(v2, S, False))
    print(gauss_evaluate(v2, S, True))


def is_equal(a, b):
    return np.all(np.abs(a-b) < 1e-10)

# x is a single (D,) vector, or a (N,D) matrix of row vectors
# vals is either (len(i),), or (N,len(i)), or (N,) in 1-D case
# FIXME: remove duplication from test code
def test_gaussian_conditional(N, D, i=[1], logw=True):
    x = np.random.rand(N, D)
    P = np.random.randn(D, D)
    P = np.dot(P, P.T)
    vals = np.random.rand(N, len(i))
    #
    xc,Pc,wc = gaussian_conditional(x, P, vals, i, logw)
    for j, (xj, vj) in enumerate(zip(x, vals)):
        a,b,c = gaussian_conditional(xj, P, vj, i, logw)
        assert is_equal(a,xc[j,:]) and is_equal(b,Pc) and is_equal(c,wc[j])
    #   
    xsingle = x[0,:]
    xs,Ps,ws = gaussian_conditional(xsingle, P, vals, i, logw)
    for j, vj in enumerate(vals):
        a,b,c = gaussian_conditional(xsingle, P, vj, i, logw)
        assert is_equal(a,xs[j,:]) and is_equal(b,Ps) and is_equal(c,ws[j])
    #
    valsingle = vals[0,:]
    xv,Pv,wv = gaussian_conditional(x, P, valsingle, i, logw)
    for j, xj in enumerate(x):
        a,b,c = gaussian_conditional(xj, P, valsingle, i, logw)
        assert is_equal(a,xv[j,:]) and is_equal(b,Pv) and is_equal(c,wv[j])
    #
    if len(i) == 1:
        val1 = np.random.rand(N)
        xx,PP,ww = gaussian_conditional(x, P, val1, i, logw)
        for j, (xj, vj) in enumerate(zip(x, val1)):
            a,b,c = gaussian_conditional(xj, P, vj, i, logw)
            assert is_equal(a,xx[j,:]) and is_equal(b,PP) and is_equal(c,ww[j])
        xy,Py,wy = gaussian_conditional(xsingle, P, val1, i, logw)
        for j, vj in enumerate(val1):
            a,b,c = gaussian_conditional(xsingle, P, vj, i, logw)
            assert is_equal(a,xy[j,:]) and is_equal(b,Py) and is_equal(c,wy[j])
    #import IPython; IPython.embed()
    print('Success')

#   x is (2,) or (N,2); and vals is (N,)
def test_gaussian_conditional_2d(N, i=1, logw=True):
    D = 2
    x = np.random.rand(N, D)
    P = np.random.randn(D, D)
    P = np.dot(P, P.T)
    vals = np.random.rand(N)
    #
    a,b,c = gaussian_conditional(x, P, vals, [i], logw)
    d,e,f = gaussian_conditional_2d(x, P, vals, i, logw)
    assert is_equal(a,d) and is_equal(b,e) and is_equal(c,f)
    #
    xsingle = x[0,:]
    a,b,c = gaussian_conditional(xsingle, P, vals, [i], logw)
    d,e,f = gaussian_conditional_2d(xsingle, P, vals, i, logw)
    assert is_equal(a,d) and is_equal(b,e) and is_equal(c,f)
    print('Success')
    #import IPython; IPython.embed()

