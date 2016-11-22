# Dodgy moving average methods, in leiu of proper estimation
#

# Mean and std-dev with removal of outliers greater than s-stddevs from mean. Can substitute median for mean.
def robust_mean(x, s, mid_func=np.mean):
    i = x==x  # true for everything except nans
    while True:
        mid, stddev = mid_func(x[i]), np.std(x[i])
        j = abs(x - mid) <= s*stddev  # use <= to account for std==0
        if np.all(i==j):
            break
        i = j
    return mid, stddev, i


# Sliding window polynomial fit; deals with end-effects by truncating number of points in polynomial
def poly_smooth_trunc(t, x, deg, window):
    assert deg < 1 + 2*window
    N = len(t)
    xs = np.zeros(N)
    for i in range(N):
        if i < window: idx = slice(0, 1+max(deg, i+window))
        elif i >= N-window: idx = slice(min(N-1-deg, i-window), N)
        else: idx = slice(i-window, i+window+1)
        #print(i,idx)
        p = np.polyfit(t[idx], x[idx], deg=deg)
        xs_idx = np.polyval(p, t[idx])
        if i < window: xs[i] = xs_idx[i]
        elif i >= N-window: xs[i] = xs_idx[i-N]
        else: xs[i] = xs_idx[window]
    return xs

# Sliding window polynomial fit. If a point appears in multiple polynomials, average them
def poly_smooth_av(t, x, deg, window):
    assert deg < 1 + 2*window
    N = len(t)
    xs = np.zeros(N)
    ns = np.zeros(N)
    for i in range(window, N-window):
        idx = slice(i-window, i+window+1)
        p = np.polyfit(t[idx], x[idx], deg=deg)
        xs[idx] += np.polyval(p, t[idx])
        ns[idx] += 1
    xs /= ns  # average entries that had multiple polys
    return xs

# Sliding window polynomial fit; Savitzky–Golay filter
# FIXME: perhaps return gradient for each point also (if, requested)
# FIXME: perhaps allow for evaluation at different points (ie., interpolation)
def poly_smooth(t, x, deg, window):
    N = len(t)
    assert deg < 1 + 2*window
    assert N > 2*window
    xs = np.zeros(N)
    for i in range(window, N-window):
        idx = slice(i-window, i+window+1)
        p = np.polyfit(t[idx], x[idx], deg=deg)
        xpoly = np.polyval(p, t[idx])
        xs[i] = xpoly[window]
        if i == window:  # start condition; use first poly to fit the initial points
            xs[:i] = xpoly[:window]
        if i == N-window-1:  # end condition; use last poly to fit the final points
            xs[i:] = xpoly[window:]
    return xs




#
# TEST CODE ------------------------------------------------------------------
#

def test_polysmooth():
    t = np.arange(10)
    x = np.sin(t)
    xs2 = poly_smooth(t, x, 1, 1)
    xs3 = poly_smooth(t, x, 3, 2)
    plt.plot(t, x, t, xs2, t, xs3)

