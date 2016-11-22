# Various spectral utilities and some time-series utilities too.
# Tim Bailey 2015.



import numpy as np
import scipy.linalg as sci
import matplotlib.pyplot as plt
import scipy.fftpack as sfp
import scipy.signal as sig

# Find truncated x such that its end wrapped around to the beginning preserves
# any apparent periodicity in signal. Slice off one side of x until periodic
# frequency strength is maximised.
def periodicity_strength(x, maxbins, leftside=True, type=2):
    if maxbins >= len(x):
        raise ValueError('Cannot truncate more bins than are available.')
    y = np.zeros(maxbins)
    for i in range(maxbins):
        xs = x[i:] if leftside else x[:len(x)-i]
        f = abs(sfp.fft(xs))
        y[i] = max(f[1:]) # note, we ignore DC component (the first bin)
        if   type == 1: y[i] /= len(xs)# normalise by total number of bins; gives max amplitude
        elif type == 2: y[i] /= sum(f) # normalise by sum of amplitudes; measures "concentration" in bin i
        elif type == 3: y[i] /= sum(f**2) # normalise by total power; not valid, I think
    return y

# type==1 normalises the autocorrelation by the number of overlapping bins
def periodicity_strength_autocorr(x, type=1):
    xm = x - np.mean(x) # FIXME: Should I subtract mean(x)? What difference does it make?
    autocorr = sig.correlate(xm, xm, mode='full')
    if type==1: # normalise by number of overlapping bins
        overlap = np.arange(1,len(x))
        autocorr /= np.hstack((overlap, len(x), overlap[::-1]))
    return autocorr

# Returns a single best-periodicity offset
# bins should be a small fraction of the total time-series
# This method truncates more of the time-series than the above methods because
# it involves a full overlap of x by x[-bins:]
def periodicity_autocorr_fraction(x, bins=50):
    autocorr = sig.correlate(x, x[-bins:], mode='valid')
    i = np.argmax(autocorr[:bins])
    i += bins
    return x[i:], i

# Dodgy peak finder, ported from my Matlab code used in ADD project
def find_peaks(x, nbins=1, dmin=1, type=1):
    """
    :param x: array of values
    :param nbins: we define a peak as a maximum centre bin (ie., a local
        maxima) with nbins on either side that slope downwards (ie., have
        decreasing value with distance from centre)
    :param dmin: minimum distance (in bins) between peaks (if two peaks are
        closer than dmin, the lesser peak is removed)
    :param type: if (0) endpoints, x(1) and x(end), cannot be peaks, if (1)
        treat endpoints as wrapped-around, or (2) as potential one-sided peaks
    :return: i, indices of peak locations
    """
    if dmin < nbins:
        raise ValueError('Minimum distance between peaks cannot be less than number of side bins')
    # Support various end-cases
    if type == 1: # endpoints wrap around; append points to facilitate this
        x = np.concatenate((x[-1-dmin:], x, x[:dmin+1])) # FIXME: check this line
    elif type == 2: # endpoints don't wrap, but may still be considered as peaks
        m = min(x)*np.ones(nbins)
        x = np.concatenate((m, x, m))
    # Get all peaks, based on adjacent (single-bin) change-in-slope
    dx = np.diff(x)
    i = nbins + np.logical_and(dx[nbins-1:-nbins]>0, dx[nbins:len(x)-nbins]<0).nonzero()[0]
    # Keep only slopes that do not change sign over the adjacent nbins
    keep = np.ones(len(i), dtype=bool)
    for j in range(1, nbins):
        keep = np.logical_and.reduce((keep, dx[i-j]>=0, dx[i+j-1]<=0)) # FIXME: check this
    i = i[keep]
    # Eliminate smaller peaks too close to a better peak
    while True:
        j = (np.diff(i) <= dmin).nonzero()[0] # find adjacent peaks that are too close
        if len(j) == 0:
            break
        k = x[i[j]] > x[i[j+1]] # determine which peak is higher
        keep = np.ones(len(i), dtype=bool)
        keep[j+k] = False # mark lesser of adjacent peaks
        i = i[keep]
    # Trim ends
    if type == 1:
        keep = np.logical_and(i>dmin, i<len(x)-dmin-1) # FIXME: check this
        i = i[keep] - dmin - 1
    elif type == 2:
        i -= nbins
    return i

# Find peaks based on maximums within a moving window
def find_peaks_windowed(x, nbins=1, type=1):
    # sliding window search for local maxima
    # make peak locations unique
    pass

# Find location of quadratic peak, adapted from my Matlab code used in ADD project
def quadratic_fit(y):
    N = len(y)
    x = np.arange(N)
    Ht = np.vstack((x**2, x, np.ones(N)))
    p = sci.solve(np.dot(Ht, Ht.T), np.dot(Ht, y)) # info-form solve
    px = -p[1]/(2*p[0]) # location of peak
    if px < 0 or px > N-1:
        px = (N-1)/2 # quadratic fit is abnormal, so use mid-point instead
    py = p[0]*px*px + p[1]*px + p[2] # peak value
    return px, py

# Convert array x into a given type
def convert(x, type):
    return np.array([type(val) for val in x])

# Fit quadratic to points surrounding peak
def refine_peaks(x, i):
    xq, iq = x[i], convert(i,float) # note, using deep copies
    for (j,idx) in enumerate(i):
        if idx != 0 and idx != len(x)-1: # FIXME: cater for wrap-around
            offset, val = quadratic_fit(x[idx-1:idx+2])
            iq[j] = idx - 1 + offset
            xq[j] = val
    return xq, iq

def truncate_to_maximise_periodicity(t, x, bins=50, variant=0):
    ps = periodicity_strength(x, bins)
    if variant == 0:
        i = np.argmax(ps)
    else:
        i = find_peaks(ps, type=2)
        i = i[np.argmax(ps[i])] if variant == 1 else i[0]
    if i == bins:
        raise ValueError('Isolated peak not found')
    return t[i:], x[i:]

#
# TEST CODE -----------------------------------------------------
#

def test_find_periodic():
    t = (np.arange(300) / 5)
    x = np.cos(t - 2)

    # Try windowing with a cosine
    cycle = np.linspace(0, 2*np.pi, 301)[:-1] # shave off the end index
    coswindow = (1 - np.cos(cycle))/2
    plt.plot(abs(sfp.fft(x)))
    plt.plot(abs(sfp.fft(x*coswindow)))
    plt.plot(abs(sfp.fft(coswindow)))

    # Try maximise periodicity #1
    ps = periodicity_strength(x, len(x)-1) #50)
    i = find_peaks(ps)
    #fit = [quadratic_fit(ps[idx-1:idx+2]) for idx in i] # fails if i=0 or i=end-of-array
    psq, iq = refine_peaks(ps, i)
    plt.figure()
    plt.plot(ps), plt.plot(i,ps[i], '.', iq,psq,'.')

    # Try maximise periodicity #2
    ps = periodicity_strength_autocorr(x)
    i = find_peaks(ps)
    psq, iq = refine_peaks(ps, i)
    plt.figure()
    plt.plot(ps)
    plt.plot(i,ps[i],'.', iq,psq,'.')

    i = find_peaks(ps, dmin=35)
    plt.plot(i, ps[i], 'o')

    # Truncated time-series
    tt,xt = truncate_to_maximise_periodicity(t, x, variant=0)
    plt.figure()
    plt.plot(t, x, tt, xt)

    plt.show()

if __name__ == "__main__":
    test_find_periodic()

