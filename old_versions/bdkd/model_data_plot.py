# Interactive plot of model and data
import sys
import time
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import lasers.timstuff.laser_timeseries as lts

TimeSeriesHandle = namedtuple('TimeSeriesHandle', 'h t ts')

class Plotter:
    def __init__(self, phandle, tsh_measured, tsh_model, h_3d, y0=None):
        canvas = phandle[0].figure.canvas
        canvas.mpl_connect('motion_notify_event', self.mousemove)
        canvas.mpl_connect('button_press_event', self.buttonclick)
        self.phandle = phandle
        self.detune = phandle[0].get_xdata()
        self.inject = phandle[0].get_ydata()
        self.measured = tsh_measured
        self.model = tsh_model
        self.h_3d = h_3d
        self.evaluated = np.ones(self.detune.shape[0])
        self.istore = [-1]
        self.y0 = y0
        if y0 is not None: # we only use y0 if we are lazy evaluating
            self.evaluated[:] = 0

    def mousemove(self, event):
        if event.inaxes != self.phandle[0].axes:
            return

        # Get detune-inject pair nearest to cursor
        # FIXME: Use kd-tree to speed up search; (will it make any difference?)
        d2 = (self.detune - event.xdata)**2 + (self.inject - event.ydata)**2
        i = np.argmin(d2)
        self.istore[-1] = i

        # Draw line to nearest detune-inject point
        self.phandle[1].set_data([self.detune[i], event.xdata], [self.inject[i], event.ydata])
        self.phandle[2].set_data(self.detune[i], self.inject[i])

        # Display value of i
        #self.phandle[0].figure.get_axes()[0].set_title(''.join(('i = ', str(i))))
        self.phandle[0].get_axes().set_title(''.join(('i = ', str(i))))

        # Evaluate generative model timeseries, if necessary
        if not self.evaluated[i]: # lazy evaluation
            self.evaluated[i] = 1
            self.model.ts[i,:] = lts.generate_model_timeseries_intensity(self.model.t, self.y0, self.detune[i], self.inject[i])

        # Plot timeseries
        set_ts_data(self.model, i)
        set_ts_data(self.measured, i)

        # Draw canvas, use any handle since all handles will get the same figure
        self.phandle[0].figure.canvas.draw()

        # Draw 3D plot; only in lazy evaluation mode
        if self.y0 is not None: # FIXME: Do this properly, with caching etc, so that its fast
            y = lts.generate_model_timeseries(self.model.t, self.y0, self.detune[i], self.inject[i])
            self.h_3d.set_data(y[:, 0], y[:, 1])
            self.h_3d.set_3d_properties(y[:, 2])
            self.h_3d.figure.canvas.draw()

    def buttonclick(self, event):
        if event.inaxes != self.phandle[0].axes:
            return
        if self.istore[-1] != -1:
            self.istore += [-1]

def set_ts_data(tsh, i):
    tsh.h.set_data(tsh.t, tsh.ts[i,:])

def main():
    # Configurables (defaults)
    lazy_evaluate = True
    #fname = '../../../someExperimentalData.npz'
    fname = 'someExperimentalData.npz'
    modstate = lts.defaultModelConditions
    #lts.theta = lts.theta._replace(J = 2.6) # can modify lts.theta like this

    # Configurables (command-line)
    if len(sys.argv) > 1: lazy_evaluate = bool(int(sys.argv[1]))
    if len(sys.argv) > 2: fname = sys.argv[2]
    if len(sys.argv) > 3: # laser parameter-set (OISSL_Theta) in configurables
        vals = [float(s) for s in sys.argv[3:]]
        lts.theta = lts.models.OISSL_Theta(*vals)
    # FIXME: collect all configurables in a config-file


    lts.theta = lts.models.OISSL_Theta(1.51461013e-01,   4.74756887e+06,   2.49912785e+00,
         2.20114318e+10,   1.10815433e+04,   3.13466722e+04,  3.00078352e+05)
    print('Theta: ', lts.theta)

    # Get measured data-set
    data = np.load(fname)

    # Timestamps of real data
    t = lts.make_data_timestamps(data['timeseries'].shape[1])

    # Timestamps of generative model and storage for its timeseries
    tm = lts.make_model_timestamps(modstate)
    ts_m = np.zeros((data['Detune'].shape[0], len(tm)))

    # Precompute generative model predictions
    if not lazy_evaluate:
        clock_s = time.clock()
        for i in range(data['Detune'].shape[0]):
            ts_m[i,:] = lts.generate_model_timeseries_intensity(tm, modstate.y0, data['Detune'][i], data['Inject'][i])
            cdiff = time.clock() - clock_s
            tremain = cdiff * (data['Detune'].shape[0]-i) / (i+1)
            print('Calculating ', i, 'of ', data['Detune'].shape[0],
                  '. Time remaining: ', tremain/60, 'mins')

    # Plotting setup
    fig = plt.figure('Model versus Data')
    fig.add_subplot(221)

    plt.subplot2grid((2,2), (1,0), colspan=2)
    h_controls = plt.plot(data['Detune'], data['Inject'], '.', 0, 0, 'r', 0, 0, 'r*')
    plt.xlabel('Detune')
    plt.ylabel('Inject')
    #h_controls[0].get_axes().set_aspect('equal')

    tsmin = np.min(data['timeseries'])
    tsmax = np.max(data['timeseries'])
    plt.subplot2grid((2,2), (0,0))
    h_ts, = plt.plot([0, t[-1]], [tsmin, tsmax])
    plt.title('Measured Data')
    plt.subplot2grid((2,2), (0,1))
    h_tsm, = plt.plot([0, tm[-1]], [tsmin, tsmax])
    plt.title('Predicted Data (ODE model)')

    ax = plt.figure('Phase Space 3D').add_subplot(111, projection='3d')
    h_3d, = ax.plot([-3,3], [-3,3], [-3,3]) # FIXME: clean up setting of axis ranges
    plt.title('ODE Model: Phase Space 3D')
    ax.set_xlabel('Ar')
    ax.set_ylabel('Ai')
    ax.set_zlabel('n')

    # Run interactive plot
    h_measured = TimeSeriesHandle(h_ts, t, data['timeseries'])
    h_model = TimeSeriesHandle(h_tsm, tm, ts_m)
    pltr = Plotter(h_controls, h_measured, h_model, h_3d, modstate.y0 if lazy_evaluate else None)
    plt.show()

    # Print selected indices
    print(pltr.istore[:-1])

#
# Run program
#

main() # execute this program
