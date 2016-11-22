# Sequential Monte Carlo and related operations

import numpy as np
import matplotlib.pyplot as plt
import pickle

from lasers.timstuff.optimisers import Scaler
import lasers.timstuff.linearised_estimation_for_laser as lel
import lasers.timstuff.laser_timeseries as lts
import lasers.timstuff.basic_utilities as bu
import lasers.timstuff.iterative_ls as ils

# Draw gaussian samples from an array of independent scalar pdfs
def gauss_samples_sigma(x, sigma, n=1):
    try: # each x has its own sigma
        X = [xi + si*np.random.standard_normal(n) for (xi,si) in zip(x, sigma)]
    except: # each x has same sigma
        X = [xi + sigma*np.random.standard_normal(n) for xi in x]
    return np.array(X)
# FIXME: instead of try-except block, above might be faster to write:
#   if np.size(sigma) == 1:  # best option, I think
# or:
#   if isinstance(sigma, float) ...
# or:
#   if not isinstance(sigma, (list, tuple, np.ndarray))
# or:
#   from collections import Iterable
#   if not isinstance(sigma, Iterable)

# Generate Nc samples for params, noise, controls, dynamics, where the number
# of time-series is defined by len(z0), ie., the length of the initial
# observation vector. Thus, s_param is (7,Nc), s_noise (2,Nc), s_control
# (2*len(z0),Nc), and s_dynamic (3*len(z0),Nc).
def generate_samples(priors, controls, z0, Nc):
    params, noise, det_inj, dyn = priors
    assert len(controls[0]) == len(z0)
    # Generate (mean,sigma) tuple for controls using nominal values and prior; (Sorry, ugly notation here)
    controls = np.array(controls).flatten('F'), np.tile(det_inj[1], len(z0))
    # Draw samples from priors of params, noise and controls
    s_param = abs(gauss_samples_sigma(params[0], params[1], Nc))  # NOTE: enforce positive
    s_noise = abs(gauss_samples_sigma(noise[0], noise[1], Nc))  # NOTE: enforce positive
    s_control = gauss_samples_sigma(controls[0], controls[1], Nc)
    # Draw samples from initial dynamic state (ar0, ai0, n0), given z0
    # Note: we don't use prior for (ar0, ai0) because we instead use p(z|ar0,ai0). But we do use prior for n0.
    s_dynamic = list()
    for sigma_z in s_noise[1,:]:
        ar_ai = lel.sample_amplitudes(z0, sigma_z)
        n = gauss_samples_sigma([dyn[0][2]], [dyn[1][2]], len(z0))
        s_dynamic += [np.vstack((ar_ai, n)).flatten('F')]
    s_dynamic = np.array(s_dynamic).T
    return s_param, s_control, s_noise, s_dynamic

# Create Nc samples from prior information and first observation
@bu.static_vars(prior_info=None)
def initialise_samples(Nc, *args):
    if len(args):
        initialise_samples.prior_info = args
    priors, controls, z0 = initialise_samples.prior_info
    params, control, noise, dynamic = generate_samples(priors, controls, z0, Nc)
    # Organise samples into particles: {theta, xk, wk}
    samples = list()
    for i in range(Nc):
        theta = (params[:,i], control[:,i], noise[:,i])  # static variables
        xk = dynamic[:,i]  # dynamic states (ar_k, ai_k, nk) at time k
        wk = 0  # particle weight; FIXME: account for prior sample weight
        samples += [(theta, xk, wk)]
    return samples

# Register and store initial state samples
@bu.static_vars(samples0=None)
def store_initial_state_samples(samples=None, i=None):
    if samples is None:
        return store_initial_state_samples.samples0
    if i is None:
        store_initial_state_samples.samples0 = samples
    else:
        store_initial_state_samples.samples0[i] = samples

# Observation model for vector of Nx dynamic-state sub-vectors
def observe_model(x):
    Dx = 3  # FIXME: 3 is magic number (dimension of dynamic state)
    Nx = len(x) // Dx
    return lts.model_laser_intensity(x.reshape(Nx, Dx))

# N-step least-squares projection to obtain p(x_k+1 | x_k, z_k+1, ..., z_k+Nz)
def project_particle(xk, params, controls, noise, zN, dt, steps):
    # Return prediction p(x_k+1|x_k) and smoothed posterior p(x_k+1 | x_k, z_k+1, ..., z_k+Nz) and its EKF weight.
    # Notes:
    # 1- We predict forward until the first measurement zN[0], then
    # 2- We augment the state to maintain state at zN[0] and correlations to current state, then
    # 3- We predict forward the current state while updating correlations to zN[0] state.
    PREDICT, AUGMENT, PREDICT_COUPLED = range(3)
    status = PREDICT  # simple state-machine for different stages of multi-step projection
    dt_step = dt / steps  # period of each prediction-step, as a fraction of the period between measurements
    N = len(xk)
    Q = np.eye(N) * (noise[0] * dt_step)**2  # process noise
    R = np.eye(zN.shape[1]) * noise[1]**2  # measurement noise
    x, P, w = xk, np.zeros((N,N)), 0  # initialise pdf: we assume xk perfectly known
    for z in zN:
        # Prediction steps (there are "steps" predictions between each z)
        for _ in range(steps):
            F, fs = ils.jac_finite_diff((x[-N:], params, controls, dt_step), lel.process_model, i=0, offset=1e-5)
            Ppred = ils.triprod(F, P[-N:, -N:], F.T) + Q
            if status == PREDICT:  # simple EKF prediction
                x = fs
                P = Ppred
            elif status == AUGMENT:  # EKF state-augmentation; keeping old-state and its correlations to new-state
                x = np.append(x, fs)
                FP = np.dot(F, P)
                P = np.vstack((np.hstack((P, FP.T)), np.hstack((FP, Ppred))))  # FIXME: check this
                status = PREDICT_COUPLED
            else:  # EKF prediction of new-state, with maintenance of correlations to old recorded-state
                assert status == PREDICT_COUPLED
                x[-N:] = fs
                FP = np.dot(F, P[-N:, :N])  # note: uses the bottom-left quadrant of augmented P
                P[-N:, :N] = FP      # bottom-left quadrant
                P[:N, -N:] = FP.T    # top-right
                P[-N:, -N:] = Ppred  # bottom-right
        # Store prediction to x_k+1, and change status
        if status == PREDICT:
            x1pred, P1pred = x.copy(), P.copy()  # prediction to x_k+1 before update with z_k+1
            status = AUGMENT
        # do update
        H, z_pred = ils.jac_finite_diff([x[-N:]], observe_model, offset=1e-5)
        v = z - z_pred  # innovation
        index = ils.tuple2index([(-N,0)]) + len(x)  # index for x[-N:]
        w += ils.moment_form_update(x, P, v, R, H, idx=index, logflag=True)
    return x1pred, P1pred, x[:N], P[:N,:N], w


# Evaluate p(z|x)
def evaluate_likelihood(x, z, sigma_z):
    zpred = observe_model(x)
    R = np.eye(len(z)) * sigma_z**2
    return bu.gauss_evaluate(z-zpred, R, True)  # FIXME: improve efficiency; gauss_evaluate_scalar()


# Propagate a single particle up to the next barrier-time
def particle_propagate_to_barrier(sample, z, dt, predict_steps, look_ahead):
    theta, xk, wk = sample
    params, controls, noise = theta
    barrier_steps = z.shape[0] - look_ahead + 1
    if barrier_steps <= 0:  # end-condition; FIXME: check this
        barrier_steps = z.shape[0]
    intermediates = list()
    for i in range(barrier_steps):
        if i + look_ahead > z.shape[0]:  # end-condition; FIXME: check this
            look_ahead = z.shape[0] - i
        zN = z[i:i+look_ahead, :]
        xpred, Ppred, xup, Pup, wup = project_particle(xk, params, controls, noise, zN, dt, predict_steps)
        # Draw sample and apply weights
        xkp = bu.gauss_samples(xup, Pup, 1)[:,0]
        wpred = bu.gauss_evaluate(xkp-xpred, Ppred, True)
        wprop = bu.gauss_evaluate(xkp-xup, Pup, True)
        wz = evaluate_likelihood(xkp, zN[0,:], noise[1])
        wkp = wk + wpred + wz - wprop
        xk, wk = xkp, wkp
        intermediates += [(xk, wk)]  # record intermediate steps
    return theta, xk, wk, intermediates[:-1]  # [:-1] removes duplicate of final xk
# Note: wup = log p(zN) is (should be?) proportional to wpred + wz - wprop only if lookahead = 1
# It won't be exactly because wz has nonlinear calculation.


# Reject samples with very low weight, and replace by propagating new samples from the previous barrier pool
def partial_rejection_control(samples, parents, sample_barrier, z_packet, timestep_info):
    # Determine samples with insufficient weight
    wmax = max([s[2] for s in samples])
    w_accept = wmax - 500  # FIXME: what is a good cutoff
    ireplace = [i for (i,s) in enumerate(samples) if s[2]<w_accept]
    # Generate replacement samples, accepting only those with w > w_accept
    new_samples = list()
    n_reject = len(ireplace)
    while len(new_samples) < len(ireplace):
        # Draw sample from last barrier
        if sample_barrier is None:  # If sample_barrier is None, draw new samples from prior
            # FIXME: Maybe draw from modified prior, given info of weighted particles already drawn
            s = initialise_samples(1)[0]
            parent = s
        else: # Else draw samples from previous barrier
            i = np.random.randint(0, len(sample_barrier))
            s = sample_barrier[i]
            parent = i
        # Propagate to current barrier
        s_new = particle_propagate_to_barrier(s[:3], z_packet, *timestep_info)
        if s_new[2] > w_accept:
            new_samples += [(s_new, parent)]
        else:
            n_reject += 1
    if n_reject:
        wmax_new = max([s[0][2] for s in new_samples])
        print('Number of rejected samples: {0}. Old wmax {1}, New wmax {2}'.format(n_reject, wmax, wmax_new))
    # Replace low-weight samples with new samples
    for (i, r) in enumerate(ireplace):
        s_new, parent = new_samples[i]
        samples[r] = s_new
        if sample_barrier is None:  # register parent sample (from prior) with store
            store_initial_state_samples(parent, r)
        else:  # record change in parent-index
            parents[r] = parent
    return samples, parents, n_reject

#
# MAIN Filter function -----------------------------------------------
#

def smc_filter_main(controls, timeseries, priors, dt):
    # Configurables
    Nsamp = 1000  # number of samples in particle filter
    predict_steps = 2  # number of prediction steps to perform between each measurement
    look_ahead = 3  # smoothing look-ahead for generating proposal distribution
    barrier_steps = 10
    # Generate initial sample set, given priors etc
    samples = initialise_samples(Nsamp, priors, controls, timeseries[:,0])
    # Storage of state history
    store_initial_state_samples(samples)
    record_samples = list()
    # Setup a few more variables
    timestep_info = dt, predict_steps, look_ahead
    sample_barrier = None
    Nz = timeseries.shape[1]
    # Main loop
    for i in range(1, Nz, barrier_steps):
        print('Barrier at: ', i+barrier_steps-1)
        # Propagate samples to next barrier period
        packet_end = i + barrier_steps + look_ahead - 1
        if packet_end > Nz:
            packet_end = Nz
        z_packet = timeseries[:, i:packet_end].T
        snext = list()
        for s in samples:
            snext += [particle_propagate_to_barrier(s[:3], z_packet, *timestep_info)]
        samples = snext
        parents = np.arange(len(samples))  # index of parent samples of current samples
        # Perform rejection control to replace low-weight samples
        while True:
            samples, parents, reject = partial_rejection_control(samples, parents, sample_barrier, z_packet, timestep_info)
            if not reject:
                break
        # Record state
        record_samples += [(samples, parents)]
        # Create new barrier
        sample_barrier = samples.copy()
    samples0 = store_initial_state_samples()
    # FIXME: do plotting
    return record_samples, samples0

#
# Plotting and analysis -------------------
#

# Check along ancestor path that each parent has same constant terms as child.
def check_parent_path(record_samples, samples0, verbose=False):
    def constants_are_different(s1, s2):
        # Return True if s1 and s2 have different constant terms in theta
        theta1, theta2 = s1[0], s2[0]
        differences = [np.any(c1-c2) for (c1, c2) in zip(theta1, theta2)]
        return np.any(differences)
    okay = True
    for i in range(len(record_samples)):
        samples, parents = record_samples[i]
        samples_prev = record_samples[i-1][0] if i>0 else samples0
        for (j, (sample, parent)) in enumerate(zip(samples, parents)):
            if constants_are_different(sample, samples_prev[parent]):
                okay = False
                if verbose: print('Error: ', i, j)
    return okay

def plot_intensity_estimate(z, record_samples, samples0, i):
    # get x_1:k history for particle i at end of simulation
    xk = list()
    #import IPython; IPython.embed()
    for (samples, parents) in reversed(record_samples):
        thetar, xkr, wkr, intermr = samples[i]
        i = parents[i]
        xk += [xkr]
        if intermr:
            xk += [ii[0] for ii in reversed(intermr)]
    # Convert to intensity and plot
    xka = np.array(xk)
    for i in range(len(xkr)//3):
        plt.figure(); plt.grid()
        idx = slice(i*3, (i+1)*3)
        plt.plot(z[i,:])
        plt.plot(lts.model_laser_intensity(xka[:,idx]))

# Given a 2-D array, eliminate all duplicate rows
def unique_rows(a):
    # from: http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array/8567929#8567929
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def analyse_sample_depletion(record_samples):
    # Show depletion in terms of parents
    unique_parent_count = [np.unique(parents).shape[0] for (_, parents) in record_samples]
    plt.figure()
    plt.plot(unique_parent_count)
    # Show depletion in terms of sys-params
    unique_param_count = []
    for (samples, _) in record_samples:
        params = [s[0][0] for s in samples]
        unique_params = unique_rows(params)
        assert unique_params.shape[1] == 7
        unique_param_count.append(unique_params.shape[0])
    plt.figure()
    plt.plot(unique_param_count)

#
# RUN CODE ------------------
#

path = './smc_results/'

if __name__ == '__main__':
    # Record or load data
    if False:
        *controls, timeseries = lel.get_example_dataset()
        priors = lel.specify_priors()
        dt = lts.defaultMeasuredDt
        with open(path+'smcresult_data.pkl', 'wb') as f:
            pickle.dump((controls, timeseries, priors, dt), f)
    else:
        with open(path+'smcresult_data.pkl', 'rb') as f:
            controls, timeseries, priors, dt = pickle.load(f)

    # Run or analyse filter
    if True:
        record_samples, samples0 = smc_filter_main(controls, timeseries, priors, dt)
        with open(path+'smcresults_debug.pkl', 'wb') as f:
            pickle.dump((record_samples, samples0), f)
    else:
        with open(path+'smcresults_debug10_4.pkl', 'rb') as f:
            record_samples, samples0 = pickle.load(f)
        plot_intensity_estimate(timeseries, record_samples, samples0, i=1)
        analyse_sample_depletion(record_samples)
        plt.show()
