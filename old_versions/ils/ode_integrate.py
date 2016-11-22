# ODE integration methods; eg., Crank-Nicolson
#
"""
This module implements several simple ODE solvers, as well as a basic wrapper
of the Scipy ODE solver, so that they have a common interface. Note, the Scipy
solver is not currently compatible with autograd, but the other methods are.
"""

import numpy as np
import scipy.integrate as scint
import scipy.linalg as scalg
from ils.linear_algebra import jac_finite_diff

# Integrate timeseries via Scipy ODE solvers; eg., dop853, dopri5, lsoda
def create_scipy_integator(model, t0, x0, args, type='dop853', rtol=1e-5):
    solver = scint.ode(model)
    solver.set_f_params(*args)
    solver.set_integrator(type, rtol=rtol)
    solver.set_initial_value(x0, t0)
    return solver

# Single-step integration using 'solver' argument
def step_scipy_solver(solver, dt):
    return solver.integrate(solver.t + dt)

# Single-step integration, compatible with other interfaces below
def step_scipy(model, t, x, dt, args, type='dop853', rtol=1e-5):
    solver = create_scipy_integator(model, t, x, args, type, rtol)
    return solver.integrate(t + dt)

# Multi-step time-series integration
def multi_step_scipy(model, t, x0, args, type='dop853', rtol=1e-5):
    solver = create_scipy_integator(model, t[0], x0, args, type, rtol)
    return np.array([x0] + [solver.integrate(tk) for tk in t[1:]])

# Simple 4th-order Runge-Kutta integration - Adapted from rk4() in matplotlib/mlab.py
# https://searchcode.com/codesearch/view/72072974/
def step_rk4(model, t, x, dt, args):
    dt2 = dt / 2.0
    k1 = model(t, x, *args)
    k2 = model(t + dt2, x + dt2*k1, *args)
    k3 = model(t + dt2, x + dt2*k2, *args)
    k4 = model(t + dt,  x + dt *k3,  *args)
    return x + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

def multi_step_rk4(model, t, x0, args):
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        x[i+1, :] = step_rk4(model, t[i], x[i, :], dt, args)
    return x

# Forward-difference single-step integration model; xnext = f(t, x, dt, args)
def step_fdiff(f, t, x, dt, args):
    return x + f(t, x, *args)*dt

# Integrate timeseries using forward-difference integration (cheap but unstable and inaccurate)
def multi_step_fdiff(model, t, x0, args):
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        x[i+1, :] = step_fdiff(model, t[i], x[i,:], dt, args)
    return x

# Crank-Nicolson single-step integration model; z = 0 = x + f(xeffective)*dt - xnext, where xeff = x*w + xnext*(1-w)
def crank_nicolson_step_twosided(f, t, x, xnext, dt, args, w=0.5):
    xeffective = x*w + xnext*(1-w)
    teffective = t + (1-w)*dt
    return x + f(teffective, xeffective, *args)*dt - xnext

# Crank-Nicolson single-step integration *without* specification of xnext
# Generates predictive integration of xnext via iterative refinement of crank_nicolson_step_twosided() model.
def step_crank_nicolson(model, t, x, dt, args, w=0.5):
    xnext = step_fdiff(model, t, x, dt, args)  # initial guess at xnext via forward-difference method
    for _ in range(2):  # iterative linearised least-squares refinement
        H, zs = jac_finite_diff((model, t, x, xnext, dt, args, w), crank_nicolson_step_twosided, i=3)  # FIXME: replace with autograd derivative
        v = np.dot(H, xnext) - zs
        y = np.dot(H.T, v)
        Y = np.dot(H.T, H)
        xnext = scalg.solve(Y, y, sym_pos=True)
    return xnext

# Integrate timeseries using Crank-Nicolson integration (cheaper but less accurate than dop853)
def multi_step_crank_nicolson(model, t, x0, args, w=0.5):
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        x[i+1, :] = step_crank_nicolson(model, t[i], x[i, :], dt, args, w)
    return x


#
# Example ODE models ----------------------------------------
#

# Lorenz differential equation by Alistair Reid NICTA 2014
def lorenz(t, state, a=10., b=8./3., c=28.):
    x, y, z = state
    dxdt = a*(y-x)
    dydt = x*(c-z) - y
    dzdt = x*y - b*z
    return np.array([dxdt, dydt, dzdt])

# TODO: add a test case that is dependent on t

#
# TEST code ----------------------------------------
#

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10, 250)
    x0 = [0.1, 0.1, 0.1]
    x_crk = multi_step_crank_nicolson(lorenz, t, x0, ())
    x_sci = multi_step_scipy(lorenz, t, x0, ())
    plt.plot(t, x_crk, '.-', t, x_sci, '.-')
    plt.show()
