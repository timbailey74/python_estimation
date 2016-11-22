# Differentiation and integration

# TODO: http://stackoverflow.com/questions/9876290/how-do-i-compute-derivative-using-numpy
# - finite and central differencing
# - relative and absolute offset size
# - ridders
# - polynomial fit (Chebechev)
# - http://deeplearning.net/software/theano/

import numpy as np
import scipy.integrate as scint
import scipy.linalg as scalg



# Use finite differencing to compute the Jacobian of the i-th array in list (x)
def jac_finite_diff(x, model, dmodel=None, i=0, backward=False, offset=1e-7):
    if backward:
        offset = -offset
    if not dmodel:
        dmodel = lambda dy: dy # nominal diff-model
    y = model(*x)
    xi = x[i] # beware, shallow copy
    lenx, leny = len(xi), len(y) # alternatively, len = np.size(x[i]) or len = x[i].size
    J = np.zeros((leny, lenx))
    for j in range(lenx):
        xj = xi[j]
        xo = xj + offset
        xi[j] = xo
        yo = model(*x)
        xi[j] = xj
        J[:,j] = dmodel(y-yo) / (xj-xo)
    return J, y

# Use central differencing scheme to compute the Jacobian of the i-th array in list (x)
def jac_central_diff(x, model, dmodel=None, i=0, offset=1e-7):
    if not dmodel:
        dmodel = lambda dy: dy # nominal diff-model
    y = model(*x)
    xi = x[i]               # beware, shallow copy
    lenx, leny = len(xi), len(y)
    J = np.zeros((leny, lenx))  # note, tuple input
    for j in range(lenx):
        xj = xi[j]          # record old value
        xu = xj + offset
        xl = xj - offset
        xi[j] = xu          # compute upper
        yu = model(*x)
        xi[j] = xl          # compute lower
        yl = model(*x)
        xi[j] = xj          # recover old value
        J[:,j] = dmodel(yu-yl) / (xu-xl)
    return J, y

# Class: Simplified Jacobian interface that behaves like autograd.jacobian
class jacobian_class:
    def __init__(self, model, i=0, central_diff=False, dmodel=None, *args, **kwargs):
        self.model = model
        self.dmodel = dmodel
        self.i = i
        self.numjac = jac_central_diff if central_diff else jac_finite_diff
        self.args, self.kwargs = args, kwargs
    def __call__(self, *args):
        J, self.value = self.numjac(args, self.model, self.dmodel, self.i, *self.args, **self.kwargs)
        return J
    # Notes:
    # 1. Can access function value as: my_jacobian.value
    # 2. Can change my_jacobian.i to differentiate wrt different terms

# Closure: Simplified Jacobian interface that behaves like autograd.jacobian
def jacobian_function(model, i=0, central_diff=False, dmodel=None, *args, **kwargs):
    numjac = jac_central_diff if central_diff else jac_finite_diff
    def evaluate(*args_f):
        nonlocal numjac
        J, _ = numjac(args_f, model, dmodel, i, *args, **kwargs)
        return J
    return evaluate

# Alias to make the closure version the default (ie., to have same name as autograd version)
jacobian = jacobian_function

#
# Integration ------------------
#


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

