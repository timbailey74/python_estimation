# Test finite-difference Jacobian versus autograd. 
# Test for vector and scalar problems

import autograd.numpy as np
import autograd as aut

from pyestimate.utilities import pi2pi
import pyestimate.calculus as cal


# Model 1: Range-bearing sensor
def rbmodel(xv, xt):
    # range-bearing model, with xv=(x,y,p), xt=(x,y)
    dx = xt[0]-xv[0]
    dy = xt[1]-xv[1]
    return np.array(
        [np.sqrt(dx**2 + dy**2),        # range
         np.arctan2(dy, dx) - xv[2]])   # bearing

def rbdiff(delta):
    # model diff for range-bearing; Caution: pass-by-reference
    delta[1] = pi2pi(delta[1])
    return delta

def generate_rb_data(seed=None):
    if seed is None:
        xv = np.array([3, -5., 2])  # note, must be floating-point array
        xt = np.array([2., 1])
    else:
        if not seed:
            seed = np.random.randint(4294967295)
            print('Seed: ', seed)
        np.random.seed(seed)
        xv = (np.random.rand(3) - 0.5) * 10
        xt = (np.random.rand(2) - 0.5) * 10
    return [xv, xt]


# Model 2: Simple 1-D system
model1d = lambda x, w : x[0] + w*(x[1]-x[0])


#
# TESTS --------------
#

def test_numerical_jacobians(seed=0):
    x = generate_rb_data(seed)
    for i in range(len(x)):
        jacc = cal.jacobian(rbmodel, i, dmodel=rbdiff)
        Jc = jacc(*x)
        jaca = aut.jacobian(rbmodel, i)
        Ja = jaca(*x)
        assert np.all(np.isclose(Jc, Ja))


def test_numerical_jacobians_2():
    x = np.array([2, 5.])
    w = np.array([7.])
    for i in range(len(x)):
        jacc = cal.jacobian(model1d, i)
        Jc = jacc(x, w)
        jaca = aut.jacobian(model1d, i)
        Ja = jaca(x, w)
        assert np.all(np.isclose(Jc, Ja))


def test_numerical_jacobians_sin():
    x = [0, 1, 2]
    J, y = jac_finite_diff([x], np.sin)
    # FIXME: np.sin won't work with autograd.jacobian
