#

#import sys, os
#sys.path.insert(0, os.path.abspath(__file__+"/../.."))

import numpy as np
import matplotlib.pyplot as plt

from pyestimate.utilities import pi2pi
from pyestimate.calculus import jac_finite_diff, jac_central_diff, \
    multi_step_crank_nicolson, multi_step_scipy, lorenz


def test_numerical_jacobians():
    def rbmodel(xv, xt):
        # range-bearing model, with xv=(x,y,p), xt=(x,y)
        dx = xt[0]-xv[0]
        dy = xt[1]-xv[1]
        return np.array(
            [np.sqrt(dx**2 + dy**2),        # range
             np.arctan2(dy, dx) - xv[2]])   # bearing

    def rbdiff(e):
        # model diff for range-bearing; Caution: pass-by-reference
        e[1] = pi2pi(e[1])
        return e

    xv = np.array([3, -5., 2])  # note, must be floating-point array
    xt = np.array([2., 1])
    x = [xv, xt]
    Hv, z = jac_central_diff(x, rbmodel, rbdiff, i=0)
    Ht, _ = jac_central_diff(x, rbmodel, rbdiff, i=1)
    #print("z = ", z, "\nHv =\n", Hv, "\nHt =\n", Ht)
    print('z = \n{0}\nHv = \n{1}\nHt = \n{2}'.format(z, Hv, Ht))

def test_numerical_jacobians_2():
    #x = np.array([0., 1, 2])
    x = [0, 1, 2]
    J, y = jac_finite_diff([x], np.sin)
    print('\n\nx = {0}\ny = {1}\nJ = \n{2}'.format(x, y, J))


def test_ode():
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10, 250)
    x0 = [0.1, 0.1, 0.1]
    x_crk = multi_step_crank_nicolson(lorenz, t, x0, ())
    x_sci = multi_step_scipy(lorenz, t, x0, ())
    #plt.plot(t, x_crk, '.-', t, x_sci, '.-')
    #plt.show()


if __name__ == "__main__":
    flags = 0xff # test all
    if flags & 0x01:
        print('test_numerical_jacobians()')
        test_numerical_jacobians()
        test_numerical_jacobians_2()
    if flags & 0x02:
        print('test_ode()')
        test_ode()

