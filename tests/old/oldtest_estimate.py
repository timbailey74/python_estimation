#
#
import numpy as np
import matplotlib.pyplot as plt

import pyestimate.information_form as info
import pyestimate.moment_form as mom
from pyestimate.utilities import index_other

def test_constrain_marginalise():
    # Check that simultaneous constrain-marginalise algorithm produces same result as
    # sequential constrain then marginalise.
    def randm(M,N,K):
        x = np.random.uniform(-K, K, (M,N))
        if M == N:
            x = np.dot(x, x.T)
        return x
    Nx = 40
    Nz = 20
    Nrem = 35

    # Make random data
    iremove = list(set(np.random.randint(0, Nx, Nrem)))
    xs = randm(Nx,1,5)
    y = randm(Nx,1,5)
    Y = randm(Nx,Nx,3)
    R = randm(Nz,Nz,2)
    zs = randm(Nz,1,2)
    Hs = np.random.uniform(-3,3, (Nz,Nx))

    # Simultaneous constrain-marginalise
    (yu, Yu) = info.constrain_and_marginalise(y, Y, R, xs, zs, Hs, iremove)

    # Compare against sequence of constrain then marginalise
    yc = y.copy()
    Yc = Y.copy()
    zz = np.zeros((zs.size, 1))
    info.update_zeromean_noise(yc, Yc, zz, R, xs, zs, Hs, np.array(range(yc.size)))
    (ym, Ym) = info.marginalise(yc, Yc, index_other(yc.size, iremove))
    print('Yu: ', 1 / np.linalg.cond(Yu))
    print('Ym: ', 1 / np.linalg.cond(Ym))
    print('Yc: ', 1 / np.linalg.cond(Yc))
    plt.plot(ym)
    plt.plot(yu)
    plt.figure()
    plt.plot(ym-yu)
    plt.figure()
    plt.plot(Ym.flatten()-Yu.flatten())
    plt.show()

#
#
#

if __name__ == "__main__":
    print('test_constrain_marginalise()')
    test_constrain_marginalise()

