
# Test the numerical accuracy of various LS solvers.
# TODO: trial QR based solvers.

import numpy as np
import scipy.linalg as sci

# Requires At == A.T, not simply A
def normal_equation_solve_v1(At, b):
    from scipy.linalg import cho_factor, cho_solve
    Y = np.dot(At, At.T)
    L = cho_factor(Y)  # CAUTION, not triangular (off-triangle side filled with random values, see SciPy docs)
    return cho_solve(L, np.dot(At, b))

def ls_solve(A, b, type=0):
    # Simple solvers
    if type == 0:
        x = sci.solve(A, b)
    elif type == 1:
        Ainv = np.linalg.inv(A)  # works only if A invertible
        x = np.dot(Ainv, b)
    # Normal equation solvers
    Y = np.dot(A.T, A)
    #c = np.linalg.cond(Y)
    y = np.dot(A.T, b)
    if type == 2:
        L = sci.cho_factor(Y) # CAUTION, not triangular (off-triangle side filled with random values, see SciPy docs)
        x = sci.cho_solve(L, y)
    elif type == 3:
        L = np.linalg.cholesky(Y)
        x = sci.cho_solve((L, True), y)
    elif type == 4:
        L = sci.cholesky(Y, lower=True)
        x = sci.cho_solve((L, True), y)
    elif type == 5:
        x = sci.solve(Y, y, sym_pos=True)
    elif type == 6:
        L = sci.cholesky(Y, lower=True)  # or L = np.linalg.cholesky(Y)
        f = sci.solve_triangular(L, y, lower=True)
        x = sci.solve_triangular(L, f, lower=True, trans=1)
        #x = sci.solve_triangular(L.T, f, lower=False)  # note, lower=False is default
    elif type == 7:
        L = sci.cholesky(Y, lower=True)  # or L = np.linalg.cholesky(Y)
        f = sci.solve(L, y, lower=False)  # note, using lower = False or True here makes no difference (see docs, only has effect is sym_pos=True)
        x = sci.solve(L.T, f)
    elif type == 8:
        L = np.linalg.cholesky(Y)
        f = np.linalg.solve(L, y)
        x = np.linalg.solve(L.T, f)
    return x


def test_solvers(s = 0):
    M, N, nominal = 3, 3, 0
    assert M>=N  # must be square or over-determined
    np.random.seed(s)
    A = np.random.rand(M, N)
    b = np.random.rand(M)
    xnom = ls_solve(A, b, nominal)
    for type in range(9):
        try:
            x = ls_solve(A, b, type)
            print(x, sum((x-xnom)**2))
        except Exception as e:
            print('Type', type, 'failed:', str(e))

#
#
#

for s in range(20):
    test_solvers(s)
