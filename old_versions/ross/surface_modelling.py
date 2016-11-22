
#

import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import eigh
from scipy.linalg import solve

from basic_utilities import sample_mean

# Fit a plane to the set of 3-D points in x, which is composed of column-vectors
# Define plane by mean-point xm and normal-vector n
def fit_plane(x):
    xm, P = sample_mean(x)
    eval, evec = eigh(P)
    assert eval[0] == min(eval)
    n = evec[:, 0]  # normal to plane is e-vector with smallest e-value
    return xm, n

# Alternative formulation for fitting plane to 3-D points
# Defines plane by p, where p[0]*x + p[1]*y + p[2] = z. This form gives non-
# unit normal vector (p[0], p[1], -1).
def fit_plane_coefficients(x):
    x, y, z = x[0,:], x[1,:], x[2,:]
    At = np.vstack((x, y, np.ones(len(x))))
    #p = normal_equation_solve_linear(At, z)  # plane parameters: p[0]*x + p[1]*y + (-1)*z + p[2] = 0
    #p = np.dot(np.linalg.inv(At.T), z)  # this seems to be more accurate than the above; FIXME: why?
    p = solve(At.T, z)  # better than the above
    #zp = p[0]*x + p[1]*y + p[2]
    return p  #, zp
# Function might also return zp, which is the projection of z onto the plane given
# (x,y) -- ie., vertical projection.

# Convert point-normal-form of a plane to coefficient-form
def point_normal_to_coefficient(x, n):
    p01 = -n[:2] / n[2]  # p[0],p[1] makes a normal vector of the form (p[0], p[1], -1)
    p2 = x[2] - p01[0]*x[0] - p01[1]*x[1]
    return np.append(p01, p2)

# Given frame (xr, Rr) defined relative to a base frame (xb, Rb), compute its
# global frame (xg, Rg). That is, put it in the same coordinate frame as the base.
def transform2global(xr, Rr, xb, Rb):
    xg = np.dot(Rb, xr) + xb
    Rg = np.dot(Rb*Rr)
    return xg, Rg

# Given two frames in the same coordinate system (x1, R1) and (x2, R2), transform
# the former relative to the latter (ie., make x2 the base-frame).
def transform2relative(x1, R1, x2, R2):
    R2t = R2.T  # rotation matrices are orthogonal; transpose equals inverse
    xr = np.dot(R2t, x1 - x2)
    Rr = np.dot(R2t*R1)
    return xr, Rr

# Compute normal vector of triangle
def triangle_normal(a, b, c, unit=False):
    n = np.cross(b-a, c-a)
    if unit: n /= np.sqrt(sum(n**2))
    return n

# Point to plane location and squared-distance.
# Where plane (x,n) is defined by a point-on-plane and a normal vector.
def point_to_plane(q, x, n, unit=False):
    t = ts = np.dot(n, q-x)
    if not unit: ts /= sum(n**2)
    v = ts * n
    p = q - v
    d2 = t * ts
    #d2 = sum(v**2)  # Note: t*ts == sum(v**2) == dist_sqr(p,q)
    return p, d2

# Vertical projection of point onto plane; using coefficient-form of plane
def point_to_plane_vertical(q, p):
    try:
        return p[0]*q[0,:] + p[1]*q[1,:] + p[2]
    except:
        return p[0]*q[0] + p[1]*q[1] + p[2]
    # FIXME: do we want q in row-form or column-form?

# Point to line, where line (x, v) is defined by a point on the line and a vector along the line.
# See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
def point_to_line(q, x, v, unit=False):
    t = np.dot(v, q-x)
    if not unit: t /= sum(v**2)
    p = x + t*v
    return p

# Project (x,y) coordinates onto line, h; so we get mapping (x,y,z) -> (h,z)
def project_points_to_major_axis(top, bot, order=False):
    xm, P = sample_mean(top[:,:2].T)
    eval, evec = eigh(P)
    assert eval[0] < eval[1]
    if evec[0,1] < 0: evec = -evec  # make x-axis projection positive
    tt = np.dot(evec[:,1], (top[:,:2]-xm).T)  # unit e-vector-based projection
    tb = np.dot(evec[:,1], (bot[:,:2]-xm).T)
    top_hv = np.vstack((tt, top[:,2]-top[0,2])).T  # FIXME: can we avoid all these transposes?
    bot_hv = np.vstack((tb, bot[:,2]-top[0,2])).T
    # Arrange points into x-order
    if order:
        i = np.argsort(tt)
        return top_hv[:, i], bot_hv[:, i], i
    else:
        return top_hv, bot_hv


#
class TriangulatedSurface:
    def __init__(self, points):
        self.tri = Delaunay(points[:, :2])  # FIXME: assumes points are row-vectors; do we want this?
        self.x = points
    def find_triangle(self, q):  # FIXME: make this function work for lists of queries
        i = self.tri.find_simplex(q[:2])
        return self.tri.simplices[i]
    def find_triangle_points(self, q):
        idx = self.find_triangle(q)
        try: #if len(idx.shape) == 1:
            return self.x[idx, :]
        except: #else:
            return [self.x[i,:] for i in idx]  # FIXME: check this works for q = list of one point
    def interpolate_intercept(self, q, vertical=True):  # FIXME: make this function work for lists of queries
        xtri = self.find_triangle_points(q)
        xm, n = fit_plane(xtri.T)
        c = point_normal_to_coefficient(xm, n)
        #c = fit_plane_coefficients(xtri.T)
        if vertical:  # vertical intercept
            interp = q.copy()
            interp[2] = point_to_plane_vertical(q, c)
        else:  # normal intercept
            interp = point_to_plane(q, xm, n, True)[0]
        return interp

#
# TEST CODE -------------------------------
#

def test_point_to_plane():
    N = 10
    x = np.vstack((np.random.rand(2, N), np.zeros(N)))
    x = np.random.rand(3, N)
    q = np.random.rand(3)
    xm, n = fit_plane(x)
    n *= 3.2
    p1, d1 = point_to_plane(q, xm, n)
    v = p1 - xm
    p2, d2, t = point_to_line(q, xm, v)

if __name__ == "__main__":
    test_point_to_plane()
