"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""

import numpy as np
import numpy.linalg
import numpy.typing as npt
from numpy.linalg import norm, svd, lstsq
from Gaussian.gaussian import gaussian, gaussian_first_der, gaussian_second_der

from numba import jit, float64, int32
import numba as nb

# constants for functions
# compute_curvature_2d:
MAX_ITER = 30
TOL_MOROZOV = 0.33333
EXP_MIN_INI = -16.0
EXP_MAX_INI = 2.0


@jit(float64(float64[:, :]), nopython=True)
def __curvature_from_circle(three_points: npt.ArrayLike) -> npt.ArrayLike:
    """
    This function is a fallback if estimating curvature using interpolation with RBFs does not work. This happens, if
    curvature is very strong such that the fault line cannot be considered a graph of a function after rotation. Here,
    we draw a circle through three consecutive points and use its radius for estimating curvature. The corresponding
    algorithm can be derived as follows:
    Let be a = (x1, y1), O = (x2, y2), b = (x3,y3) three distinct points and

          / x^2 +y^2   x  y  1\
          | x1^2+y1^2  x1 y1 1|
      A = | x2^2+y2^2  x1 y2 1|  .
          \ x3^2+y3^2  x3 y3 1/

    Then, direct computation by a Laplace expansion of A along the first row reveals that {det(A) = 0} is a circle
    through these points around (d2/2d1, -d3/2d1) with radius r = sqrt((d2^2+d3^2)/(4d1^2) + d4/d1), where
         |x1 y1 1|        |x1^2+y1^2 y1 1|         |x1^2+y1^2 x1 1|         |x1^2+y1^2 x1 y1|
    d1 = |x2 y2 1|,  d2 = |x2^2+y2^2 y2 1|,   d3 = |x2^2+y2^2 x2 1|,   d4 = |x2^2+y2^2 x2 y2|
         |x3 y3 1|        |x3^2+y3^2 y3 1|         |x3^2+y3^2 x3 1|         |x3^2+y3^2 x3 y3|
    Its reciprocal is the radius we are looking for.
    However, we apply this function to shifted coordinates such that (x2,y2) = 0. Therefore, all this simplifies to
    d1 = <a', b> with a' = (y1, -x1), d2 = y1 ||b||^2 - y3 ||a||^2, d3 = x1 ||b||^2 - x3 ||a||^2, d4 = 0, and ultimately

    curvature = 1/r = (2d1)/(||a||^2 ||b||^2 ||a-b||)                                                                (*)

    For the angle alpha between two vectors a and b, we have cos alpha = <a,b>/(||a|| ||b||) or
    cos(90- alpha) = <a', b>/(||a|| ||b||) =  0.5 d1

    Apart from scaling, computing the curvature with (*) becomes unstable, if both the nominator and the denominator
    tend to zero. Our analysis reveals that this happens, if approx. a = b (note that a' is perpendicular to a). This
    situation occurs however for consecutive points only if the resolution is far too coarse for the fault line or if
    the sorting is wrong. In any case, we need more points in this area. Therefore, we just return the value of the
    curvature, which should be rather large even if wrong to enforce further refinement in this region.

    Args:
        three points (npt.ArrayLike):
            set of three consecutive points. Shape: (npoints, ndim)

    Returns:
        the reciprocal of the radius (aka curvature) of the circle the three given points lie upon (float)

    """
    a = three_points[0, :]
    b = three_points[2, :]

    d1 = a[1] * b[0] - a[0] * b[1]

    # this numpy-based variant is much slower than computing things manually
    # curvature = 2*np.abs(d1)/(np.linalg.norm(a-b)*np.linalg.norm(a) * np.linalg.norm(b))

    curvature = 2 * np.abs(d1) / (
                np.linalg.norm(a - b) * np.sqrt((a[0] * a[0] + a[1] * a[1]) * (b[0] * b[0] + b[1] * b[1])))
    return curvature


@jit(float64[:, :](float64[:, :]), nopython=True)
def compute_dist_mat(point_set: npt.ArrayLike) -> npt.ArrayLike:
    """
   This function computes the distance matrix for the set of points in points.

    Args:
        point_set (npt.ArrayLike):
            set of points. Shape: (npoints, ndim)

    Returns:
        dist_mat (npt.ArrayLike):
            distance matrix. Shape: (npoints, npoints)

    """
    npoints, dim = point_set.shape

    dist_mat_aux = np.zeros((npoints, npoints, dim))

    for ipoint in range(npoints):
        dist_mat_aux[:, ipoint, :] = point_set - point_set[ipoint]

    dist_mat_aux = np.power(dist_mat_aux, 2)
    dist_mat = np.sum(dist_mat_aux, axis=2)
    dist_mat = np.sqrt(dist_mat)

    return dist_mat


@jit(float64(float64[:, :]), nopython=True)
def compute_maximal_angle(poly_line: npt.ArrayLike) -> float:
    """This function computes the maximal angle between consecutive line segments of a polygonal line given by the
    points in poly_line.

    Args:
        poly_line (npt.ArrayLike):
            ordered point set describing a non-closed polygon. Shape: (npoints, ndim)

    Returns:
        The maximum angle between two consecutive line segments (float)

    """
    # if the norm between consecutive points is smaller than eps, we consider them identical.
    eps = 1e-10

    # segments of the polygonal line
    segs = poly_line[:-1, :] - poly_line[1:, :]

    # the euclidean norms of the segments
    num_segs = segs.shape[0]
    norm_segs = np.zeros(num_segs)

    for iseg in range(num_segs):
        norm_segs[iseg] = norm(segs[iseg])

    # this is more elegant, but numba 0.54.1 does not support the axis-keyword in np.linalg.norm
    # norm_segs = norm(segs, axis=1)

    # if the euclidean norm between consecutive points is too small, we consider them duplicate
    duplicates = np.any(norm_segs < eps)

    assert not duplicates, 'In the point set, two consecutive points are (almost) identical.'

    # We compute the angle by its definition: angle(a,b) = arccos(<a,b>/(||a|| ||b||))
    denominator = norm_segs[:-1] * norm_segs[1:]

    nominator = np.sum(segs[0:-1, :] * segs[1:, :], axis=1)

    # min, as the cosine =1 for parallel segments and cosine=-1 for angles approaching pi
    max_angle = np.arccos(np.maximum(np.min(nominator / denominator), -1.0))

    return max_angle


@jit(nb.float64[:](float64[:, :]), nopython=True)
def estimate_normal(point_set: npt.ArrayLike) -> npt.ArrayLike:
    """This function provides a normal vector for the given point set by computing the optimally fitting plane or line
    in the sense that the sum of the squared distances between the points and the plane is minimal.
    % It is well known (see e.g. Shakarji, M.: Least-Squares Fitting Algorithms of the NIST Algorithm Testing System,
    % J. Res. Natl. Inst. Stand. Technol. 103, 633 (1998), https://nvlpubs.nist.gov/nistpubs/jres/103/6/j36sha.pdf)
    % that the optimal fitting plane contains the mean of the coordinates.
    % Therefore, we shift the point such that 0 is the new mean beforehand. The normal vector is the right singular
    vector with respect to the smallest singular value of the point set. As the singular values are provided in
    descending order, this is the "last" right singular vector. For an explanation and the algorithm itself, we refer
    to the aforementioned article.

    Args:
        point_set (npt.ArrayLike):
            set of points. Shape: (npoints, ndim)

    Returns:
        normed normal vector with respect to points (npt.ArrayLike). Shape: (ndim)

    """
    # number of points and their dimension
    npoints, dim = point_set.shape

    assert npoints >= dim, \
        "The number of points is less than the dimension. Computing a normal vector is impossible."

    # mean values of the points (coordinate-wise)
    x_mean = np.zeros(dim)
    for icoord in range(npoints):
        x_mean += point_set[icoord, :]

    x_mean = x_mean / npoints

    # variant using numpy is slower than the explicit loop when using numba
    # x_mean = (1 / npoints) * np.ones(npoints).dot(points)

    # actually, this is V^T, such that the right singular vector corresponding to the smallest singular value is the
    # last row, not the last column
    right_sing_vecs = svd(point_set - x_mean, full_matrices=False)[2]

    return right_sing_vecs[dim - 1, :]


@jit(float64[:](float64[:, :], float64[:, :], float64[:], float64, float64), nopython=True)
def reg_ls(int_mat: npt.ArrayLike, penalty_mat: npt.ArrayLike, rhs: npt.ArrayLike, eps: float,
           order: float) -> npt.ArrayLike:
    """
    This function solves a linear system of equations in the least-squares sense, regularized by the penalty matrix
    penalty_mat such that the residual is at most eps in any of its components.

    Args:
        int_mat (npt.ArrayLike):
            interpolation matrix. Shape: (npoints, npoints)
        penalty_mat (npt.ArrayLike):
            penalty matrix. Shape: (npoints, npoints)
        rhs (npt.ArrayLike):
            right-hand side
        eps (float):
            tolerance to fault line
        order (float):
            Order of the norm

    Returns:
        coefficients of the regularized least-squares-solution
    """

    npoints = int_mat.shape[0]

    int_t_int = np.zeros((npoints, npoints))
    rhs_ls = np.zeros(npoints)

    for i in range(npoints):
        for j in range(npoints):
            rhs_ls[i] += int_mat[j, i] * rhs[j]

            for k in range(npoints):
                int_t_int[i, j] += int_mat[k, i] * int_mat[k, j]

    # vectorized versions are slower than the loop when compiled with numba
    # rhs_ls = int_mat.T @ rhs
    # int_t_int_vec = int_mat.T @ int_mat

    # estimation of regularisation parameter due to Morozov
    # For performance reasons, we do not solve the resulting root finding problem for the regularization parameter mu
    # with simple bisection, but write mu = 10**exp and perform bisection on the exponent exp. We do not need to find
    # the exact root, but a reasonable approximation suffices. If the lower and upper bound of the exponent are smaller
    # than 1/3, we know the optimal mu up to a factor of 10^{1/3} = 2.15. This is precise enough for our purpose.
    expmin = EXP_MIN_INI
    expmax = EXP_MAX_INI
    nit = 0

    while expmax - expmin > TOL_MOROZOV:
        nit = nit + 1

        exp_new = 0.5 * (expmin + expmax)
        mu_new = 10 ** exp_new

        # this aux variable prevents recomputing the system matrix all the time
        work_mat = int_t_int + mu_new * penalty_mat

        # Depending on the regularisation parameter, it may happen in some cases that work_mat is numerically singular.
        # Then, np.linalg.solve raises an exception. Catching this in a try-catch-construct works, but prevents the code
        # from being compiled with numba. So, we switched from solve to lstsq, which is no performance hit, but even
        # faster in many cases.
        coeffs = lstsq(work_mat, rhs_ls, rcond=1e-15)[0]

        # res is the maximal absolute deviation from the interpolation value
        aux = np.zeros(npoints)
        for i in range(npoints):
            for j in range(npoints):
                aux[i] += int_mat[i, j] * coeffs[j]

        # the vectorized variant is actually slower than the loop when compiled with numba
        # res = norm(int_mat @ coeffs - rhs, np.inf)
        res = norm(aux - rhs, order)

        # the points are precise up to eps only
        if res > eps:
            expmax = exp_new
        else:
            expmin = exp_new

        # my is so small that do not need to regularize at all. If this is clear, we can stop.
        if mu_new < 1e-14:
            break

        # should never happen if the constants have their default values
        assert nit <= MAX_ITER, 'Maximal number of iterations reached for searching the regularization parameter.'

    return coeffs


@jit(nb.types.Tuple((nb.float64, nb.float64[:, :]))(nb.float64[:, :], int32, nb.float64), nopython=True)
def estimate_curvature_2d(point_set: npt.ArrayLike, idx: int, eps: float) -> [float, npt.ArrayLike]:
    """We assume that a planar curve is locally represented by a set of points, points. This function estimates
    the curvature in the point in points with index idx. The point set is assumed to be ordered, however, the
    points do not need to be equidistant.
    After shifting and rotation, we assume that the curve can be represented as a graph of an unknown function. It is
    straightforward to interpolate this graph using Gaussian Radial Basis Functions and to estimate the true curvature
    by the one of this interpolating function.
    In the context of fault detection, however, the points in points are located on the curve only up to a tolerance
    eps. Therefore, we do not interpolate, but penalise the second
    derivative of the interpolating RBF function subject to a maximal residual < eps. Note that the maximal residual
    coincides with the maximal deviation in the value at an interpolation point. As the points are known up to eps, it
    is pointless to interpolate more exactly. For computing this, we employ Tikhonov regularization with parameter
    estimation following Morozov.
    If the curve cannot be considered a graph even after rotation, we draw as a fallback a circle through the points
    with indices idx-1, idx and idx+1 and use its radius for estimating the curvature.
    It may happen that in case of extremely unevenly distributed points, the unregularised matrix A^TA is ill
    conditioned. Unevenly distributed points may occur: If the fault line has a kink, adaptive refinement/coarsening
    concentrates the points at the kink and eliminates points far from it. However, the regularisation with the
    second derivative reduces the condition number of the interpolation matrix to reasonable size, such that we can
    ignore possible warnings issued by the linear algebra library.

    Args:
        point_set (npt.ArrayLike):
            set of points. Shape: (npoints, ndim)
        idx (integer):
            index of the point in points at which to estimate the curvature
        eps (float):
            tolerance to fault line

    Returns:
        estimated curvature of line represented by points in point idx (float)
        coordinates of the left and right midpoints of idx on the RBF curve (npt.ArrayLike)
    """

    # approximation with RBFs
    npoints = point_set.shape[0]

    assert 0 <= idx < npoints, 'invalid value for idx (must be positive and < number of points)'

    assert point_set.shape[1] == 2, 'points in points must belong to R^2'

    assert npoints > 2, 'Not enough points (at least three needed) given for estimating the curvature'

    # we shift such that the point to estimate the curvature at becomes 0
    old_center = point_set[idx, :]
    point_set = point_set - old_center

    # orthogonal transformation: we consider the rightmost and leftmost point. Let alpha be the angle between the line
    # connecting these points and the x-axis. We turn all points in PointSet by -alpha degrees applying rot_mat with
    # c = cos(alpha) and s = sin(alpha). Of course, this a heuristic approach which can fail.
    cs = (point_set[-1, :] - point_set[0, :]) / norm(point_set[-1, :] - point_set[0, :])
    rot_mat = np.array([[cs[0], -cs[1]], [cs[1], cs[0]]])

    #  transform the point set (note that rotations preserve curvature)
    point_set = point_set @ rot_mat

    # We assume that, after rotation, the part of the line can be considered as a graph of a function. As the points
    # have been presorted, the x-values must be descending or ascending after transformation. If not, our assumption
    # was wrong. As a fallback, we compute the curvature from three consecutive points using a simple circle.
    if not (np.all(point_set[0:npoints - 1, 0] < point_set[1:npoints, 0]) or
            np.all(point_set[0:npoints - 1, 0] > point_set[1:npoints, 0])):

        if 0 < idx < npoints - 1:
            curvature = __curvature_from_circle(point_set[idx - 1:idx + 2, :])
            # Note that the coordinates of the point with index idx are shifted to 0.
            new_points = 0.5*np.vstack((point_set[idx-1, :], point_set[idx+1, :]))

        # in the case of the very first point: draw a circle through the first three points
        elif idx == 0:

            # the second of the three points must be 0 (this is NOT points[idx,:]), so center points around second
            # point
            aux_center = point_set[1, :]
            point_set = point_set - aux_center
            curvature = __curvature_from_circle(point_set[0:3, :])
            new_points = (0.5*point_set[0] + aux_center).reshape((-1, 2))

        # in the case of the very last point: draw a circle through the last three points
        elif idx == npoints-1:

            # the second of the three points must be 0 (this is NOT points[idx,:]), so center points around
            # second to last point
            aux_center = point_set[idx - 1, :]
            point_set = point_set - aux_center
            curvature = __curvature_from_circle(point_set[idx-2:npoints, :])
            new_points = (0.5*point_set[idx-2] + aux_center).reshape((-1, 2))

        new_points = new_points @ rot_mat.T + old_center
        return curvature, new_points

    # build interpolation matrix
    # It depends on the x-values of the rotated points only, the y-values contribute to the right-hand side
    int_mat = np.zeros((npoints, npoints))
    scale_vec = (abs((point_set[0, 0] - point_set[-1, 0])) / npoints) * np.ones(npoints)
    for i in range(npoints):
        int_mat[0:npoints, i] = gaussian(point_set[:, 0:1], point_set[i, 0:1], scale_vec[i])

    # we want to penalise the second derivative of the almost interpolating RBF function f, aka ||f''||^2. We
    # approximate this by \sum (f''(x_i))^2, where x_i denote the interpolation points. As
    # f''(x_i) = \sum coeff_j phi_j''(x_i), we can express the vector (f''(x_1), ..., f''(x_n)) by
    #
    # /f''(x_1)\   /phi_1''(x_1) ... phi_n''(x_1) \   /coeff_1\
    # |    .   |   |      .                .      |   |   .   |
    # |    .   | = |      .                .      | * |   .   |
    # |    .   |   |      .                .      |   |   .   |
    # \f''(x_n)/   \phi_1''(x_n) ... phi_n''(x_n) /   \coeff_n/
    #                    = B                            = x
    #
    # Therefore, we have ||f''||^2 \approx <Bx, Bx> = x^T (B^TB) x, such that penalty_mat = B^T B.
    penalty_mat = np.zeros((npoints, npoints))
    phi = np.zeros(npoints)

    for i in range(npoints):
        penalty_mat[0:npoints, i] = gaussian_second_der(point_set[:, 0:1], point_set[i, 0:1], scale_vec[i])[:, 0, 0]

    penalty_mat = penalty_mat.T @ penalty_mat

    # if the points are close together, penalty_mat is very large, albeit well conditioned. We normalise
    # penalty_mat in order to balance the size of int_mat and penalty_mat.
    penalty_mat = (10 / norm(penalty_mat, 2)) * penalty_mat

    coeffs = reg_ls(int_mat, penalty_mat, point_set[:, 1], eps, np.inf)

    if 0 < idx < npoints-1:
        # aux_pos = 0.5*np.array([points[idx-1,0:1], points[idx+1,0:1]])
        aux_pos = 0.5 * np.vstack((point_set[idx - 1, 0:1], point_set[idx + 1, 0:1]))
    elif idx == npoints-1:
        aux_pos = 0.5*point_set[idx-1, 0:1].reshape((-1, 1))
    elif idx == 0:
        aux_pos = 0.5*point_set[idx+1, 0:1].reshape((-1, 1))

    phi_val = np.zeros((aux_pos.shape[0], npoints))

    # compute first derivative in points[idx, :]
    for i in range(npoints):
        phi_val[:, i] = gaussian(aux_pos, point_set[i, 0:1], scale_vec[i])
        phi[i] = gaussian_first_der(point_set[idx:idx + 1, 0:1], point_set[i, 0:1], scale_vec[i])[0, 0]

    new_points = phi_val @ coeffs
    new_points = np.vstack((aux_pos[:, 0], new_points)).T@rot_mat.T + old_center
    phidot = phi @ coeffs

    # compute second derivative in points[idx, :]
    for i in range(npoints):
        phi[i] = gaussian_second_der(point_set[idx:idx + 1, 0:1], point_set[i, 0:1], scale_vec[i])[0, 0, 0]

    phiddot = phi @ coeffs

    curvature = abs(phiddot / (1 + phidot ** 2) ** 1.5)
    return curvature, new_points


if __name__ == '__main__':
    points_test = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
    normal_vec = estimate_normal(points_test)

    # MATLAB reference value is [0,1,0]
    print("normal_vec:", normal_vec)

    points_test = np.array([[0, 0], [1, 0], [1, 1], [2, 2], [1, 2]], dtype=np.float64)
    angle = compute_maximal_angle(points_test)

    # MATLAB reference value is 2.356194490192345
    print("angle:", angle)

    # MATLAB reference value is 0.632455532033676
    # curvature by fallback to __curvature_from_circle
    curv = estimate_curvature_2d(points_test, 2, 1e-3)
    print("curvature:", curv)

    # MATLAB reference value is 0.216877906163879
    points_test = np.array([[0, 0], [1, 0], [1, 1], [1, 2], [2, 3]], dtype=np.float64)

    curv = estimate_curvature_2d(points_test, 2, 1e-3)
    print("curvature:", curv)
