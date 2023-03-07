"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
import numpy as np
import numpy.typing as npt
from numba import jit, float64, boolean

EPS_DIST_SQ = 1e-8
EPS_DET = 1e-10


@jit(boolean(float64[:, :], float64[:, :]), nopython=True)
def is_on_poly_line(poly_line: npt.ArrayLike, points_test: npt.ArrayLike) -> bool:
    """This function checks if any point in points_test (in the following called test point) is on the polygonal line
    defined by poly_line. It works for two dimensions only.
    If the test point is on the polygonal line, it is on one of the line segments (or very close to one). This is the
    case, if the orthogonal projection of the test point on the corresponding line is on the line segment and if the
    distance to the corresponding line is zero or very small. The orthogonal projection on the line defined by
    ref_point[i], ref_point[i+1] is, setting line_seg = ref_point[i+1] - ref_point[i]:

    p(test point) = ref_point[i] + alpha * line_seg ,    alpha = <test point, line_seg> /norm(line_seg)^2

    If 0 <= alpha <= 1, the projection is on line_seg, and we compute the distance to the line segments in this case.

    Args:
        poly_line (npt.ArrayLike):
            point set defining a polygonal line. Shape: (npoints, 2)
        points_test (npt.ArrayLike)
            points to test. Shape: (npoints, 2)

    Returns:
        True if a point in points_test is (almost) on the polygonal line, False otherwise.

    """

    # tolerance for square of segment length
    eps_len_sq = 1e-20
    on_line_segment = False

    # works in 2D only
    assert (poly_line.shape[1] == 2), 'Function is_on_poly_line works for 2D only.'

    # number of points in num_points_poly and num_points_test, resp.
    num_points_poly = poly_line.shape[0]
    num_points_test = points_test.shape[0]

    # line segments (at the same time the direction vectors of the corresponding lines)
    line_segs = poly_line[1:num_points_poly, :] - poly_line[0:num_points_poly - 1, :]

    for ipoint in range(num_points_test):
        for iseg in range(num_points_poly - 1):

            # square of the length of line segment
            seg_length_sq = line_segs[iseg, 0] * line_segs[iseg, 0] + line_segs[iseg, 1] * line_segs[iseg, 1]

            # The projection-based test is reliable only for segments longer than eps_len. If the segment is too small,
            # we just skip this line segment und continue with the next one.
            if abs(seg_length_sq) > eps_len_sq:
                aux_vec = points_test[ipoint] - poly_line[iseg]

                # When using numba, this explicit variant is considerably faster than a numpy-based implementation.
                alpha = (aux_vec[0] * line_segs[iseg, 0] + aux_vec[1] * line_segs[iseg, 1]) / seg_length_sq

                # If so, the projected test point is on the current line segment.
                if 0 <= alpha <= 1:
                    dist = points_test[ipoint] - poly_line[iseg] - alpha * line_segs[iseg]

                    # square of the distance
                    dist_sq = dist[0] * dist[0] + dist[1] * dist[1]
                    if dist_sq < EPS_DIST_SQ:
                        on_line_segment = True
                        return on_line_segment
            else:
                continue

    return on_line_segment


@jit(boolean(float64[:, :], float64[:, :], boolean), nopython=True)
def poly_lines_intersect(poly_1: npt.ArrayLike, poly_2: npt.ArrayLike, mode_self_int: bool) -> bool:
    """This function tests, if two polygonal lines given by the points in poly_1 and poly_2, which are assumed to be
    ordered, intersect.

    We test any line segment from the first line with any each line segment of the second one if they intersect.
    This can be efficiently done using determinants:
                  i+1
                  x
                 /
                /
     j x-------/------------------x j+1
              /
             /
            x
           i
    The determinant in 2D determines whether three points are sorted clockwise or counterclockwise. If
    det((i,i+1), (i,j)) and det((i,i+1), (j+1,i)) are both positive of both negative, then both points j and j+1 are
    both left or both right from the line segment from i to i+1. Then, the line segment from i to i+1 cannot intersect
    the line segment from j to j+1. However, if the signs differ, this does not necessarily mean that the two line
    segments intersect:
     j x-----------------------x j+1
             i+1 x
                /
               /
            i x
    So, if j and j+1 are on different sides with respect to the line segment from i to i+1, then we repeat our test
    for points i and i+1 with respect to the line segment from j to j+1.
    This function works in two dimensions only.
    Args:
        poly_1 (npt.ArrayLike):
            first point set. Shape: (npoints,2)
        poly_2 (npt.ArrayLike):
            first point set. Shape: (npoints,2)
        mode_self_int (bool):
            should be set to true for testing self intersection of one polygonal line, as is enables some performance
            optimisations

    Returns:
        bool: True if lines intersect, False otherwise.

    """

    do_intersect = False

    assert (poly_1.shape[1] == 2 and poly_2.shape[1] == 2), 'Function poly_lines_intersect works for 2D only.'

    num_points_1 = poly_1.shape[0]
    num_points_2 = poly_2.shape[0]

    if num_points_1 < 2 or num_points_2 < 2:
        return do_intersect

    line_segs_1 = poly_1[0:num_points_1 - 1] - poly_1[1:]
    line_segs_2 = poly_2[0:num_points_2 - 1] - poly_2[1:]

    # This is actually the l1-norm. We scale the line segments for numerical stability.
    norm_line_segs_1 = np.abs(line_segs_1)
    norm_line_segs_1 = norm_line_segs_1[:, 0] + norm_line_segs_1[:, 1]

    for iseg in range(num_points_1 - 1):
        line_segs_1[iseg] = line_segs_1[iseg] / norm_line_segs_1[iseg]

    for iseg in range(num_points_1 - 1):
        if mode_self_int:
            iterator = range(iseg + 1, num_points_2 - 1)
        else:
            iterator = range(num_points_2 - 1)

        for jseg in iterator:
            # If greater 0: all points of the other segment on the same side.
            aux1 = (line_segs_1[iseg, 0] * (poly_1[iseg, 1] - poly_2[jseg, 1]) -
                    line_segs_1[iseg, 1] * (poly_1[iseg, 0] - poly_2[jseg, 0])) * \
                   (line_segs_1[iseg, 0] * (poly_1[iseg, 1] - poly_2[jseg + 1, 1]) -
                    line_segs_1[iseg, 1] * (poly_1[iseg, 0] - poly_2[jseg + 1, 0]))

            # Exclude the geometric situation shown above: test, if i and i+1 are on different sides with respect to the
            # line segment from j to j+1.
            if aux1 < -EPS_DET:
                aux2 = (line_segs_2[iseg, 0] * (poly_2[jseg, 1] - poly_1[iseg, 1]) -
                        line_segs_2[iseg, 1] * (poly_2[jseg, 0] - poly_1[iseg, 0])) * \
                       (line_segs_2[iseg, 0] * (poly_2[jseg, 1] - poly_1[iseg + 1, 1]) -
                        line_segs_2[iseg, 1] * (poly_2[jseg, 0] - poly_1[iseg + 1, 0]))

                if aux2 < -EPS_DET:
                    do_intersect = True
                    break

    return do_intersect


@jit(boolean(float64[:, :]))
def self_intersection(points: npt.ArrayLike) -> bool:
    """This function tests, if a polygonal line given by the points in points, which are assumed to be ordered,
    intersects itself. This can be a hint for failed sorting. This function is a wrapper for line_segs_intersect.

    Args:
        points (npt.ArrayLike):
            point defining the polygonal line to test. Shape: (npoints, 2)

    Returns:
        bool: True if no intersection, False otherwise.

    """

    return poly_lines_intersect(points, points, True)


def component_closed(points: npt.ArrayLike, idx_points: npt.ArrayLike, dist_vec: npt.ArrayLike, n_points: int,
                     mode: int) -> bool:
    """
    This function tests if a boundary component is closed after adding a point to prevent endless loops in expand_2d. We
    test if any point on a remote part of the fault line is in fact close to the last point. To do so, we exploit that
    the points on the fault line are already ordered. We take the first or last half of the points on the fault line
    depending on if we expand the start or the end of the fault line. We compute their distance to the last point. In
    (sub)algorithm expand, this is the point added at last. If this distance is small, we conclude that the fault line
    must be closed.
    This does not make sense for boundary components consisting of 6 points only or even less.

    Args:
        points (npt.ArrayLike):
            points defining the segment. Shape: (n_points, 2)
        idx_points (npt.ArrayLike):
            index array, points[idx_points] is sorted. Shape: (n_points,)
        dist_vec (npt.ArrayLike):
            array containing the distance between consecutive points in points. Shape: (n_points-1,)
        n_points (int):
            number of points in points
        mode (int):
            Mode of operation in expand (0: expand the current starting point, 1: extend it beyond the current end)

    Returns:
        bool: True if the segment is closed, False otherwise.
    """
    closed_comp = False
    n_dim = points.shape[1]

    if n_points > 6:

        aux_points = np.min((n_points - 1, np.floor(0.5 * n_points))).astype(int)

        if mode == 0:
            aux_max = int(n_points - 1)
            aux_min = int(np.max((0, aux_max - aux_points)))
            dist_to_test_points = points[idx_points[aux_min:aux_max], :] - points[-1, :]
            dist_vec_test = dist_vec[idx_points[aux_min:aux_max]]
        else:
            dist_to_test_points = points[idx_points[0:aux_points], :] - points[-1, :]
            dist_vec_test = dist_vec[idx_points[0:aux_points]]

        # Compute the distance of the new points to the test set.
        dist_to_test_points = np.power(dist_to_test_points, 2)

        for idim in range(1, n_dim):
            dist_to_test_points[:, 0] = dist_to_test_points[:, 0] + dist_to_test_points[:, idim]

        dist_to_test_points[:, 0] = np.sqrt(dist_to_test_points[:, 0])

        # The factor 0.7 has the following reason: if two segments really overlap, the minimal distance is at most
        # 0.5*max_dist_for_surface_points, if a point is on the line segment between two consecutive points. However,
        # as the fault line may be curved, the true minimal distance may be a little larger.
        if np.any(dist_to_test_points[:, 0] < 0.7 * dist_vec_test):
            closed_comp = True

        return closed_comp


if __name__ == '__main__':
    poly = np.asarray([[0, 0], [1, 1], [1, 2]], dtype=np.float64)
    points = np.asarray([[0.5, 0.8], [3, 3], [3, 5]])
    print('is_on_poly_line:', is_on_poly_line(poly, points))

    poly = np.asarray([[0, 0], [1, 1], [1, 2], [2, 3], [2, 4], [3, 5]], dtype=np.float64)
    points = np.asarray([[0.5, 1], [3, 3], [3, 5], [4, 7]])
    print('poly_lines_intersect', poly_lines_intersect(points, poly, False))

    points = np.asarray([[0.5, 1], [3, 3], [3, 5]])
    print('is_on_poly_line', is_on_poly_line(poly, points))

    print('self_intersection', self_intersection(poly))
