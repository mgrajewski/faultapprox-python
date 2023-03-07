"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
from warnings import warn

import numpy as np
import numpy.typing as npt
from scipy.linalg import norm
from typing import Tuple

from computeClassification import compute_classification
from newtriplets import start_pairs, triplets_by_bisection
from entities.Entities import FaultApproxParameters, ProblemDescr
from sort import sort_on_fault
from geometry import estimate_curvature_2d

SAFE_FACTOR = 1.5


def fill_2d(points_iclass: npt.ArrayLike, points_jclass: npt.ArrayLike, i_class: int, j_class: int,
            class_vals: npt.ArrayLike, fault_approx_pars: FaultApproxParameters, problem_descr: ProblemDescr,
            normal_vec_of_plane: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike, int, bool, npt.ArrayLike]:
    """
    It may happen that between two consecutive triplets on a fault line, there is a substantial gap. However, the
    maximal distance between consecutive triplets on a fault line is prescribed by the user setting the parameter
    fault_approx_pars.max_dist_for_surface_points. Therefore, we need to fill such gaps. This is what this function
    does.
    We sort the given triplets (aka point pairs) on the fault line and compute the distance between consecutive
    triplets. If we detect a gap, we create new equidistant point pairs. The starting points for bisection are computed
    by the estimated local curvature of the fault line. For creating them, we divide the straight line from the current
    existing point on the fault line to its successor equidistantly and deviate from it in normal direction
    symmetrically in both directions. The distance to the line is driven by the estimated curvature and safeguarded such
    that for a straight line, no further function evaluations apart from the ones for classification of the starting
    points should be necessary. This is for efficiency reasons.
    In the case of strong curvature, it may happen that by coincidence, a newly added point pair is very close to an
    existing one. Before adding a new point, we consider the minimal distance to the existing ones and add the point
    only if this distance is greater than max_dist_for_surface_points*min_dist_factor.
    We do not resort the triplets on the line after filling gaps but just append them to the arrays of existing points.

    Args:
        points_iclass (npt.ArrayLike):
            points in i_class approximating the fault line. Shape: (npoints, ndim)
        points_jclass (npt.ArrayLike):
            points in j_class approximating the fault line, counterpart to points_left. Shape: (npoints, ndim)
        i_class (int):
            class number
        j_class (int):
            class number (we consider the fault line between the i-th and j-the class)
        class_vals (npt.ArrayLike):
            Array containing the class values. These are not necessarily the class indices. Imagine that
            f(\Omega) = {1,2,5}. Then, class_vals = [1,2,5], whereas the class indices range from 0 to 2.
            Shape: (nclasses,)
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.
        normal_vec_of_plane (npt.ArrayLike):
            Necessary for future 3D computations

    Returns:  
        npt.ArrayLike: points_iclass
        npt.ArrayLike :points_jclass
        int: n_points
        bool: points_added
        bool: success
    """

    points_iclass = points_iclass.astype(float)
    points_jclass = points_jclass.astype(float)

    success = False

    max_dist_for_surface_points = fault_approx_pars.max_dist_for_surface_points
    eps_safemax = fault_approx_pars.alpha
    abstol_bisection = fault_approx_pars.abstol_bisection

    aux_vec = np.zeros(3, dtype=float)

    # true, if fill_gaps_in_fault_line added some triplets
    points_added = False

    idx_points_surf_ordered, sort_success = sort_on_fault(points_iclass, True, problem_descr, fault_approx_pars)

    if not sort_success:
        warn(f"fill_gaps_in_fault_line failed, i_class = {i_class} , j_class = {j_class}")
        return points_iclass, points_jclass, points_iclass.shape[0], points_added, success

    # resort point sets
    points_iclass = points_iclass[idx_points_surf_ordered[0]]
    points_jclass = points_jclass[idx_points_surf_ordered[0]]

    n_points, ndim = points_iclass.shape

    # Here, the triplets on the fault line are ordered. Fill gaps, if necessary.
    i_idx_next_segment = 0
    idx_segment = np.ones(n_points, dtype=int) * -1
    start_points_left = np.zeros((2 * n_points - 1, ndim), dtype=float)
    start_points_right = np.zeros((2 * n_points - 1, ndim), dtype=float)

    # distance to the next point on the fault line
    dist_next = np.linalg.norm(points_iclass[:-1, :] - points_iclass[1:, :], axis=1)

    for i_point in range(n_points - 1):

        if dist_next[i_point] > max_dist_for_surface_points:

            # number of points to fill the gap
            num_subdivisions = int(np.ceil(dist_next[i_point] / max_dist_for_surface_points))
            idx_segment[i_point] = i_idx_next_segment
            i_idx_next_segment += num_subdivisions - 1

            parameters = np.arange(1, num_subdivisions, 1) / num_subdivisions
            parameters = parameters[np.newaxis, :]

            points_on_line = (1 - parameters).T * 0.5 * (points_iclass[i_point, :] + points_jclass[i_point, :]) + \
                              0.5 * parameters.T * (points_iclass[i_point + 1] + points_jclass[i_point + 1, :])

            if start_points_left.shape[0] < i_idx_next_segment:
                start_points_left = np.concatenate(
                    (start_points_left, np.zeros((2 * n_points - 1, ndim), dtype=float)), axis=0)
                start_points_right = np.concatenate(
                    (start_points_right, np.zeros((2 * n_points - 1, ndim), dtype=float)), axis=0)

            if i_idx_next_segment <= start_points_left.shape[0]:
                start_points_left[idx_segment[i_point]:i_idx_next_segment] = points_on_line
            else:
                start_points_left_new = np.zeros((i_idx_next_segment, ndim))
                start_points_left_new[0:start_points_left.shape[0], :] = start_points_left
                start_points_left_new[idx_segment[i_point]:i_idx_next_segment] = points_on_line
                start_points_left = start_points_left_new

            aux_vec[0:ndim] = points_iclass[i_point + 1, :] - points_iclass[i_point, :]

            # compute unit normal vector
            normal_vec = np.cross(aux_vec, normal_vec_of_plane)
            normal_vec = normal_vec / norm(normal_vec)

            # For 2 dimensions: use refined method of computing starting values based upon estimated curvature.
            if ndim == 2:
                if 1 < i_point < n_points - 3:
                    curv = estimate_curvature_2d(0.5 * (points_iclass[i_point - 2:i_point + 3] +
                                                        points_jclass[i_point - 2:i_point + 3]), 2, abstol_bisection)[0]
                # Estimating the curvature is most reliable for the midpoint of the subset, therefore, we enlarge its
                # value by an additional safety factor here.
                elif i_point <= 1:
                    if n_points > 2:
                        curv = SAFE_FACTOR * estimate_curvature_2d(0.5 * (points_iclass[0:min(n_points, 5)] +
                                                                          points_jclass[0:min(n_points, 5)]), 1,
                                                                   abstol_bisection)[0]
                    else:
                        curv = 0
                # In this case, ipoint >= 3, such that we can compute the curvature in any case.
                else:
                    curv = SAFE_FACTOR * estimate_curvature_2d(
                        0.5*(points_iclass[max(0, n_points - 5):n_points] +
                             points_jclass[max(0, n_points - 5):n_points]), 3, abstol_bisection)[0]

            else:
                if 1 < i_point < n_points - 3:
                    curv = estimate_curvature_2d(0.5 * (points_iclass[i_point - 2:i_point + 3, ~normal_vec_of_plane] +
                                                        points_jclass[i_point - 2:i_point + 3, ~normal_vec_of_plane]), 2,
                                                 abstol_bisection)[0]
                elif i_point <= 1:
                    if n_points > 2:
                        curv = SAFE_FACTOR * estimate_curvature_2d(
                            0.5*(points_iclass[0:min(n_points, 5), ~normal_vec_of_plane] +
                                 points_jclass[0:min(n_points, 5), ~normal_vec_of_plane]), 1, abstol_bisection)[0]
                    else:
                        curv = 0
                else:
                    curv = SAFE_FACTOR * estimate_curvature_2d(
                        0.5*(points_iclass[max(0, n_points - 5):n_points, ~normal_vec_of_plane] +
                             points_jclass[max(n_points - 5):n_points, ~normal_vec_of_plane]), 3, abstol_bisection)[0]

            # Some elaborate computation shows that the maximal deviation from a straight line between two points is
            # 0.25*curv*dist^2 + 1/16*curv^3.*dist^4 * O(h^6) for a planar curve with curvature curv. However, our
            # points are known up to abstol_bisection only. Therefore, we should at least consider this deviation from
            # a straight line. If the estimation of the curvature is way too large, we fall back to the old heuristics
            # eps_safemax*dist as deviation.
            error_ind = 0.25 * curv * dist_next[i_point] ** 2 + 1.0 / 16 * curv ** 3. * dist_next[i_point] ** 4
            error_ind = np.minimum(eps_safemax * dist_next[i_point], np.maximum(0.95 * abstol_bisection, error_ind))

            points_aux = points_on_line + error_ind * normal_vec[0:ndim]

            # Ensure that points_aux is inside the domain.
            points_aux = np.minimum(np.maximum(problem_descr.x_min + fault_approx_pars.eps, points_aux),
                                    problem_descr.x_max - fault_approx_pars.eps)

            if i_idx_next_segment <= start_points_right.shape[0]:
                start_points_right[idx_segment[i_point]:i_idx_next_segment] = points_aux
            else:
                start_points_right_new = np.zeros((i_idx_next_segment, ndim))
                start_points_right_new[0:start_points_right.shape[0], :] = start_points_right
                start_points_right_new[idx_segment[i_point]:i_idx_next_segment] = points_aux
                start_points_right = start_points_right_new

            points_aux = points_on_line - error_ind * normal_vec[0:ndim]

            # Ensure that points_aux is inside the domain.
            points_aux = np.minimum(np.maximum(problem_descr.x_min + fault_approx_pars.eps, points_aux),
                                    problem_descr.x_max - fault_approx_pars.eps)
            start_points_left[idx_segment[i_point]:i_idx_next_segment] = points_aux

    # Shorten and create vectors accordingly.
    start_points_left = start_points_left[0:i_idx_next_segment]
    start_points_right = start_points_right[0:i_idx_next_segment]

    # If there are no points to add at all, skip the computation.
    if start_points_left.shape[0] == 0:
        success = True
        return points_iclass, points_jclass, n_points, points_added, success

    aux_arr = compute_classification(np.concatenate((start_points_left, start_points_right), axis=0),
                                     problem_descr)

    class_left = aux_arr[0:i_idx_next_segment]
    class_right = aux_arr[i_idx_next_segment:aux_arr.shape[0]]

    # Find points near the fault line, each with a counterpart in the opposite class.
    point_pairs, idx_success, class_points_success = start_pairs(start_points_left,
                                                                 start_points_right,
                                                                 class_left, class_right,
                                                                 class_vals[i_class],
                                                                 class_vals[j_class],
                                                                 problem_descr)
    # Compute points near the fault line by bisection.
    points_left, points_right, finished = triplets_by_bisection(point_pairs[idx_success, 0:ndim],
                                                                point_pairs[idx_success, ndim:2 * ndim],
                                                                class_points_success[idx_success, 0],
                                                                class_points_success[idx_success, 1],
                                                                problem_descr, fault_approx_pars)

    # Compute array with the indices of the points where triplets_by_bisection succeeded in the complete array of
    # points.
    aux = np.arange(0, idx_success.shape[0])
    idx_success = aux[idx_success]

    # Avoid adding almost duplicate points.
    for i in range(finished.shape[0]):

        if finished[i]:

            # Compute distance to all points on the fault line.
            dist_vec = points_iclass - points_left[i]

            dist_vec = np.power(dist_vec, 2)
            for i_dim in range(1, ndim):
                dist_vec[:, 0] += dist_vec[:, i_dim]
            dist_vec = dist_vec[:, 0]

            if np.sqrt(np.min(dist_vec)) > \
                    fault_approx_pars.max_dist_for_surface_points * fault_approx_pars.min_dist_factor:
                n_points += 1
                points_added = True

                if class_points_success[idx_success[i], 0] - 1 == i_class:
                    points_iclass = np.concatenate((points_iclass, points_left[i].reshape(1, -1)))
                    points_jclass = np.concatenate((points_jclass, points_right[i].reshape(1, -1)))
                else:
                    points_iclass = np.concatenate((points_iclass, points_right[i].reshape(1, -1)))
                    points_jclass = np.concatenate((points_jclass, points_left[i].reshape(1, -1)))

    success = True
    return points_iclass, points_jclass, n_points, points_added, success


def divide_into_components(point_set: npt.ArrayLike, max_dist_for_surface_points: float) -> \
        Tuple[npt.ArrayLike, int, npt.ArrayLike]:
    """
    This function separates a given fault line in its components. To do so, it relies on the fact that points is
    ordered. This functions applies to 2D only. The function detects several components, if two subsequent points have
    a mutual distance greater than maxDistForSurfacePoints. Note that this function is called AFTER algorithm fill.
    Args:
        point_set (npt.ArrayLike):
            set of points representing the fault. Shape: (num_points, ndim)
        max_dist_for_surface_points (float):
            maximal distance between triplets on a fault

    Returns:
        segments (npt.ArrayLike)
        num_of_segs (int)
        num_points_per_seg (npt.ArrayLike)

    """
    num_points = point_set.shape[0]
    dist = point_set[0:num_points - 1] - point_set[1:num_points]
    dist[:, 0] = np.sum(np.power(dist, 2), axis=1)
    dist[:, 0] = np.sqrt(dist[:, 0])
    seg_aux = dist[:, 0] > 3 * max_dist_for_surface_points
    num_of_segs = (seg_aux[seg_aux > 0]).shape[0] + 1

    num_points_per_seg = np.zeros(num_of_segs, dtype=int)
    seg_end = np.flatnonzero(seg_aux > 0)

    segments = np.empty(num_of_segs, dtype=object)

    seg_start = 0
    for i_seg in range(num_of_segs - 1):
        segments[i_seg] = point_set[seg_start:seg_end[i_seg] + 1]
        num_points_per_seg[i_seg] = seg_end[i_seg] - seg_start + 1
        seg_start = seg_end[i_seg] + 1

    segments[num_of_segs - 1] = point_set[seg_start:num_points]
    num_points_per_seg[num_of_segs - 1] = num_points - seg_start
    return segments, num_of_segs, num_points_per_seg


# just for testing
def main():
    print('YNI.')


if __name__ == '__main__':
    main()
