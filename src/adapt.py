"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
from warnings import warn

import numpy as np
import numpy.typing as npt
from scipy.linalg import norm
from typing import Tuple
from numba import jit, float64, int32
import numba as nb

from geometry import estimate_curvature_2d, compute_maximal_angle
from computeClassification import compute_classification
from newtriplets import start_pairs, triplets_by_bisection
from checkPoints import self_intersection
from entities.Entities import FaultApproxParameters, ProblemDescr


def adapt_2d(point_sets_surface: npt.ArrayLike, n_points_surf: npt.ArrayLike, i_class: int,
             j_class: int, i_comp: int, problem_descr: ProblemDescr, fault_approx_pars: FaultApproxParameters,
             class_vals: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """This function realizes (sub)algorithm adapt. It locally refines and coarsens a set of triplets representing the
    i_comp-th component of the fault line \Gamma_{i_class,j_class} according to the estimated error of a polygonal
    approximation of the fault based on the given set of triplets.  Hereby, we always rely on the "medium" points of the
    triplets.

    Args:
        point_sets_surface (npt.ArrayLike):
            Array of arrays containing the point sets which represent the faults.
        n_points_surf (npt.ArrayLike):
            Array of arrays containing the number of points per set
        i_class (int):
            class index
        j_class (int):
            class index
        i_comp (int):
            index of the component of \Gamma_{i_class,j_class}
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.
        class_vals (npt.ArrayLike):
            Array containing the class values. These are not necessarily the class indices. Imagine that
            f(\Omega) = {1,2,5}. Then, class_vals = [1,2,5], whereas the class indices range from 0 to 2.
            Shape: (nclasses,)

    Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike]:
        point_sets_surface, n_points_surf
    """

    # aux array for adaptive insertion of points
    i_idx_try = np.array([0, 1, -1, 2, -2])

    # maximal admissible angle between subsequent line segments
    max_admissible_angle = 3.1

    # maximal number of adaptive refinement/coarsening steps
    max_iter_adapt = fault_approx_pars.max_iter_adapt

    # for safeguarding the search for initial starting pairs for bisection
    eps_safemax = fault_approx_pars.alpha

    # dimension
    points_surf = point_sets_surface[i_class, j_class][i_comp]
    n_dim = points_surf.shape[1]

    for i_iter_adapt in range(max_iter_adapt):

        points_surf = point_sets_surface[i_class, j_class][i_comp]
        num_points = n_points_surf[i_class, j_class][i_comp]

        # Euclidean length of each line segment.  The line segments are created by connecting the "medium" points of
        # subsequent triplets.
        seg_length = norm((points_surf[0:num_points - 1] - points_surf[1:num_points]).T, axis=0)

        if num_points > 2:
            # Compute the curvature and new starting points for creating starting pairs in case of refinement.
            curvature, start_points = __curv_and_new_pts(num_points,
                                                         0.5 * (points_surf +
                                                                point_sets_surface[j_class, i_class][i_comp]),
                                                         fault_approx_pars.abstol_bisection)
        else:
            break

        # We approximate the maximal curvature of a segment with the maximum of the ones at its start- and end point.
        curv_aux = np.maximum(curvature[0:-1], curvature[1:])

        # The error indicator applies to the approximation with line segments. The approximation with RBFs used for
        # finding starting values for bisection is far better.
        # As heuristic, we just take the higher order error contribution as estimation of the deviation of the
        # true curve from the RBF approximation.
        err_ind_hot = 1 / 16 * np.power(curv_aux, 3) * np.power(seg_length, 4)
        err_ind = 0.25 * curv_aux * np.power(seg_length, 2) + err_ind_hot

        # As the position of the "medium" points is accurate up to abstol_bisection only, it makes sense to further
        # refine only line segments sufficiently longer than this.
        # The factor 3 is due to Pythagoras and the fact that the distance between two approximations of the same point
        # on the fault line is at most 2*abstol_bisection.
        segs_to_refine = np.logical_and(err_ind > fault_approx_pars.err_max,
                                        seg_length > 3.0 * fault_approx_pars.abstol_bisection)
        if np.any(segs_to_refine):
            # We aim at inserting new triplets in the approximate middle of selected line segments.

            # Hack for using circular shift (if the last segment is to be refined, this last index is shifted at the
            # beginning of the vector.) We avoid this by adding False at the end. Note that the number of points exceeds
            # the number of line segments by one.
            segs_to_refine = np.append(segs_to_refine, [False])
            if num_points > 3:
                mid_points = start_points[segs_to_refine[:-1], :]
            else:
                mid_points = 0.5 * (points_surf[np.roll(segs_to_refine, 1)] + points_surf[segs_to_refine])

            normals = points_surf[np.roll(segs_to_refine, 1)] - points_surf[segs_to_refine]
            normals[:, 1] *= -1
            normals = np.flip(normals, axis=1)
            norm_normals = np.sqrt(np.power(normals[:, 0], 2) + np.power(normals[:, 1], 2))
            normals = np.divide(normals, norm_normals.reshape(-1, 1))

            # We want to find triplets in the approximate middle of two consecutive triplets on the fault line by
            # bisection. For this, we need starting values. We have a heuristic estimation for how much a point on the
            # true fault line may deviate from its RBF approximation (this is what err_ind_hot estimates). Therefore, we
            # can use this information for getting reasonable starting values for bisection. However, we need to
            # safeguard this guess by eps_safemax*dist (if the curvature is largely overestimated) and the tolerance for
            # bisection, as points are on any fault only up to this tolerance.
            # If this heuristic fails, start_pairs still finds valid starting pairs in almost all cases, but at the cost
            # of additional function evaluations.
            alpha = np.minimum(eps_safemax * seg_length[segs_to_refine[:-1]],
                               np.maximum(err_ind_hot[segs_to_refine[:-1]],
                                          0.95 * fault_approx_pars.abstol_bisection))
            points_right = mid_points + normals * alpha.reshape((-1, 1))
            points_left = mid_points - normals * alpha.reshape((-1, 1))

            # Ensure that points_right is inside the domain.
            points_right = np.minimum(np.maximum(problem_descr.x_min + fault_approx_pars.eps, points_right),
                                      problem_descr.x_max - fault_approx_pars.eps)

            # Ensure that points_left is inside the domain.
            points_left = np.minimum(np.maximum(problem_descr.x_min + fault_approx_pars.eps, points_left),
                                     problem_descr.x_max - fault_approx_pars.eps)

            i_idx_to_refine = np.flatnonzero(segs_to_refine == 1)
            aux_arr = compute_classification(np.concatenate((points_right, points_left), axis=0), problem_descr)
            class_right = aux_arr[0:points_right.shape[0]]
            class_left = aux_arr[points_left.shape[0]:]

            point_pairs, idx_ok, class_points_ok = start_pairs(points_right, points_left,
                                                               class_right, class_left,
                                                               class_vals[i_class],
                                                               class_vals[j_class],
                                                               problem_descr)

            points_right, points_left, finished = triplets_by_bisection(point_pairs[idx_ok, 0:n_dim],
                                                                         point_pairs[idx_ok, 2:2 * n_dim],
                                                                        class_points_ok[idx_ok, 0],
                                                                        class_points_ok[idx_ok, 1],
                                                                        problem_descr, fault_approx_pars)

            aux = np.arange(0, idx_ok.shape[0])
            idx_ok = aux[idx_ok]

            triplets_added = 1
            for i in range(idx_ok.shape[0]):
                insert_ok = False
                if finished[i]:
                    # insert new points
                    if class_points_ok[i, 0] == class_vals[i_class]:
                        new_point_i_class = points_left[i]
                        new_point_j_class = points_right[i]
                    elif class_points_ok[i, 0] == class_vals[j_class]:
                        new_point_j_class = points_left[i]
                        new_point_i_class = points_right[i]
                    # It may happen that a new valid triplet has been found, which however belongs to another fault
                    # line. Then, we skip this triplet and move on to the next one. This part of the implementation
                    # should be improved.
                    else:
                        continue

                    # It is not given that if we intend to insert a new triplet in between two consecutive triplets i
                    # and i+1, that the new triplet found is indeed in between these consecutive triplets.
                    # Therefore, we test this. If this is not the case, we test some of the neighbouring line segments
                    # for insertion if they exist. If this still fails, we give up.
                    for i_test in range(i_idx_try.shape[0]):
                        i_shift = i_idx_try[i_test]

                        i_idx_start = i_idx_to_refine[idx_ok[i]] + triplets_added + i_shift - 1

                        # if the corresponding line segment exists
                        if n_points_surf[i_class, j_class][i_comp] >= i_idx_start + 2 and i_idx_start >= 0:
                            points_test = np.concatenate(
                                (points_surf[0:i_idx_start + 1],
                                 new_point_i_class.reshape(1, -1), points_surf
                                 [i_idx_start + 1:n_points_surf[i_class, j_class][i_comp]]), axis=0)

                            # We want to insert a triplet "somewhere" in the middle between two subsequent triplets
                            # with indices i_idx_start and i_idx_start+1.However, we need to ensure that the new
                            # triplet is indeed somewhere in the middle. For doing so, we compare the distance of the
                            # new triplet to the given ones relative to their distance.
                            # If min_dist is smaller than some threshold, then the new triplet is very close to one of
                            # the existing triplets.
                            min_dist = np.minimum(np.linalg.norm(points_surf[i_idx_start, :] - new_point_i_class, 2),
                                                  np.linalg.norm(points_surf[i_idx_start + 1, :] - new_point_i_class,
                                                                 2))

                            min_dist = min_dist / np.linalg.norm(points_surf[i_idx_start, :] -
                                                                 points_surf[i_idx_start + 1, :], 2)

                            # If min_dist is not in the range given here, the new triplet is not in the middle at all,
                            # such that we do not insert it.
                            # The range is usually violated in case of wrong sorting only.
                            if 0.2 < min_dist < 1.0:

                                # max_angle = compute_maximal_angle(points_test)
                                max_angle = compute_maximal_angle(
                                    points_test[max(i_idx_start - 2, 0):min(i_idx_start + 3, points_test.shape[0]), :])
                                no_intersection = not self_intersection(points_test)

                                if no_intersection and max_angle < max_admissible_angle:
                                    points_surf = points_test
                                    point_sets_surface[j_class, i_class][i_comp] = np.concatenate((
                                        point_sets_surface[j_class, i_class][i_comp][0:i_idx_start + 1],
                                        new_point_j_class.reshape(1, -1),
                                        point_sets_surface[j_class, i_class][i_comp]
                                        [i_idx_start + 1:n_points_surf[j_class, i_class][i_comp]]), axis=0)

                                    # If we have found a suitable segment, we can stop.
                                    insert_ok = True
                                    break

                    if insert_ok:
                        n_points_surf[i_class, j_class][i_comp] += 1
                        # num_points_surf[j_class, i_class][i_comp] might reference
                        # num_points_surf[i_class, j_class][i_comp], in this case we only count up
                        # n_points_surf[i_class, j_class][i_comp]
                        if n_points_surf[i_class, j_class] is not n_points_surf[j_class, i_class]:
                            n_points_surf[j_class, i_class][i_comp] += 1
                        triplets_added += 1
                    else:
                        warn(
                            f'adding triplets in adaptive refinement failed for a fault line between classes '
                            '{i_class} and {j_class}')

            num_points = n_points_surf[i_class, j_class][i_comp]

            seg_length = norm((points_surf[0:num_points - 1] - points_surf[1:num_points]).T, axis=0)

            # coarsening step
            if num_points > 2:
                # curvature on any triplet
                curvature = \
                    __curv_and_new_pts(num_points, 0.5 * (points_surf + point_sets_surface[j_class, i_class][i_comp]),
                                       fault_approx_pars.abstol_bisection)[0]
                # We approximate the maximal curvature of a line segment with the maximum of the ones at its start- and
                # end point.
                curv_aux = np.maximum(curvature[0:-1], curvature[1:])
            else:
                break

            # If points have been added, recompute the error indicator per line segment.
            err_ind = 0.25 * curv_aux * np.power(seg_length, 2) + \
                1 / 16 * np.power(curv_aux, 3) * np.power(seg_length, 4)

        # --- coarsening step ---

        # Delete a triplet, if the estimated error of both line segments the triplet belongs to is smaller than err_min.
        points_to_remove = err_ind < fault_approx_pars.err_min
        points_to_remove = np.logical_and(points_to_remove[0:-1], points_to_remove[1:])

        # Heuristic: Never delete the first or the last triplet.
        points_to_remove = np.append(points_to_remove, [False])
        points_to_remove = np.append([False], points_to_remove)

        if not np.any(np.any(segs_to_refine)) and not np.any(points_to_remove):
            break

        # Heuristic: Never delete consecutive triplets.
        for i in range(1, points_to_remove.shape[0] - 1):
            points_to_remove[i] = np.logical_and(points_to_remove[i], ~points_to_remove[i - 1])

        points_surf = points_surf[~points_to_remove]
        point_sets_surface[j_class, i_class][i_comp] = point_sets_surface[j_class, i_class][i_comp][~points_to_remove]
        n_points_surf[i_class, j_class][i_comp] -= np.flatnonzero(points_to_remove).shape[0]

        # num_points_surf[j_class, i_class][i_comp] might reference num_points_surf[i_class, j_class][i_comp], in this
        # case we only count up n_points_surf[i_class, j_class][i_comp]
        if n_points_surf[i_class, j_class] is not n_points_surf[j_class, i_class]:
            n_points_surf[j_class, i_class][i_comp] -= np.flatnonzero(points_to_remove).shape[0]

        point_sets_surface[i_class, j_class][i_comp] = points_surf

    # This is no duplicate, but necessary in case of an early exit in the loop.
    point_sets_surface[i_class, j_class][i_comp] = points_surf

    return point_sets_surface, n_points_surf


@jit(nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(int32, float64[:, :], float64), nopython=True)
def __curv_and_new_pts(num_points: int, points_surf: npt.ArrayLike, abstol_bisection: float) -> \
        Tuple[float, npt.ArrayLike]:
    curvature = np.zeros(num_points)
    start_points = np.zeros((num_points - 1, 2))

    # For all "inner" points: consider five points for computing the curvature and the starting points for creating
    # starting pairs for bisection.
    # This applies for all fault components containing at least five points.
    for i_point in range(2, num_points - 1):
        points_local = points_surf[i_point - 2:i_point + 3]
        curvature[i_point], start_aux = estimate_curvature_2d(points_local, 2, abstol_bisection)
        start_points[i_point - 1:i_point + 1] = start_points[i_point - 1:i_point + 1] + start_aux

    start_points[2:num_points - 2] = 0.5 * start_points[2:num_points - 2]

    # Consider first, second, last and second but last points, if feasible.
    if num_points >= 4:
        points_local = points_surf[0:min(5, num_points)]
        curvature[0] = estimate_curvature_2d(points_local, 0, abstol_bisection)[0]
        curvature[1], start_aux = estimate_curvature_2d(points_local, 1, abstol_bisection)
        start_points[0] = start_aux[0]
        points_local = points_surf[-min(5, num_points):]
        curvature[-2], start_aux = estimate_curvature_2d(points_local, min(5, num_points) - 2, abstol_bisection)
        curvature[-1] = estimate_curvature_2d(points_local, min(5, num_points) - 1, abstol_bisection)[0]
        start_points[-1] = start_aux[1]
    # Fault component consists of three points only: compute the curvature based upon these three points.
    elif num_points == 3:
        points_local = points_surf
        curvature[0] = estimate_curvature_2d(points_local, 0, abstol_bisection)[0]
        curvature[1], start_points = estimate_curvature_2d(points_local, 1, abstol_bisection)
        curvature[2] = estimate_curvature_2d(points_local, 2, abstol_bisection)[0]

    # Two points only: We can not obtain any curvature information, so we just return 0 and as starting value the
    # midpoint.
    else:
        start_points[0] = 0.5 * (points_surf[0, :] + points_surf[1, :])

    return curvature, start_points
