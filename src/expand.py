"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
from warnings import warn

import numpy as np
import numpy.typing as npt

from scipy.linalg import norm
from typing import Tuple

from checkPoints import component_closed
from computeClassification import compute_classification
from newtriplets import start_pairs, single_triplet_by_bisection
from entities.Entities import FaultApproxParameters, ProblemDescr
from geometry import estimate_curvature_2d, reg_ls

GROWTH_FACTOR = 1.5


def __eval_poly(t, coeffs, q_mat, x_mean):
    ndim = x_mean.shape[0]
    n_points = t.shape[0]

    deg = coeffs.shape[0] - 1

    int_mat = np.zeros((n_points, deg + 1))
    for i in range(deg + 1):
        int_mat[:, i] = t ** i

    x = int_mat @ coeffs

    if ndim == 2:
        x = np.vstack((t, x)).T @ q_mat.T + x_mean
    else:
        x = np.vstack((t, x, np.zeros(n_points, 1))).T @ q_mat + x_mean

    return x


def expand_2d(left_domain: npt.ArrayLike, idx_points: npt.ArrayLike, i_class: int, j_class: int,
              i_seg: int, class_vals: npt.ArrayLike,
              points_i_class: npt.ArrayLike, points_j_class: npt.ArrayLike, normal_vec_plane: npt.ArrayLike,
              problem_descr, fault_approx_pars: FaultApproxParameters, mode: int) -> \
                        Tuple[npt.ArrayLike, npt.ArrayLike, int, npt.ArrayLike, npt.ArrayLike, bool, bool]:
    """
    This function extends a fault line up to its end.
    
    We assume that the points on the fault line are ordered. There are two modes of operation: mode 0 for extending the
    fault line before its current starting point and mode 1 for extending it beyond its current end.
    Depending on mode, we take the first or last three points (if existing) of the fault line and fit an interpolating
    polynomial curve. We then create new points on this line such that their distance is approximately the distance of
    the ones already existing. These points and the corresponding auxiliary points are used to obtain new points on the
    fault line.
    We repeat this process until we leave the domain, or we intrude in a subdomain neither belonging to i_class nor
    j_class. In this case, we take the first of these points and discard all others of this kind. We adjust its
    parameter by bisection to obtain the maximal parameter where no third class is involved. This point marks
    approximately the end of the fault line.
    For any fault line, we store at which side of the domain the fault line leaves the domain if so in the array
    left_domain. This information is necessary for constructing the approximating polygons of any subdomain in function
    ReconstructSubdomains.

    The information of left_domain is coded as follows:
    1: left boundary,
    2: right boundary,
    3: bottom boundary,
    4: top boundary,
    0: fault line end inside the domain

    This routine is applied for expanding fault surfaces in 3D as well. Here, we take some points from the fault
    surface and treat them as points defining a curve to continue. Again, we need points near the curve on both sides
    of the fault line. Using just a vector normal to the curve as in 2D does not work. Therefore, we need additional
    information which is provided by normal_vec_plane.

    Args:
        left_domain (npt.ArrayLike): See note.
        idx_points (npt.ArrayLike): Array of point indices according to their position on the fault line.
        i_class (int): We deal with the i_segs-th part of the fault line between classes i_class und j_class.
        j_class (int)
        i_seg (int)
        class_vals (npt.ArrayLike):
            Array containing the class values. These are not necessarily the class indices. Imagine that
            f(\Omega) = {1,2,5}. Then, class_vals = [1,2,5], whereas the class indices range from 0 to 2.
            Shape: (nclasses,)
        points_i_class (npt.ArrayLike): The fault points near the current fault line in i_class.
        points_j_class (npt.ArrayLike): The fault points near the current fault line in j_class.
        normal_vec_plane (npt.ArrayLike):
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.
        mode (int): Mode of operation (0 or 1, see description above).

    Returns:
        Tuple[ npt.ArrayLike, npt.ArrayLike, int, npt.ArrayLike, npt.ArrayLike, bool]:
            idx_points_enlarged: array of point indices according to their position on the fault line, but enlarged by
            the indices of the points added
            resort: If the fault line is closed, the start and the end of the fault line in fact overlap. This requires
            a complete resorting of the points. We do not resort inside this routine but indicate the requirement by
            setting resort = true.
            points_added: If really points have been added, this variable is True.

    Raises:
        Exception: If there is an invalid mode in expand_2d.
    """
    num_points_on_curve = 4

    points_added = False

    # maximal number of expansion cycles (to prevent endless loops)
    max_cycles = 500

    finished = False
    bdomain_left = False

    num_points = points_i_class.shape[0]

    if num_points < 2:
        warn("Not enough triplets provided for expanding the fault line.")
        resort = False
        points_added = False
        idx_points_enlarged = idx_points
        n_points = num_points
        return idx_points_enlarged, left_domain, n_points, points_i_class, points_j_class, resort, points_added

    # dist_vec contains actual distance between consecutive points.
    dist_vec_aux = (points_i_class[idx_points[0:num_points - 1], 0] -
                    points_i_class[idx_points[1:num_points], 0]) ** 2 + \
                   (points_i_class[idx_points[0:num_points - 1], 1] - points_i_class[idx_points[1:num_points], 1]) ** 2

    dist_vec_aux = np.sqrt(dist_vec_aux)
    dist_vec = np.zeros(num_points, dtype=float)
    dist_vec[idx_points] = np.concatenate(([dist_vec_aux[0]],
                                           0.5 * (dist_vec_aux[0:num_points - 2] + dist_vec_aux[1:num_points - 1]),
                                           [dist_vec_aux[-1]]))

    # requirement for complete resorting (not necessary by default)
    resort = False

    # for computing valid point pairs as starting values for bisection
    alpha = fault_approx_pars.alpha

    # Points closer than eps are regarded as identical.
    eps = fault_approx_pars.eps

    n_points, ndim = points_i_class.shape

    # auxiliary vector
    aux_vec_loc = np.zeros(3)

    # index array for the extended set points on the surface (at the beginning, this is the original set of points)
    idx_points_enlarged = idx_points

    # number of cycles in expansion
    ncycles = 0

    while not finished and ncycles < max_cycles:

        ncycles = ncycles + 1

        # extend at the beginning
        if mode == 0:
            idx_set = np.arange(min(n_points, num_points_on_curve) - 1, -1, -1)

        # extend at the end
        elif mode == 1:
            idx_set = np.arange(max(0, n_points - num_points_on_curve), n_points)
        else:
            raise Exception("Invalid mode of operation in expand_2d.")

        num_points_to_interpolate = idx_set.shape[0]

        points_on_fault = 0.5 * (points_i_class[idx_points_enlarged[idx_set], :] +
                                 points_j_class[idx_points_enlarged[idx_set], :])

        # extrapolate current points
        new_point_on_fault, avg_dist, extra_par, data_pars, q_mat, x_mean, coeffs = extrapolate_on_fault_line(
            points_on_fault, 1, fault_approx_pars)

        finished_surf_point = False

        i_try = 0

        while i_try <= 3 and not finished_surf_point:

            i_try += 1

            extra_par = 1.0/2**(i_try-1)*(extra_par - data_pars[num_points_to_interpolate - 1]) + \
                data_pars[num_points_to_interpolate - 1]

            new_point_on_fault = __eval_poly(extra_par, coeffs, q_mat, x_mean)

            # distance of the new point relative to the average distance of the existing points
            rel_dist = norm(new_point_on_fault - points_on_fault[num_points_to_interpolate - 1, :], 2, axis=1) / \
                avg_dist

            # logical array indicating which points on the extrapolated fault line are outside the domain
            outside_domain = np.any(np.logical_or(new_point_on_fault < problem_descr.x_min,
                                                  new_point_on_fault > problem_descr.x_max), axis=1)

            # compute normal vector
            aux_vec_loc[0:ndim] = new_point_on_fault - points_on_fault[num_points_to_interpolate - 1, :]

            normal_vec_loc = np.cross(aux_vec_loc, normal_vec_plane)
            normal_vec_loc = normal_vec_loc[0:ndim] / norm(normal_vec_loc, axis=0)

            # this is purely heuristic
            step_size = np.maximum(0.95 * fault_approx_pars.abstol_bisection,
                                   0.5 * np.minimum(alpha * avg_dist,
                                                    3.0 * fault_approx_pars.abstol_bisection * (rel_dist ** 2 + 1)))

            step_size = step_size[:, np.newaxis]

            point_left = new_point_on_fault + step_size * normal_vec_loc
            point_left = np.minimum(np.maximum(problem_descr.x_min + eps, point_left), problem_descr.x_max - eps)

            point_right = new_point_on_fault - step_size * normal_vec_loc
            point_right = np.minimum(np.maximum(problem_descr.x_min + eps, point_right), problem_descr.x_max - eps)

            if not outside_domain:
                aux_arr = compute_classification(np.concatenate((point_left, point_right), axis=0), problem_descr)

                class_point_left = aux_arr[0]
                class_point_right = aux_arr[1]

                outside_sub = (class_point_left != class_vals[i_class] and class_point_left != class_vals[j_class]) or \
                              (class_point_right != class_vals[i_class] and class_point_right != class_vals[j_class])

            else:
                class_point_left = -1
                class_point_right = -1

                outside_sub = False

            # If there are some points on the curve which are neither in i_class nor in j_class, we have reached the
            # boundary of the domain or the subdomain and have therefore reached the last cycle of the expansion
            # loop. As points outside the valid domain have class -1, this includes the case the domain has been left.
            # Note that if the "central" points from which the left and the right points was generated is outside the
            # domain, we automatically assume that this holds for both the left and the right points. Concentrating here
            # on just point_left therefore does not harm.
            if outside_domain:
                finished = True
                bdomain_left = True

                # Approximate the intersection of the extrapolated line and the domain boundary. We know that
                # data_pars[num_points_to_interpolate-1] belongs to a point inside the domain. Therefore, it is a lower
                # bound for the parameter of the intersection. On the other hand, the point corresponding to extra_par
                # does not belong to the domain. Therefore, this is an upper bound for the parameter.
                x_new, iedge = extra_new_point_on_domain_bdry(data_pars[num_points_to_interpolate-1], extra_par,
                                                              points_on_fault[-1, :], new_point_on_fault[0], coeffs,
                                                              q_mat, x_mean, problem_descr, fault_approx_pars)

                # update_vec is not necessarily the normal vector of the extrapolated fault line in x_new, so we call it
                # update_vec.
                update_vec = np.cross(normal_vec_plane, aux_vec_loc)

                # from x_new, we need to create a pair of points as starting point for bisection. We need to ensure that
                # the line segment between these two points is parallel to the appropriate domain boundary. We do this
                # by modifying update_vec.
                if iedge == 1:
                    # x2 is minimal (bottom boundary/front surface of cube)
                    update_vec[1] = 0.0
                elif iedge == 2:
                    # x1 is maximal (right boundary/right surface of cube)
                    update_vec[0] = 0.0
                elif iedge == 3:
                    # x2 is maximal (top boundary/back surface of cube)
                    update_vec[1] = 0.0
                elif iedge == 4:
                    # x1 is minimal (left boundary/left surface of cube)
                    update_vec[0] = 0.0
                elif iedge == 5:
                    # x3 is minimal (bottom surface of cube)
                    update_vec[2] = 0.0
                elif iedge == 6:
                    # x3 is maximal (top surface of cube)
                    update_vec[2] = 0.0

                rel_dist = norm(x_new - points_on_fault[num_points_to_interpolate - 1, :]) / avg_dist
                update_vec = update_vec[0:ndim]

                # heuristic control of step length
                update_vec = update_vec / norm(update_vec) * np.max((0.95 * fault_approx_pars.abstol_bisection,
                                                                     3.0 * fault_approx_pars.abstol_bisection *
                                                                     (rel_dist ** 2 + 1.0)))

                x_new_left = x_new + update_vec
                x_new_right = x_new - update_vec

                # Ensure that the new points near the boundary are really inside the domain.
                x_new_left = np.maximum(np.minimum(x_new_left, problem_descr.x_max - eps), problem_descr.x_min + eps)
                x_new_right = np.maximum(np.minimum(x_new_right, problem_descr.x_max - eps), problem_descr.x_min + eps)

                [class_left, class_right] = compute_classification(np.vstack((x_new_left, x_new_right)), problem_descr)
                left_domain[i_class, j_class, i_seg] = iedge

                # Find starting values for bisection.
                point_pairs, idx_ok, class_points_ok = start_pairs(x_new_left.reshape((-1, ndim)),
                                                                   x_new_right.reshape((-1, ndim)),
                                                                   np.array([class_left]), np.array([class_right]),
                                                                   class_vals[i_class], class_vals[j_class],
                                                                   problem_descr)
                class_points_ok = class_points_ok[0]

                if idx_ok:
                    point_left, point_right, finished_surf_point = single_triplet_by_bisection(point_pairs[0, 0:ndim],
                                                                                               point_pairs[0,
                                                                                               ndim:2 * ndim],
                                                                                               class_points_ok[0],
                                                                                               class_points_ok[1],
                                                                                               problem_descr,
                                                                                               fault_approx_pars)

                left_domain[i_class, j_class, i_seg] = iedge

            # There is no indication that the fault line has left the domain. Therefore, we can continue to expand
            # unless it ends inside the domain.
            elif outside_sub:
                finished = True
                bdomain_left = False

                # Did we expand the fault line beyond its end? If so, either point_left or point_right belongs neither
                # to i_class nor to j_class.
                # In this case, take the last known point in the subdomain and the first known outside, discard all
                # others and find the end of the extrapolated fault line by bisection.
                # However, this is a heuristic approach, as it may happen that e.g. due to insufficient distance to the
                # extrapolation curve, both points belong to the same class.
                #                 %                      II
                #                 %        -------------------------->     x point_left
                #                 % ______x________x________x_____         x extrapolated point
                #                 %                               \--__    x point_right
                #                 %              I                 \   ---__
                #                 %                                 \       ---__
                #                 %                                  \    III    ---__
                #
                # Then, we do not detect that we extrapolated the fault line beyond its end. We will fail to find a
                # valid starting pair such that prolongation stops and no point is added, even if the end of the fault
                # line is not yet reached.

                # A fault line ends inside the domain by intersecting a subdomain with a third class.
                # We aim at finding the intersection of the extrapolated fault line with that third subdomain using
                # bisection.

                # A lower bound t_min for the parameter of the intersection is the parameter of the last known
                # point inside the domain.
                t_min = data_pars[num_points_to_interpolate - 1]

                # Correspondingly, the parameter of the first point on the extrapolated fault line which is known
                # to be outside the subdomain is an upper bound for the parameter of the intersection.
                t_max = extra_par[0]

                # This point is known to be inside the subdomain.
                x_min = __eval_poly(np.array([t_min]), coeffs, q_mat, x_mean)[0]
                x_max = __eval_poly(np.array([t_max]), coeffs, q_mat, x_mean)[0]
                dist = norm(x_max - x_min)

                x_min_aux_all = np.empty((0, 2*ndim))
                x_min_all = np.empty((0, ndim))
                class_min_all = np.empty((0, ndim), dtype=int)

                iiter = 1
                while iiter < 10 and dist > fault_approx_pars.abstol_bisection:

                    t_new = 0.5 * (t_min + t_max)

                    x_new = __eval_poly(np.array([t_new]), coeffs, q_mat, x_mean)[0]

                    aux_vec_loc[0:ndim] = x_new - x_min
                    normal_vec_loc = np.cross(aux_vec_loc, normal_vec_plane)
                    normal_vec_loc = normal_vec_loc[0:ndim]/np.linalg.norm(normal_vec_loc, 2)
                    rel_dist = norm(points_on_fault[num_points_to_interpolate-1, :] - x_new, 2)/avg_dist

                    # this is purely heuristic
                    step_size = max(0.95 * fault_approx_pars.abstol_bisection, 0.5 *
                                    min(alpha * avg_dist,
                                        3.0 * fault_approx_pars.abstol_bisection * (rel_dist ** 2 + 1)))

                    x_new_left = x_new + step_size*normal_vec_loc
                    x_new_right = x_new - step_size*normal_vec_loc

                    x_new_left = np.maximum(np.minimum(x_new_left, problem_descr.x_max - eps),
                                            problem_descr.x_min + eps)
                    x_new_right = np.maximum(np.minimum(x_new_right, problem_descr.x_max - eps),
                                             problem_descr.x_min + eps)

                    class_new = compute_classification(np.vstack((x_new_left, x_new_right)), problem_descr)

                    if np.any(np.logical_and(class_new != class_vals[i_class], class_new != class_vals[j_class])):
                        t_max = t_new
                        x_max = x_new
                    else:
                        t_min = t_new
                        x_min = x_new
                        x_min_aux_all = np.vstack((x_min_aux_all, np.hstack((x_new_left, x_new_right))))
                        x_min_all = np.vstack((x_min_all, x_min))
                        class_min_all = np.vstack((class_min_all, class_new))

                    dist = norm(x_max - x_min)

                    iiter += 1

                iiter = 0
                finished_surf_point = False

                while iiter < x_min_all.shape[0] and not finished_surf_point:

                    iiter += 1
                    x_new = x_min_all[-iiter]

                    # We need a point which is inside the current subdomain, and we know that x_min is inside the
                    # domain.
                    # However, if t_min is (almost) at its initial value, x_min is in fact the last known point on the
                    # fault. It does not make sense to add this point again.
                    if norm(x_new - points_on_fault[num_points_to_interpolate-1]) > \
                            fault_approx_pars.abstol_bisection:

                        x_new_left = x_min_aux_all[-1, 0:ndim]
                        x_new_right = x_min_aux_all[-1, ndim: 2*ndim]

                        class_new[0] = compute_classification(x_new, problem_descr)[0]

                        x_new_suitable = class_new[0] == class_vals[i_class] or class_new[0] == class_vals[j_class]

                        if x_new_suitable:
                            # x_new belongs to a valid point pair
                            if class_new[0] == class_vals[i_class] and class_min_all[-iiter, 0] == class_vals[j_class]:
                                x_new_left = x_new
                                class_point_left = class_new[0]
                                x_new_right = x_min_aux_all[-iiter, 0:ndim]
                                class_point_right = class_min_all[-iiter, 0]
                            elif class_new[0] == class_vals[j_class] and class_min_all[-iiter, 0] == \
                                    class_vals[i_class]:
                                x_new_left = x_new
                                class_point_left = class_new[0]
                                x_new_right = x_min_aux_all[-iiter, 0:ndim]
                                class_point_right = class_min_all[-iiter, 0]
                            elif class_new[0] == class_vals[j_class] and class_min_all[-iiter, 1] == \
                                    class_vals[i_class]:
                                x_new_left = x_new
                                class_point_left = class_new[0]
                                x_new_right = x_min_aux_all[-iiter, ndim:2*ndim]
                                class_point_right = class_min_all[-iiter, 1]
                            elif class_new[0] == class_vals[i_class] and class_min_all[-iiter, 1] == \
                                    class_vals[j_class]:
                                x_new_left = x_new
                                class_point_left = class_new[0]
                                x_new_right = x_min_aux_all[-iiter, ndim:2*ndim]
                                class_point_right = class_min_all[-iiter, 1]

                            point_left, point_right, finished_surf_point = single_triplet_by_bisection(
                                x_new_left, x_new_right, class_point_left, class_point_right, problem_descr,
                                fault_approx_pars)
                            class_points_ok = np.array([class_point_left, class_point_right], dtype=int)

                left_domain[i_class, j_class, i_seg] = 0

            # expansion continues
            else:

                # Find starting values for bisection.
                point_pairs, idx_ok, class_points_ok = start_pairs(point_left,
                                                                   point_right,
                                                                   np.array([class_point_left]),
                                                                   np.array([class_point_right]),
                                                                   class_vals[i_class],
                                                                   class_vals[j_class],
                                                                   problem_descr)

                class_points_ok = class_points_ok[0]

                # If no point has been found at all, stop. If we have no indication that the fault line left the domain,
                # we must assume that it ended inside.
                if idx_ok:
                    point_left, point_right, finished_surf_point = single_triplet_by_bisection(point_pairs[0, 0:ndim],
                                                                                               point_pairs[0,
                                                                                               ndim:2 * ndim],
                                                                                               class_points_ok[0],
                                                                                               class_points_ok[1],
                                                                                               problem_descr,
                                                                                               fault_approx_pars)
                else:
                    left_domain[i_class, j_class, i_seg] = 0
                    finished = True

            # Add the new point if appropriate.
            if finished_surf_point:

                # Compute the distance to the currently last point on the fault line.
                # If this distance is tiny, replace this point by the new one.
                last_current_point = points_on_fault[-1, :]
                if class_points_ok[0] == class_vals[i_class]:
                    dist_to_new_point = norm(last_current_point - point_left)
                else:
                    dist_to_new_point = norm(last_current_point - point_right)

                # The new point is sufficiently far away from the last known one: add it.
                if dist_to_new_point > avg_dist * fault_approx_pars.min_dist_factor:
                    i_idx_add = n_points
                    n_points += 1

                    if mode == 0:
                        idx_points_enlarged = np.hstack((n_points-1, idx_points_enlarged))
                    elif mode == 1:
                        idx_points_enlarged = np.hstack((idx_points_enlarged, n_points-1))

                # The new point is almost the last known one: replace the last known one by the new one and stop
                # expanding.
                else:
                    i_idx_add = idx_points_enlarged[idx_set[-1]]
                    finished = True

                if class_points_ok[0] == class_vals[i_class]:
                    if i_idx_add == points_i_class.shape[0]:
                        points_i_class = np.concatenate((points_i_class, point_left.reshape(1, -1)), axis=0)
                        points_j_class = np.concatenate((points_j_class, point_right.reshape(1, -1)), axis=0)
                    else:
                        points_i_class[i_idx_add] = point_left
                        points_j_class[i_idx_add] = point_right
                else:
                    if i_idx_add == points_i_class.shape[0]:
                        points_i_class = np.concatenate((points_i_class, point_right.reshape(1, -1)), axis=0)
                        points_j_class = np.concatenate((points_j_class, point_left.reshape(1, -1)), axis=0)
                    else:
                        points_i_class[i_idx_add] = point_right
                        points_j_class[i_idx_add] = point_left

                # If the fault line appears to start or end inside the domain, it may be closed. We test this here.
                if left_domain[i_class, j_class, i_seg] <= 0:
                    closed_comp = component_closed(points_i_class, idx_points_enlarged, dist_vec, n_points, mode)

                    if closed_comp:
                        # If the fault component is closed, it does not start or end at the domain boundary. We finish
                        # expanding this fault line and state that it starts and ends inside the domain.
                        left_domain[i_class, j_class, i_seg] = 0

                        # The new point is in fact somewhere between existing points. This requires resorting.
                        resort = True
                        finished = True

                points_added = True
                if i_idx_add < dist_vec.shape[0]:
                    dist_vec[i_idx_add] = dist_to_new_point
                else:
                    dist_vec = np.hstack((dist_vec, dist_to_new_point))

            # No suitable point for adding found: try again. We assume in this case that the fault line starts or ends
            # inside the domain unless found otherwise.
            else:
                if left_domain[i_class, j_class, i_seg] < 0 and not bdomain_left:
                    left_domain[i_class, j_class, i_seg] = 0

    return idx_points_enlarged, left_domain, n_points, points_i_class, points_j_class, resort, points_added


def extrapolate_on_fault_line(points_on_curve: npt.ArrayLike, num_points_new: int,
                              fault_approx_pars: FaultApproxParameters) -> Tuple[npt.ArrayLike, float,
                                                                                 npt.ArrayLike, npt.ArrayLike,
                                                                                 npt.ArrayLike, npt.ArrayLike,
                                                                                 npt.ArrayLike]:
    """This function computes num_points_new points on an extrapolated fault line. It is constructed using the points
    given in points_on_curve and fitting a polynomial in local coordinates. As the points are not known exactly, we do
    not interpolate exactly, but penalise the second derivative in a least-squares approximation. Following Morozov, we
    choose the regularisation such that the maximal residual is approx. fault_approx_pars.abstol_bisection.
    These points are constructed such that their distance is the minimum of GROW_FACTOR*the average distance of the
    existing points, fault_approx_pars.max_dist_for_surface_points and the requirement that the deviation from a
    straight line is at most (approximately) fault_approx_pars.err_max.

    Args:
        points_on_curve (npt.ArrayLike):
            Points on the curve representing a fault line (must be ordered). Shape: (npoints, ndim)
        num_points_new (int):
            Number of new points on the curve aka fault line constructed by extrapolation.
        fault_approx_pars (FaultApproxParameters):
            Class containing all relevant parameters for our method of detecting fault lines.

    Returns:
        new_points_on_curve (npt.ArrayLike):
            cartesian coordinates of the new points. Shape: (number of new points, ndim)
        avg_dist (float):
            average distance of the points on the fault line used for extrapolation
        extra_pars (npt.ArrayLike):
            Parameters aka local coordinates of the new points on the polynomial curve.
        data_pars (npt.ArrayLike):
            Parameters aka local coordinates of the points in points_on_curve.
        q_mat (npt.ArrayLike):
            orthogonal matrix; the local coordinates are computed by xloc = q_mat@x + x_mean
        x_mean (npt.ArrayLike):
            origin of the local coordinate system; the local coordinates are computed by xloc = q_mat@x + x_mean
        coeffs (npt.ArrayLike):
            coefficients of the extrapolating polynomial
    """

    num_points_on_curve, ndim = points_on_curve.shape

    # we need at least two points for extrapolation
    if num_points_on_curve < 2:
        warn('Too few points to extrapolate')
        new_points_on_curve = np.array([0, 2])
        avg_dist = 0.0
        data_pars = np.array([])
        extra_pars = np.array([])
        q_mat = np.zeros((ndim, ndim))
        x_mean = np.zeros(0)
        coeffs = np.array([])
        return new_points_on_curve, avg_dist, extra_pars, data_pars, q_mat, x_mean, coeffs

    # average distance of the points on the fault line used for extrapolation
    avg_dist = norm(points_on_curve[num_points_on_curve - 1] - points_on_curve[0], 2) / (num_points_on_curve - 1)

    x_mean = (1.0 / num_points_on_curve) * np.ones(num_points_on_curve).dot(points_on_curve)

    points_shifted = points_on_curve - x_mean

    q_mat = np.linalg.svd(points_shifted)[2].T
    points_rot = points_shifted @ q_mat

    # We limit the step size for extrapolation based upon curvature. It does not make sense to extrapolate very far
    # with a lot of numerical amount and to adaptively refine in between the current and the new points anyway
    # afterwards. To do so, we compute the step size which would lead to the maximal admissible deviation
    # fault_approx_pars.err_max from a straight line segment. This is a natural upper bound for the step size in
    # extrapolation.
    #
    # The maximal deviation d of a curve with curvature c from an interpolating line segment with length l is
    #   d = 1/4 c l^2 + 1/16 c^3 l^4 + h.o.t
    # Rearranging leads to l = 2/c^2(sqrt{1+4cd} - 1), which is numerically unstable if cd is small. We set (cl)^2 = v
    # and search for the bigger root of the quadratic equation
    #   v^2 + 4 v - 16cd = 0, aka v^2 + 2pv - q = 0.
    # According to Vieta, q = vmin*vmax with the two roots vmin and vmax. Therefore,
    #   vmin = - (2 + sqrt{4+ 16cd}) = -2 (1 + sqrt{1 + 4cd}),
    # which is stable to evaluate and vmax = -16cd/vmin, and ultimately lmax = sqrt{vmax}/c. Inserting vmax yields
    #   lmax = sqrt{-16cd/vmin}/c = 4 sqrt{d/(-c vmin)}

    # Estimating curvature is possible only for more than two points on the fault line and two dimensions.
    if num_points_on_curve > 2 and ndim == 2:
        curv = estimate_curvature_2d(points_shifted, num_points_on_curve - 1,
                                     fault_approx_pars.abstol_bisection)[0] + \
               estimate_curvature_2d(points_shifted, num_points_on_curve - 2, fault_approx_pars.abstol_bisection)[0]

        if curv > 1e-10:
            lmax = 2.0 * (1 + np.sqrt(1.0 + 4 * curv * fault_approx_pars.err_max))
            lmax = 4.0 * np.sqrt(fault_approx_pars.err_max / (curv * lmax))
        else:
            lmax = 1e10

    # Limiting step size based upon curvature does not work, if only two points on the fault line are known.
    else:
        lmax = 1e10

    step_size = min([fault_approx_pars.max_dist_for_surface_points, GROWTH_FACTOR * avg_dist, lmax])

    if np.all(points_rot[0:num_points_on_curve - 1, 0] < points_rot[1:num_points_on_curve, 0]):
        extra_pars = points_rot[num_points_on_curve - 1, 0] + step_size * np.arange(1, num_points_new + 1)
    elif np.all(points_rot[0:num_points_on_curve - 1, 0] > points_rot[1:num_points_on_curve, 0]):
        extra_pars = points_rot[num_points_on_curve - 1, 0] - step_size * np.arange(1, num_points_new + 1)

    # It is not possible to extrapolate the fault line by any function respecting the order of the points. This is
    # usually the case if the points are on a curve with very strong curvature. Therefore, we just take the two last
    # points and perform linear extrapolation.
    else:
        num_points_on_curve = 2
        if points_rot[num_points_on_curve - 2, 0] < points_rot[num_points_on_curve - 1, 0]:
            extra_pars = points_rot[num_points_on_curve - 1, 0] + step_size * np.arange(1, num_points_new + 1)
        else:
            extra_pars = points_rot[num_points_on_curve - 1, 0] - step_size * np.arange(1, num_points_new + 1)

    int_mat = np.zeros((num_points_on_curve, num_points_on_curve))
    penalty_mat = np.zeros((num_points_on_curve, num_points_on_curve))
    for i in range(num_points_on_curve):
        int_mat[:, i] = points_rot[0: num_points_on_curve, 0] ** i
        penalty_mat[:, i] = i * (i - 1) * points_rot[0: num_points_on_curve, 0] ** (np.max((0, i - 2)))

    # Regularization by penalizing the second derivative is pointless if only two points exist so far: Then, we perform
    # linear extrapolation, and its second derivative is zero anyway.
    if num_points_on_curve > 2:
        penalty_mat = penalty_mat.T @ penalty_mat
        coeffs = reg_ls(int_mat, penalty_mat, points_rot[:, 1], fault_approx_pars.abstol_bisection, 2)
    else:
        coeffs = np.linalg.solve(int_mat, points_rot[0:num_points_on_curve, 1])

    new_points_on_curve = __eval_poly(extra_pars, coeffs, q_mat, x_mean)
    data_pars = points_rot[:, 0]

    # It may happen that the true distance of the extrapolated points to the existing ones is too large, as the arc
    # length on the curve is much greater than the distance of the parameters. In this case, we adjust the parameters
    # accordingly. For the sake of simplicity, we consider the true distance of the last existing to the first new
    # point only.
    true_dist = np.linalg.norm(new_points_on_curve - points_on_curve[num_points_on_curve - 1, :], axis=1)

    if true_dist[0] > 1.1 * fault_approx_pars.max_dist_for_surface_points:
        tmax = extra_pars[0]
        tmin = data_pars[num_points_on_curve - 1]

        iiter = 1

        # The maximal number of iterations should never be reached in practical examples. If so, the adjusted values are
        # at least better than the original ones.
        tnew = 0.0
        while iiter < 20:
            tnew = 0.5 * (tmin + tmax)
            xnew = __eval_poly(np.array([tnew]), coeffs, q_mat, x_mean)

            dist = norm(xnew - points_on_curve[num_points_on_curve - 1, :])
            res = dist - fault_approx_pars.max_dist_for_surface_points

            # if the distance to the existing points is max_dist_for_surface_points up to 10%
            if np.abs(res) < 0.1 * fault_approx_pars.max_dist_for_surface_points:
                break
            else:
                if dist > fault_approx_pars.max_dist_for_surface_points:
                    tmax = tnew
                else:
                    tmin = tnew

            iiter = iiter + 1

        extra_pars = data_pars[num_points_on_curve - 1] + \
            np.arange(1, num_points_new + 1) * (tnew - data_pars[num_points_on_curve - 1])
        new_points_on_curve = __eval_poly(extra_pars, coeffs, q_mat, x_mean)

    return new_points_on_curve, avg_dist, extra_pars, data_pars, q_mat, x_mean, coeffs


def extra_new_point_on_domain_bdry(t_min: float, t_max: float, x_min: npt.ArrayLike, x_max: npt.ArrayLike,
                                   coeffs: npt.ArrayLike, q_mat: npt.ArrayLike, x_mean: npt.ArrayLike,
                                   problem_descr: ProblemDescr, fault_approx_pars: FaultApproxParameters) ->\
        Tuple[npt.ArrayLike, int]:
    """
    This function aims at finding the intersection of an extrapolated fault line with the domain boundary using
    bisection. We exploit that the domain is either a rectangle or a cuboid.
    We assume that the fault line has been extrapolated with a polynomial in local coordinates. We approximate the
    parameter value t of the intersection of that polynomial with the domain boundary by bisection.
    For doing so, we need a lower and upper bound t_min and t_max for that parameter t along with the corresponding
    points x_min and x_max.

    Args:
        t_min (float):
            lower bound for the parameter value of the intersection
        t_max (float):
            upper bound for the parameter value of the intersection
        x_min (npt.ArrayLike):
            point on the extrapolating polynomial corresponding to t_min given in global coordinates
        x_max (npt.ArrayLike):
            point on the extrapolating polynomial corresponding to t_max given in global coordinates
        coeffs (npt.ArrayLike):
            coefficients of the extrapolating polynomial
        q_mat (npt.ArrayLike):
            orthogonal matrix describing the local coordinate system
        x_mean (npt.ArrayLike):
            origin of the local coordinate system
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.

    Returns:
        x_new (npt.ArrayLike):
            approximate intersection of fault line and domain boundary
        iedge (int):
            index of the domain boundary part, at which the fault line leaves the domain
    """

    # indices of the domain edges (in 2D):
    #        y /\
    #          │     3
    #          ├───────────┐
    #          │           │
    #        4 │           │ 2
    #          │           │
    #          └───────────│─────> x
    #                1
    #
    # indices of the domain surfaces (in 3D)
    #                   6          3              z
    #                   ┊         ╱               △      y
    #               ┌───┊───────╱───┐             │     ╱
    #             ╱ │   ┊     ╱   ╱ │             │   ╱
    #           ╱   │   v   ↙   ╱   │             │ ╱
    #         ╱     │         ╱     │             └─────────> x
    #        ┌───────────────┐      │
    #  4┈┈┈┈┈│┈>    │        │    <┈┈┈┈┈┈┈2
    #        │      └--------│------┘
    #        │     ╱         │     ╱
    #        │   ╱   ↗  △    │   ╱
    #        │ ╱   ╱    ┊    │ ╱
    #        └───╱──────┊────┘
    #          ╱        ┊
    #         1         5

    i_aux_min = np.array([4, 1, 5], dtype=int)
    i_aux_max = np.array([2, 3, 6], dtype=int)

    # Points closer than eps are considered identical.
    eps = fault_approx_pars.eps

    ndim = x_min.shape[0]

    # euclidean distance of these points
    dist = norm(x_max - x_min)
    iiter = 1

    while iiter < 10:

        # The factor 0.5 in the following condition is just a safety factor (iterating one time more does not increase
        # the number of function evaluations).
        if dist < 0.5 * fault_approx_pars.abstol_bisection:
            break
        else:
            t_new = 0.5 * (t_min + t_max)
            x_new = __eval_poly(np.array([t_new]), coeffs, q_mat, x_mean)[0]

            if np.any(np.logical_or(x_new < problem_descr.x_min, x_new > problem_descr.x_max)):
                t_max = t_new
                x_max = x_new
            else:
                t_min = t_new
                x_min = x_new

            # The distance will approximately halve per iterations step, but only approximately (the
            # extrapolation is not necessarily a straight line).
            dist = norm(x_max - x_min)

        iiter = iiter + 1

    # index of the edge or facet at which the fault line leaves the domain
    iedge = 0

    for i in range(0, ndim):
        if x_max[i] > problem_descr.x_max[i]:
            iedge = i_aux_max[i]
        elif x_max[i] < problem_descr.x_min[i]:
            iedge = i_aux_min[i]

    # The approximation of the intersection is not necessarily inside the domain, so we enforce this here.
    x_new = np.maximum(np.minimum(x_new, problem_descr.x_max - eps), problem_descr.x_min + eps)

    # Moreover, ensure that the new point is up to eps on the domain boundary (it is somewhere inside near a boundary
    # up to abstol_bisection, but we can do better).
    if iedge == 1:
        # x2 is minimal (bottom boundary/front surface of cube)
        x_new[1] = problem_descr.x_min[1] + eps
    elif iedge == 2:
        # x1 is maximal (right boundary/right surface of cube)
        x_new[0] = problem_descr.x_max[0] - eps
    elif iedge == 3:
        # x2 is maximal (top boundary/back surface of cube)
        x_new[1] = problem_descr.x_max[1] - eps
    elif iedge == 4:
        # x1 is minimal (left boundary/left surface of cube)
        x_new[0] = problem_descr.x_min[0] + eps
    elif iedge == 5:
        # x3 is minimal (bottom surface of cube)
        x_new[2] = problem_descr.x_min[1] + eps
    elif iedge == 6:
        # x3 is maximal (top surface of cube)
        x_new[2] = problem_descr.x_max[0] - eps

    return x_new, iedge


def test2d():
    """Testfunction 2d.

    """
    from tests.test_funcs.TestFunc2D import func_fd_2d_cl3_c0_01
    problem_descr = ProblemDescr()
    problem_descr.test_func = func_fd_2d_cl3_c0_01
    problem_descr.x_min = np.array([0, 0])
    problem_descr.x_max = np.array([1, 1])
    fault_approx_pars = FaultApproxParameters()
    fault_approx_pars.max_dist_for_surface_points = 0.05
    fault_approx_pars.abstol_bisection = 0.001
    fault_approx_pars.num_points_local = 10

    points_on_curve = np.array([[0, 0], [1, 1], [2, 4]])
    num_points_new = 2
    new_points_on_curve, avg_dist, extra_pars, data_pars, q_mat, x_mean, coeffs = \
        extrapolate_on_fault_line(points_on_curve, num_points_new, fault_approx_pars)

    print(new_points_on_curve)


def test3d():
    """Testfunction 3d.
    """
    from tests.test_funcs.TestFunc2D import func_fd_2d_cl2_c1_03
    left_domain = np.zeros((2, 2), dtype=int)
    idx_points_surf_ordered = np.array([1, 2]) - 1
    i_class = 1
    j_class = 2
    i_seg = 1
    class_vals = np.array([1, 2])
    points_i_class = np.array([[0.028, 0.295, 0.336], [0.001, 0.295, 0.337]])
    points_j_class = np.array([[0.028, 0.295, 0.335], [0, 0.295, 0.336]])
    normal_vec_plane = np.array([-1, 0, 0])
    problem_descr = ProblemDescr()
    problem_descr.test_func = func_fd_2d_cl2_c1_03
    problem_descr.x_min = np.array([0, 0, 0])
    problem_descr.x_max = np.array([1, 1, 1])
    fault_approx_pars = FaultApproxParameters()
    fault_approx_pars.max_dist_for_surface_points = 0.05
    fault_approx_pars.num_points_local = 5
    fault_approx_pars.abstol_bisection = 0.001
    mode = 1
    idx_points_surf_ordered_enlarged, left_domain, n_points, points_i_class, points_j_class, resort, points_added =\
        expand_2d(left_domain, idx_points_surf_ordered, i_class, j_class, i_seg, class_vals,
                  points_i_class, points_j_class, normal_vec_plane, problem_descr, fault_approx_pars,
                  mode)
    print(idx_points_surf_ordered_enlarged, left_domain, n_points, points_i_class, points_j_class, resort, points_added,
          sep="\n\n")


if __name__ == '__main__':
    test2d()
