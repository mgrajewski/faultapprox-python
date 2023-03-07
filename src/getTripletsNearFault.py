"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
from warnings import warn
import logging

import numpy as np
import numpy.typing as npt
from typing import Tuple

from scipy.linalg import norm, svd
import tools.statistics as stats
import tools.ncalls as ncalls
import copy

from adapt import adapt_2d
from checkPoints import self_intersection, is_on_poly_line, poly_lines_intersect
from computeClassification import compute_classification
from newtriplets import triplets_by_bisection, single_triplet_by_bisection
from fill import fill_2d, divide_into_components
from geometry import compute_dist_mat
from expand import expand_2d
from entities.Entities import FaultApproxParameters, ProblemDescr
from sort import sort_on_fault


def get_triplets_near_fault(barycentres: npt.ArrayLike, point_set: npt.ArrayLike, class_of_barys: npt.ArrayLike,
                            class_of_points: npt.ArrayLike, problem_descr: ProblemDescr,
                            fault_approx_pars: FaultApproxParameters) -> \
        Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, bool, npt.ArrayLike]:
    """
    This function constructs points near the fault line(s)/surfaces.
    For details, we refer to the paper. We sketch our algorithm in what follows:
    1) building block initialize: We subdivide the set of barycentres according to class and search for any barycentre
       in class i_class its nearest neighbour in class j_class. The line from i_class to j_class is supposed to
       intersect the fault line/surface separating subset i_class from subset j_class. We find two points, one in class
       i_class and the other one in class j_class on this line close to the fault line/surface by bisection.
     2) building block fill:
        2D: We order these points according to their position on the fault line. This enables us to compute the distance
        to the next point on the line in the same class. If the distance to that point is too large, we try to add more
        points between these two. If after adding such points, there are still significant gaps between consecutive
        points, this indicates that the fault line between class i_class and class j_class consists of several separated
        components. In this case, we split the set of points on the current fault line into groups where each group
        represents one component.
        3D: YNI
     3) building block expand:
        We try to continue each segment of the fault line after the last and before the first known point on that
        component in 2D or try to expand the decision surface until its boundaries in 3D.
     4) building block adapt:
        We improve the approximation of the components of the fault lines/surfaces by adding or removing points
        according to (estimated) curvature.

    Args:
        barycentres (npt.ArrayLike):
            set of barycentres. Shape: (n_barys, ndim)
        point_set (npt.ArrayLike):
            initial point set. Shape: (n_points, ndim)
        class_of_barys (npt.ArrayLike):
            class indices of the barycentres. Shape: (n_barys)
        class_of_points  (npt.ArrayLike):
            class indices of the points in points. Shape: (n_points)
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.

    Returns:
        point_sets_surface (npt.ArrayLike, dtype=object):
            array of arrays containing the components of the point sets between subsets. Shape: (n_classes, n_classes)
        left_domain_start (npt.ArrayLike, dtype=int):
            In 2D, this array codes information on whether a fault line starts on the domain boundary (value > 0, we
            refer to get_points_near_surface_2D for ASCII-art explaining how) or inside the domain (value = 0).
            Shape: (n_classes, n_classes; #components)
        left_domain_end (npt.ArrayLike, dtype=int):
            The same as left_domain_start, but for ending of fault lines
        success (bool):
            flag, if the fault lines have been found successfully
        normals_surface (npt.ArrayLike, dtype=object):
            in 2D, just a dummy return value, in 3D: array of arrays containing approximate outer normals for all
            points in several components of the point sets in point_sets_surface (organized analogously to
            point_sets_surface). Shape: (n_classes, n_classes)

    Raises:
        Exception: If class_vals in np is not a natural number.
    """

    # initialise with dummy return values
    left_domain_start = 0
    left_domain_end = 0

    success = False

    # number of barycentres
    n_barys = barycentres.shape[0]

    # dimension
    n_dim = point_set.shape[1]

    if n_dim < 2 or n_dim > 3:
        raise Exception("The dimension of the domain must be 2 or 3.")

    # Array containing the class values. These are not necessarily the class indices. Imagine that f(\Omega) = {1,2,5}.
    # Then class_vals = [1,2,5], whereas the class indices range from 0 to 2.
    class_vals = np.unique(class_of_points).astype(int)

    # number of different classes in the point set
    n_classes = class_vals.shape[0]

    if np.any(class_vals < 1):
        raise Exception("Class values < 1 occurred. Classes must be however natural numbers.")

    normals_surface = 0

    # Get the opposite mapping: Class value -> i
    class_vals_inv = np.ones((np.max(class_vals) + 1), dtype=int) * -1
    class_vals_inv[class_vals] = np.arange(0, n_classes)

    # Possibly, there will be several lines or surfaces separating the different classes. We store the points near the
    # surface as follows: All points in class i that belong to the surface separating class i from j are stored in
    # point_sets_surface[i,j], all points in class j that belong to the surface separating class i from j are stored in
    # point_sets_surface[j,i].
    point_sets_surface = np.empty((n_classes, n_classes), dtype=object)

    # number of points on fault line point_sets_surface[i, j]
    n_points_surf = np.empty((n_classes, n_classes), dtype=np.ndarray)

    # We preallocate the arrays in order to avoid dynamic resizing. We shorten them later.
    for i_class in range(n_classes):
        for j_class in range(n_classes):
            if i_class != j_class:
                point_sets_surface[i_class, j_class] = np.empty(1, dtype=object)
                point_sets_surface[i_class, j_class][0] = np.full((n_barys, n_dim), np.nan)

            n_points_surf[i_class, j_class] = np.array([0], dtype=int)

    for i_class in range(n_classes):
        idx_i_class = class_vals[i_class]

        logging.debug('-- compute initial set of boundary points, class ' + str(idx_i_class))

        # all points and barycentres not in the current class
        points_not_in_class = np.concatenate((point_set[class_of_points != idx_i_class],
                                              barycentres[class_of_barys != idx_i_class]), axis=0)

        class_of_points_aux = np.concatenate((class_of_points[class_of_points != idx_i_class],
                                              class_of_barys[class_of_barys != idx_i_class]), axis=0)

        # all barycentres in current class
        barycentres_in_class = barycentres[class_of_barys == idx_i_class]

        n_barys_in_class = barycentres_in_class.shape[0]
        n_points_not_in_class = points_not_in_class.shape[0]

        dist_mat_aux = np.zeros((n_points_not_in_class, n_barys_in_class, n_dim))
        for j_class in range(n_barys_in_class):
            dist_mat_aux[:, j_class, :] = points_not_in_class - barycentres_in_class[j_class]

        dist_mat_aux = np.power(dist_mat_aux, 2)
        dist_mat_aux[:, :, 0] = np.sum(dist_mat_aux, axis=2)

        # distance matrix to find the nearest point to the current barycentre which is not in its class
        dist_mat = np.sqrt(dist_mat_aux[:, :, 0])

        i_idx_nearest_neighbour = np.argsort(dist_mat, axis=0, kind='stable')

        idx_next_point_not_in_class = i_idx_nearest_neighbour[0, 0:n_barys_in_class]

        class_next_point_not_in_class = (class_of_points_aux[idx_next_point_not_in_class]).astype(int)
        point_in_class = barycentres_in_class[0:n_barys_in_class]
        point_not_in_class = points_not_in_class[idx_next_point_not_in_class]
        idx_i_class_vec = idx_i_class * np.ones(n_barys_in_class, dtype=int)

        points_left_final, points_right_final, finished = triplets_by_bisection(point_in_class,
                                                                                point_not_in_class, idx_i_class_vec,
                                                                                class_next_point_not_in_class,
                                                                                problem_descr,
                                                                                fault_approx_pars)
        for i_point in range(n_barys_in_class):
            # index of the point not in the current class, but nearest to the given one
            if finished[i_point]:
                i_aux = class_vals_inv[class_next_point_not_in_class[i_point]]

                n_points_surf[i_class, i_aux][0] += 1
                n_points_surf[i_aux, i_class][0] += 1

                point_sets_surface[i_class, i_aux][0][n_points_surf[i_class, i_aux][0] - 1] = \
                    points_left_final[i_point]
                point_sets_surface[i_aux, i_class][0][n_points_surf[i_aux, i_class][0] - 1] = \
                    points_right_final[i_point]
            else:
                warn("Bisection failed in building block iniapprox.")

    # Last part of iniapprox: remove duplicates and clusters. Two points are considered duplicate if closer than eps.
    # Moreover, remove almost duplicates: points that are closer to their nearest neighbour than
    # min_dist_factor*max_dist_for_surface_points are collected into clusters which are then reduced according to their
    # geometry (being in a cluster is transitive here).
    logging.debug('-- remove duplicates in initial point sets')

    for i_class in range(n_classes-1):
        for j_class in range(i_class+1, n_classes):
            point_sets_surface[i_class, j_class][0], point_sets_surface[j_class, i_class][0] = remove_duplicates(
                point_sets_surface[i_class, j_class][0], point_sets_surface[j_class, i_class][0],
                fault_approx_pars.eps)

            n_points_surf[i_class, j_class][0] = point_sets_surface[i_class, j_class][0].shape[0]
            n_points_surf[j_class, i_class][0] = point_sets_surface[j_class, i_class][0].shape[0]

    if problem_descr.extended_stats:
        stats.point_sets_surf.append(copy.deepcopy(point_sets_surface))
        stats.n_points_surf.append(copy.deepcopy(n_points_surf))
        stats.ncalls.append(ncalls.total)
        stats.pos_in_code.append('initial')

    logging.debug('-- remove clusters in initial point sets')

    for i_class in range(n_classes-1):
        for j_class in range(i_class+1, n_classes):
            point_sets_surface[i_class, j_class][0], point_sets_surface[j_class, i_class][0], \
                n_points_surf[i_class, j_class][0] = remove_clusters(point_sets_surface[i_class, j_class][0],
                                                                     point_sets_surface[j_class, i_class][0],
                                                                     n_points_surf[i_class, j_class][0],
                                                                     fault_approx_pars)

            n_points_surf[j_class, i_class][0] = n_points_surf[i_class, j_class][0]

    if problem_descr.extended_stats:
        stats.point_sets_surf.append(copy.deepcopy(point_sets_surface))
        stats.n_points_surf.append(copy.deepcopy(n_points_surf))
        stats.ncalls.append(ncalls.total)
        stats.pos_in_code.append('after_cluster_rem')

    # Consistency check: if there are no points near any fault line at all (for whatever reason), skip the
    # computation.
    points_on_fault_lines = False
    for i_class in range(n_classes - 1):
        for j_class in range(i_class + 1, n_classes):
            points_on_fault_lines = points_on_fault_lines or np.any(
                point_sets_surface[i_class, j_class][0][:, 0])

    if not points_on_fault_lines:
        warn("Unable to find any points near a fault line. Skip computation.")

        return point_sets_surface, left_domain_start, left_domain_end, success, normals_surface

    if n_dim == 2:
        point_sets_surface, n_points_surf, left_domain_start, left_domain_end, success = \
            get_triplets_near_fault_2d(point_sets_surface, n_points_surf, n_classes, class_vals,
                                       fault_approx_pars, problem_descr)
    elif n_dim == 3:
        point_sets_surface, normals_surf, n_points_surf, success = \
            get_points_near_surface_3d(point_sets_surface, n_points_surf, n_classes, class_vals, fault_approx_pars,
                                       problem_descr)

    return point_sets_surface, left_domain_start, left_domain_end, success, normals_surface


def get_triplets_near_fault_2d(point_sets_surface: npt.ArrayLike, n_points_surf: npt.ArrayLike, n_classes: int,
                               class_vals: npt.ArrayLike, fault_approx_pars: FaultApproxParameters,
                               problem_descr: ProblemDescr) -> \
        Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, bool]:
    """ This function provides the 2d-specific part of the algorithm. For details, we refer to "Detecting and
    approximating decision boundaries in low dimensional spaces" (http://arxiv.org/abs/2302.08179).

    Args:
        point_sets_surface (npt.ArrayLike, dtype=object):
            array of arrays containing the components of the point sets between subsets. Shape: (n_classes, n_classes)
        n_points_surf (npt.ArrayLike, dtype=np.ndarray):
            array of arrays containing the number of points in the point sets. Shape: (n_classes, n_classes)
        n_classes (int):
            number of different classes
        class_vals (npt.ArrayLike):
            Array containing the class values. These are not necessarily the class indices. Imagine that
            f(\Omega) = {1,2,5}. Then, class_vals = [1,2,5], whereas the class indices range from 0 to 2.
            Shape: (nclasses,)
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.

    Returns:
        point_sets_surface (npt.ArrayLike, dtype=object):
            array of arrays containing the components of the point sets between subsets. Shape: (n_classes, n_classes)
        n_points_surf (npt.ArrayLike, dtype=np.ndarray):
            array of arrays containing the number of points in the point sets. Shape: (n_classes, n_classes)
        left_domain_start (npt.ArrayLike, dtype=int):
            In 2D, this array codes information on whether a fault line starts on the domain boundary (value > 0, we
            refer to the code below for ASCII-art explaining how) or inside the domain (value = 0).
            Shape: (n_classes, n_classes, #components)
        left_domain_end (npt.ArrayLike, dtype=int):
            The same as left_domain_start, but for ending of fault lines
        bsuccessful (bool):
            flag, if the fault lines have been processed successfully
    """

    # desired maximum distance of a point on the fault line to the next one
    max_dist_for_surface_points = fault_approx_pars.max_dist_for_surface_points

    num_comps_per_fault_line = np.zeros((n_classes, n_classes), dtype=int)
    bsuccessful = False

    for i_class in range(n_classes):
        for j_class in range(i_class + 1, n_classes):

            logging.debug(f'-- fill gaps on boundary between classes {class_vals[i_class]} and {class_vals[j_class]}')

            # Up to this point, there is only one segment per fault line.
            if n_points_surf[i_class, j_class][0] > 0:
                # Fill possible gaps in the fault line by adding more points in between consecutive points if necessary.

                i_try = 1

                # We need to repeat this process: It may happen that due to filling some gaps in the first pass, the
                # assumed shape of the fault line changes substantially due to additional information at hand.
                # In this case, new gaps may arise which then have to be filled again. We repeat that process until no
                # more gaps are found and filled or the maximal number of filling passes has been reached.
                points_added = True
                while points_added and i_try <= fault_approx_pars.max_trials_for_filling_gaps_in_lines:
                    point_sets_surface[i_class, j_class][0], point_sets_surface[j_class, i_class][0], \
                        n_points_surf[i_class, j_class][0], points_added, success_fill_gaps, \
                        = fill_2d(point_sets_surface[i_class, j_class][0],
                                  point_sets_surface[j_class, i_class][0],
                                  i_class, j_class, class_vals, fault_approx_pars, problem_descr,
                                  np.array([0, 0, 1]))

                    if not success_fill_gaps:
                        warn(f'get_triplets_near_fault failed for classes {class_vals[i_class]}'
                             f'and {class_vals[j_class]}.')

                        # dummy return values
                        left_domain_start = 0
                        left_domain_end = 0
                        return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, bsuccessful

                        # In fact, points have been added on both sides of the fault line simultaneously, forming
                        # triplets.
                    n_points_surf[j_class, i_class][0] = n_points_surf[i_class, j_class][0]

                    # sort again
                    idx_points_surf_ordered, sorting_successful = sort_on_fault(
                        point_sets_surface[i_class, j_class][0], True, problem_descr,
                        fault_approx_pars)
                    point_sets_surface[i_class, j_class][0] = point_sets_surface[i_class, j_class][0][
                        idx_points_surf_ordered[0]]
                    point_sets_surface[j_class, i_class][0] = point_sets_surface[j_class, i_class][0][
                        idx_points_surf_ordered[0]]

                    if not sorting_successful:
                        warn(f'Get_triplets_near_fault failed for classes {class_vals[i_class]}'
                             f' and {class_vals[j_class]}.')

                        # dummy return values
                        left_domain_start = 0
                        left_domain_end = 0
                        return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, bsuccessful

                    i_try += 1

                # Test, if the fault line between class i_class and j_class consists in fact of several unconnected
                # components.
                point_sets_surface[i_class, j_class], num_comps_per_fault_line[i_class, j_class], \
                    n_points_surf[i_class, j_class] = divide_into_components(point_sets_surface[i_class, j_class][0],
                                                                             max_dist_for_surface_points)

                for i_seg in range(num_comps_per_fault_line[i_class, j_class]):
                    intersection = self_intersection(point_sets_surface[i_class, j_class][i_seg])

                    if intersection:
                        warn(f'Self-intersecting boundary components between classes {class_vals[i_class]} and '
                             f'{class_vals[j_class]} detected. Skip computation.')
                        left_domain_end = 0
                        left_domain_start = 0
                        return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, bsuccessful

                # Repeat sorting for the counterparts on the other side of the fault line.
                num_comps_per_fault_line[j_class, i_class] = num_comps_per_fault_line[i_class, j_class]

                num_points_total = n_points_surf[j_class, i_class][0]

                # Hack: as i_class < j_class, n_points_surf[i_class, j_class] has not been updated yet and therefore
                # contains the number of points in S_i,j prior to separation in components.
                # Therefore, this entry contains the total number of points in S_i,j.
                n_points_surf[j_class, i_class] = n_points_surf[i_class, j_class].copy()

                points_temp = np.empty(num_comps_per_fault_line[j_class, i_class], dtype=np.ndarray)
                i_start = 0

                i_end = point_sets_surface[i_class, j_class][0].shape[0]
                for i_seg in range(num_comps_per_fault_line[j_class, i_class] - 1):
                    points_temp[i_seg] = point_sets_surface[j_class, i_class][0][i_start:i_end]
                    i_start = i_end
                    i_end = i_start + point_sets_surface[i_class, j_class][i_seg + 1].shape[0]
                points_temp[num_comps_per_fault_line[j_class, i_class] - 1] = \
                    point_sets_surface[j_class, i_class][0][i_start:num_points_total]
                point_sets_surface[j_class, i_class] = points_temp

    if problem_descr.extended_stats:
        stats.point_sets_surf.append(copy.deepcopy(point_sets_surface))
        stats.n_points_surf.append(copy.deepcopy(n_points_surf))
        stats.ncalls.append(ncalls.total)
        stats.pos_in_code.append('after_filling_gaps')

    # maximal number of components per fault line
    max_num_comps = np.max(num_comps_per_fault_line)

    left_domain_start = - 1 * np.ones((n_classes, n_classes, max_num_comps), dtype=int)
    left_domain_end = - 1 * np.ones((n_classes, n_classes, max_num_comps), dtype=int)

    # Extend all segments of a fault line until near to their true start and end points.
    # When extended, perform adaptive refinement/coarsening.
    #
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
    logging.debug('-- expand boundaries')

    for i_class in range(n_classes):
        for j_class in range(i_class + 1, n_classes):
            if n_points_surf[i_class, j_class][0] > 0:
                for i_seg in range(num_comps_per_fault_line[i_class, j_class]):
                    num_points_comp = n_points_surf[i_class, j_class][i_seg]

                    idx_points_surf_ordered = np.arange(num_points_comp)
                    # Try to extrapolate the fault line by fitting a polynomial curve to the first and the last points
                    # in the point set.

                    # If the component consists of one point only: find more by scattering.
                    if num_points_comp == 1:
                        single_point = point_sets_surface[i_class, j_class][i_seg]
                        aux_vec = np.array([[-1.5, -1], [0.5, -1], [0.5, 1], [-1.5, 1]])
                        pairs_found = False

                        for i_try in range(3):
                            # scatter four points around the single one
                            x_test = single_point + 1 / 2 ** (
                                    i_try + 1) * fault_approx_pars.max_dist_for_surface_points * aux_vec
                            class_aux = compute_classification(x_test, problem_descr)

                            # points in class i_class
                            x_test_1 = x_test[class_aux == class_vals[i_class]]

                            # points in class j_class
                            x_test_2 = x_test[class_aux == class_vals[j_class]]

                            # Build all combinations between these points.
                            n_points_i = x_test_1.shape[0]
                            n_points_j = x_test_2.shape[0]
                            n_combis = n_points_i * n_points_j
                            points_i_class = np.zeros((n_combis, 2))
                            points_j_class = np.zeros((n_combis, 2))
                            for i in range(n_points_i):
                                points_j_class[i * n_points_j:(i + 1) * n_points_j] = x_test_2
                                points_i_class[i * n_points_j:(i + 1) * n_points_j] = (np.ones((n_points_j, 1)) *
                                                                                       x_test_1[i])

                            for i in range(n_combis):
                                point_left, point_right, finished = single_triplet_by_bisection(
                                    points_i_class[i], points_j_class[i], class_vals[i_class],
                                    class_vals[j_class],
                                    problem_descr, fault_approx_pars)

                                if finished:
                                    n_points_surf[i_class, j_class][i_seg] += 1
                                    n_points_surf[j_class, i_class][i_seg] += 1

                                    if n_points_surf[i_class, j_class][i_seg] > \
                                            point_sets_surface[i_class, j_class][i_seg].shape[0]:
                                        point_sets_surface[i_class, j_class][i_seg] = \
                                            np.concatenate((point_sets_surface[i_class, j_class][i_seg],
                                                            point_left[np.newaxis, :]))
                                        point_sets_surface[j_class, i_class][i_seg] = \
                                            np.concatenate((point_sets_surface[j_class, i_class][i_seg],
                                                            point_right[np.newaxis, :]))
                                    else:
                                        point_sets_surface[i_class, j_class][i_seg][
                                            n_points_surf[i_class, j_class][i_seg] - 1] = point_left
                                        point_sets_surface[j_class, i_class][i_seg][
                                            n_points_surf[j_class, i_class][i_seg] - 1] = point_right
                                    pairs_found = True

                            if pairs_found:
                                break

                        num_points_comp = n_points_surf[i_class, j_class][i_seg]
                        idx_points_surf_ordered, sorting_successful = sort_on_fault(
                            point_sets_surface[i_class, j_class][i_seg], True, problem_descr,
                            fault_approx_pars)
                        idx_points_surf_ordered = idx_points_surf_ordered[0]

                    if num_points_comp == 1:
                        warn(f'There is only one point on the boundary between classes {class_vals[i_class]} '
                             f'and {class_vals[j_class]}. Searching more by scattering failed.')
                        warn('Finding subdomains terminated.')

                        # dummy return values
                        left_domain_start = 0
                        left_domain_end = 0
                        return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, bsuccessful

                    # beginning of fault line
                    (idx_points_surf_ordered, left_domain_start, n_points_surf[i_class, j_class][i_seg],
                     point_sets_surface[i_class, j_class][i_seg],
                     point_sets_surface[j_class, i_class][i_seg], _, _) = expand_2d(
                        left_domain_start, idx_points_surf_ordered, i_class, j_class, i_seg, class_vals,
                        point_sets_surface[i_class, j_class][i_seg],
                        point_sets_surface[j_class, i_class][i_seg], np.array([0, 0, 1]),
                        problem_descr,
                        fault_approx_pars, 0)

                    n_points_surf[j_class, i_class][i_seg] = n_points_surf[i_class, j_class][i_seg]

                    point_sets_surface[i_class, j_class][i_seg] = \
                        point_sets_surface[i_class, j_class][i_seg][idx_points_surf_ordered]
                    point_sets_surface[j_class, i_class][i_seg] = \
                        point_sets_surface[j_class, i_class][i_seg][idx_points_surf_ordered]

                    idx_points_surf_ordered = np.arange(n_points_surf[i_class, j_class][i_seg])

                    # end of fault line
                    (idx_points_surf_ordered, left_domain_end, n_points_surf[i_class, j_class][i_seg],
                     point_sets_surface[i_class, j_class][i_seg],
                     point_sets_surface[j_class, i_class][i_seg], resort, _) = expand_2d(
                        left_domain_end, idx_points_surf_ordered, i_class, j_class, i_seg, class_vals,
                        point_sets_surface[i_class, j_class][i_seg],
                        point_sets_surface[j_class, i_class][i_seg], np.array([0, 0, 1]),
                        problem_descr,
                        fault_approx_pars, 1)

                    n_points_surf[j_class, i_class][i_seg] = n_points_surf[i_class, j_class][i_seg]

                    point_sets_surface[i_class, j_class][i_seg] = \
                        point_sets_surface[i_class, j_class][i_seg][idx_points_surf_ordered]
                    point_sets_surface[j_class, i_class][i_seg] = \
                        point_sets_surface[j_class, i_class][i_seg][idx_points_surf_ordered]

                    # Resort the points on the fault line if indicated.
                    if resort:
                        idx_points_surf_ordered, sorting_successful = sort_on_fault(
                            point_sets_surface[i_class, j_class][i_seg], True, problem_descr,
                            fault_approx_pars)
                        point_sets_surface[i_class, j_class][i_seg] = \
                            point_sets_surface[i_class, j_class][i_seg][idx_points_surf_ordered[0]]
                        point_sets_surface[j_class, i_class][i_seg] = \
                            point_sets_surface[j_class, i_class][i_seg][idx_points_surf_ordered[0]]

                        if not sorting_successful:
                            warn(f'Resorting of points during expanding the boundary between ' 
                                 f'classes {class_vals[i_class]} and {class_vals[j_class]} failed.')

                            # dummy return values
                            left_domain_start = 0
                            left_domain_end = 0
                            return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, bsuccessful

                    # Remove duplicates: Duplicates should actually not occur, but they can due to wrong sorting which
                    # remained unnoticed.
                    point_sets_surface[i_class, j_class][i_seg], point_sets_surface[j_class, i_class][i_seg] = \
                        remove_duplicates(point_sets_surface[i_class, j_class][i_seg],
                                          point_sets_surface[j_class, i_class][i_seg],
                                          fault_approx_pars.eps)

                # Consistency check: test, if the different segments intersect. It may happen that due to failed
                # sorting, some single point or so was erroneously considered a separate boundary segment. Now,
                # additional points have been found. If these different boundary segments are really different ones,
                # they do not intersect.
                if num_comps_per_fault_line[i_class, j_class] > 0:
                    for i_seg in range(num_comps_per_fault_line[i_class, j_class]):

                        j_seg = i_seg + 1
                        # A while-loop is better here, as num_comps_per_fault_line may decrease by merging, which is
                        # not reflected in an ordinary for-loop.
                        while j_seg <= num_comps_per_fault_line[i_class, j_class] - 1:

                            do_not_intersect = not poly_lines_intersect(
                                point_sets_surface[i_class, j_class][i_seg],
                                point_sets_surface[i_class, j_class][j_seg], False)
                            point_on_line = is_on_poly_line(
                                point_sets_surface[i_class, j_class][i_seg],
                                point_sets_surface[i_class, j_class][j_seg])

                            # If two segments intersect, they must in fact be one segment. Therefore, we merge
                            # them.
                            if not do_not_intersect or point_on_line:
                                # Maybe these two segments are in fact the same: try to merge them.
                                test = np.concatenate((point_sets_surface[i_class, j_class][i_seg],
                                                       point_sets_surface[i_class, j_class][j_seg]), axis=0)
                                idx_points_surf_ordered, sorting_successful = sort_on_fault(test, True,
                                                                                            problem_descr,
                                                                                            fault_approx_pars)

                                if sorting_successful:
                                    test = test[idx_points_surf_ordered[0]]
                                else:
                                    warn(f'Two components of the boundary between classes {class_vals[i_class]} '
                                         f'and {class_vals[j_class]} were found to intersect and their merge failed.')
                                    return point_sets_surface, n_points_surf, left_domain_start, left_domain_end,\
                                        bsuccessful

                                # t´Test, if the new boundary does not intersect itself.
                                do_not_intersect = not self_intersection(test)
                                if do_not_intersect:
                                    point_sets_surface[i_class, j_class][i_seg] = test
                                    point_sets_surface[i_class, j_class] = np.delete(point_sets_surface[
                                                                                         i_class, j_class], j_seg)

                                    n_points_surf[i_class, j_class][i_seg] = (
                                            n_points_surf[i_class, j_class][i_seg] +
                                            n_points_surf[i_class, j_class][j_seg])
                                    n_points_surf[i_class, j_class] = np.delete(n_points_surf[i_class, j_class],
                                                                                j_seg)
                                    if n_points_surf[j_class, i_class] is not n_points_surf[i_class, j_class]:
                                        n_points_surf[j_class, i_class][i_seg] = (
                                                n_points_surf[j_class, i_class][i_seg] +
                                                n_points_surf[j_class, i_class][j_seg])
                                    n_points_surf[j_class, i_class] = np.delete(n_points_surf[j_class, i_class],
                                                                                j_seg)

                                    point_sets_surface[j_class, i_class][i_seg] = np.concatenate(
                                        (point_sets_surface[j_class, i_class][i_seg],
                                         point_sets_surface[j_class, i_class][j_seg]), axis=0)
                                    point_sets_surface[j_class, i_class] = np.delete(point_sets_surface[
                                                                                         j_class, i_class], j_seg)

                                    num_comps_per_fault_line[i_class, j_class] -= 1
                                    num_comps_per_fault_line[j_class, i_class] -= 1

                                    # Reset left_domain_start and left_domain_end, as the corresponding segment is
                                    # gone.
                                    left_domain_start[i_class, j_class, j_seg:-1] = \
                                        left_domain_start[i_class, j_class, j_seg + 1:]

                                    left_domain_end[i_class, j_class, j_seg:-1] = \
                                        left_domain_end[i_class, j_class, j_seg + 1:]

                                    left_domain_end[i_class, j_class, -1] = -1
                                    left_domain_start[i_class, j_class, -1] = -1

                                    # maximal number of components per fault line
                                    max_num_segs_new = np.max(num_comps_per_fault_line)

                                    if max_num_segs_new < max_num_comps:
                                        left_domain_start = left_domain_start[:, :, 0: max_num_segs_new]
                                        left_domain_end = left_domain_end[:, :, 0: max_num_segs_new]
                                        max_num_comps = max_num_segs_new

                                else:
                                    warn(f'Two components of the boundary between classes {class_vals[i_class]} and '
                                         f'{class_vals[j_class]} were'
                                         f' found to intersect. However, their merge lead to another intersecting'
                                         f' boundary components. Stop computation.')
                                    return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, \
                                        bsuccessful
                            else:
                                j_seg = j_seg + 1

    if problem_descr.extended_stats:
        stats.point_sets_surf.append(copy.deepcopy(point_sets_surface))
        stats.n_points_surf.append(copy.deepcopy(n_points_surf))
        stats.ncalls.append(ncalls.total)
        stats.pos_in_code.append('after_prol_lines')

    # adaptive refinement and coarsening according to estimated curvature
    for i_class in range(n_classes):
        for j_class in range(i_class + 1, n_classes):

            logging.debug(
                f'-- adaptive refinement on boundary between classes {class_vals[i_class]} and {class_vals[j_class]}')

            if n_points_surf[i_class, j_class][0] > 0:

                # adaptive refinement and coarsening according to estimated curvature
                for i_seg in range(num_comps_per_fault_line[i_class, j_class]):
                    point_sets_surface, n_points_surf = adapt_2d(point_sets_surface, n_points_surf,
                                                                 i_class, j_class, i_seg, problem_descr,
                                                                 fault_approx_pars, class_vals)

    if problem_descr.extended_stats:
        stats.point_sets_surf.append(copy.deepcopy(point_sets_surface))
        stats.n_points_surf.append(copy.deepcopy(n_points_surf))
        stats.ncalls.append(ncalls.total)
        stats.pos_in_code.append('after_adaptive_ref')

    # If necessary, reverse the order of fault line segments.
    for i_class in range(n_classes):
        for j_class in range(i_class + 1, n_classes):
            for i_seg in range(num_comps_per_fault_line[i_class, j_class]):
                if n_points_surf[i_class, j_class][i_seg] > 1:
                    # Sort the points near the fault line such that the subdomain i_class is right to the line.

                    # Take a point from somewhere in the middle of the line.
                    i_idx = int(np.maximum(1, np.round(n_points_surf[i_class, j_class][i_seg] / 2) - 1))

                    # vector pointing from point with index i_idx to i_idx-1
                    aux_vec = (point_sets_surface[i_class, j_class][i_seg][i_idx] -
                               point_sets_surface[i_class, j_class][i_seg][i_idx - 1])
                    normal_vec = fault_approx_pars.alpha * np.array([aux_vec[1], -1 * aux_vec[0]])

                    # The points are known up to 2*abstol_bisection only. Therefore, a normal vector smaller than this
                    # does not make sense. If the boundary points are very close, our simple estimation of the normal
                    # vector may be unreliable, such that we should elongate it even a little more to be safe that in
                    # case of wrong ordering, the auxiliary point is really beyond the boundary of the subdomain. This
                    # leads to the factor of 3.
                    if norm(normal_vec) < 3.0 * fault_approx_pars.abstol_bisection:
                        normal_vec = normal_vec / norm(normal_vec) * 3.0 * fault_approx_pars.abstol_bisection

                    # The normal vector can be overly large (this can happen e.g. after adaptive refinement and
                    # coarsening of a straight line, which is represented by a few points only). Then, we shorten it to
                    # a size of max_dist_for_surface_points.
                    elif norm(normal_vec) > max_dist_for_surface_points:
                        normal_vec = normal_vec / norm(normal_vec) * max_dist_for_surface_points

                    # The aux point should be in class class_vals[i_class]. If not, either the point is badly chosen or
                    # the order is wrong.
                    # Special treatment is required if the point with index i_idx is the end of the boundary segment.
                    # This may happen if e.g. the boundary segment consists of two points only. Above, we excluded that
                    # i_idx is the first point on the boundary segment.
                    # The first and the last point on a boundary segment is unsuitable for our purpose.

                    # Point is somewhere in the middle: just take that point. Note that i_idx >= 1, such that
                    # n_points_surf[i_class, j_class][i_comp] >= 3
                    if i_idx < n_points_surf[i_class, j_class][i_seg] - 1:
                        aux_point = point_sets_surface[i_class, j_class][i_seg][i_idx] - normal_vec

                    # Very last point, more than two points on the segment: take the predecessor
                    elif i_idx == n_points_surf[i_class, j_class][i_seg] - 1 and \
                            n_points_surf[i_class, j_class][i_seg] > 2:
                        aux_point = point_sets_surface[i_class, j_class][i_seg][i_idx - 1] - normal_vec

                    # Very last point, and just two points on the segment: compute an auxiliary point based on the mean
                    # of the two points.
                    else:
                        aux_point = point_sets_surface[i_class, j_class][i_seg][i_idx - 1] + 0.5 * aux_vec - normal_vec

                    class_aux = compute_classification(np.array([aux_point]), problem_descr)
                    if class_aux == class_vals[j_class]:
                        point_sets_surface[i_class, j_class][i_seg] = np.flip(
                            point_sets_surface[i_class, j_class][i_seg], axis=0)

                        # Start point becomes end point.
                        left_domain_end[i_class, j_class, i_seg], left_domain_start[
                            i_class, j_class, i_seg] = (left_domain_start[i_class, j_class, i_seg],
                                                        left_domain_end[i_class, j_class, i_seg])
                    elif class_aux == class_vals[i_class]:
                        point_sets_surface[j_class, i_class][i_seg] = np.flip(
                            point_sets_surface[j_class, i_class][i_seg], axis=0)

                    # It might happen that the auxiliary point is outside the domain or inside an entirely different
                    # subdomain. In this case, we reverse it and try again. If it still fails, we give up.
                    else:
                        # Point is somewhere in the middle: just take that point. Note that i_idx >= 1, such that
                        # n_points_surf[i_class, j_class][i_comp] >= 3.
                        if i_idx < n_points_surf[i_class, j_class][i_seg] - 1:
                            aux_point = point_sets_surface[i_class, j_class][i_seg][i_idx - 1] + normal_vec

                        # Very last point, more than two points on the segment: take the predecessor.
                        elif i_idx == n_points_surf[i_class, j_class][i_seg] - 1 and \
                                n_points_surf[i_class, j_class][i_seg] > 2:
                            aux_point = point_sets_surface[i_class, j_class][i_seg][i_idx - 1] + normal_vec

                        # Very last point, and just two points on the segment: compute an auxiliary point based on the
                        # mean of the two points.
                        else:
                            aux_point = point_sets_surface[i_class, j_class][i_seg][i_idx - 1] + \
                                        0.5 * aux_vec + normal_vec

                        class_aux_2 = compute_classification(aux_point, problem_descr)

                        if class_aux_2 == class_vals[i_class]:
                            point_sets_surface[i_class, j_class][i_seg] = np.flip(
                                point_sets_surface[i_class, j_class][i_seg], axis=0)

                            # Start point becomes end point.
                            left_domain_end[i_class, j_class, i_seg], left_domain_start[
                                i_class, j_class, i_seg] = (left_domain_start[i_class, j_class, i_seg],
                                                            left_domain_end[i_class, j_class, i_seg])
                        elif class_aux_2 == class_vals[j_class]:
                            point_sets_surface[j_class, i_class][i_seg] = np.flip(
                                point_sets_surface[j_class, i_class][i_seg], axis=0)
                        else:
                            warn(f'Determination of orientation failed for the boundary between classes '
                                 f'{class_vals[i_class]} and {class_vals[j_class]}.')
                            return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, bsuccessful
    bsuccessful = True
    return point_sets_surface, n_points_surf, left_domain_start, left_domain_end, bsuccessful


def get_points_near_surface_3d(point_sets_surface, n_points_surf, n_classes, class_vals, fault_approx_pars,
                               problem_descr):
    normals_surface = 0
    bsuccessful = False
    return point_sets_surface, normals_surface, n_points_surf, bsuccessful


def remove_duplicates(points_iclass: npt.ArrayLike, points_jclass: npt.ArrayLike, epstol: float) ->\
        Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    This function removes duplicate triplets in points_iclass/points_jclass. Triplets closer than epstol are considered
    identical.

    Args:
        points_iclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)
        points_jclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)
        epstol (float):
            tolerance for assuming identity

    Returns:
        points_iclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)
        points_jclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)

    Raises:
        None
    """
    num_aux = points_iclass.shape[0]

    # Inner loop is pointless if the point with index i has been removed before.
    for i in range(num_aux):
        if not np.isnan(points_iclass[i, 0]):
            for j in range(i + 1, num_aux):
                if not np.isnan(points_iclass[j, 0]):
                    if abs(norm(points_iclass[i] - points_iclass[j])) < epstol:
                        points_iclass[j] = np.nan
                        points_jclass[j] = np.nan

    points_iclass = points_iclass[~np.isnan(points_iclass[:, 0])]
    points_jclass = points_jclass[~np.isnan(points_jclass[:, 0])]

    return points_iclass, points_jclass


def remove_clusters(points_iclass: npt.ArrayLike, points_jclass: npt.ArrayLike, n_points: int,
                    fault_approx_pars: FaultApproxParameters) -> Tuple[npt.ArrayLike, npt.ArrayLike, int]:
    """This function reduces clusters according to their geometry.
    If the cluster is very small, we just take the mean value of its points and replace the cluster by its mean.
    If the cluster is larger, it is important to remove the inner points, not the points on the boundary of the cluster.
    It is therefore straightforward in 2D to consider e.g. three consecutive points and to remove the one in the middle
    if these three points are very close. However, there is no sorting of the points along the fault line yet such
    that it is hard to tell which of the points is the one in the middle. Therefore, we perform local sorting. To do
    so, we assume that the fault line is somehow smooth such that we can find a new coordinate system in which the
    fault line locally resembles the x-axis. This coordinate system allows us then to establish sorting of the points
    by just comparing their (new) x-coordinates.
    We reduce the 3D-case to the 2D case. If there is a cluster, we have to find a representative point or, if the
    cluster is larger, some of them. Thanks to the eigendecomposition, we find the axis the cluster is oriented to and
    con sort the cluster points according to it. Then, we proceed as in 2D.

    Args:
        points_iclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)
        points_jclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)
        n_points (int):
            number of points in points_iclass and points_jclass
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.

    Returns:
        points_iclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)
        points_jclass (npt.ArrayLike):
            array of points. Shape (n_points, n_dim)
        n_points (int):
            number of points in points_iclass and points_jclass
    """

    min_dist_factor = fault_approx_pars.min_dist_factor
    max_dist_for_surface_points = fault_approx_pars.max_dist_for_surface_points

    # compute distance matrix
    dist_mat = compute_dist_mat(points_iclass)

    # As the true number of points is decreased when removing clusters, we store the original number of points.
    n_points_ini = n_points

    already_in_cluster = np.zeros(n_points_ini, dtype=bool)

    cluster_points_all = list()
    for i_point in range(n_points_ini):

        # if point belongs to no cluster yet
        if not already_in_cluster[i_point]:
            cluster_points = dist_mat[i_point] < min_dist_factor * max_dist_for_surface_points

            # The point itself has distance zero, but this is no cluster, so there is no need to proceed.
            if np.flatnonzero(cluster_points).shape[0] > 1:
                finished = False
                while not finished:
                    cluster_points_aux = np.zeros(n_points_ini)
                    i_idx = np.flatnonzero(cluster_points)
                    for j_point in range(1, i_idx.shape[0]):
                        cluster_points_new = dist_mat[i_idx[
                            j_point]] < min_dist_factor * max_dist_for_surface_points
                        cluster_points_aux = np.maximum(cluster_points_aux, cluster_points_new)
                    finished = not np.any(cluster_points - cluster_points_aux)
                    cluster_points = np.maximum(cluster_points, cluster_points_aux)
                already_in_cluster = np.maximum(already_in_cluster, cluster_points).astype(bool)
                cluster_points_all.append(np.flatnonzero(cluster_points))
            else:
                already_in_cluster[i_point] = True

    # remove clusters
    for i_cluster, i_idx in enumerate(cluster_points_all):

        # If the cluster just consists of two points, take their mean value.
        num_points_in_cluster = i_idx.shape[0]
        if num_points_in_cluster == 2:

            mean_i_class = (1.0 / num_points_in_cluster * np.ones((1, num_points_in_cluster))) @ points_iclass[i_idx]
            mean_j_class = (1.0 / num_points_in_cluster * np.ones((1, num_points_in_cluster))) @ points_jclass[i_idx]
            points_iclass[i_idx[0], :] = mean_i_class
            points_jclass[i_idx[0], :] = mean_j_class
            points_iclass[i_idx[1], :] = np.nan
            points_jclass[i_idx[1], :] = np.nan

            n_points -= 1
        else:
            # Test, how large the cluster is.
            dist_mat_loc = dist_mat[i_idx[:, np.newaxis], i_idx]

            cluster_size = np.max(dist_mat_loc)

            # If the cluster is very small, take the mean.
            if cluster_size < min_dist_factor * max_dist_for_surface_points:
                mean_i_class = (1.0 / num_points_in_cluster * np.ones((1, num_points_in_cluster))) @ \
                                points_iclass[i_idx]
                mean_j_class = (1.0 / num_points_in_cluster * np.ones((1, num_points_in_cluster))) @ \
                                points_jclass[i_idx]
                points_iclass[i_idx[0], :] = mean_i_class
                points_jclass[i_idx[0], :] = mean_j_class
                points_iclass[i_idx[1:num_points_in_cluster], :] = np.nan
                points_jclass[i_idx[1:num_points_in_cluster], :] = np.nan

                n_points = n_points - num_points_in_cluster + 1

            # Take selected points from the cluster: leftmost and rightmost point and some points in the middle, if
            # necessary.
            else:
                num_points_middle = 0

                # cluster points
                cluster_points = points_iclass[i_idx]

                # local midpoint
                x_mid = np.matmul(1.0 / num_points_in_cluster * np.ones((1, num_points_in_cluster)), cluster_points)

                # Compute local coordinate system based on the normal vector.
                cluster_points -= x_mid
                q_mat = svd(cluster_points)[2].T

                # cluster points in the local coordinate system
                cluster_points = cluster_points @ q_mat

                # Sort points locally according to the longest axis. As singular values are provided in descending
                # order, this is always the first coordinate.
                i_idx_sorted = np.argsort(cluster_points[:, 0])
                cluster_points_sorted = cluster_points[i_idx_sorted]

                # cluster points in class i_class
                cluster_points_i_class = points_iclass[i_idx[i_idx_sorted]]

                # Replace the two cluster points with the lowest index with the two extremal cluster points, as these
                # will be kept in any case.
                points_iclass[i_idx[0]] = cluster_points_i_class[0]
                points_iclass[i_idx[1]] = cluster_points_i_class[num_points_in_cluster - 1]

                cluster_points_j_class = points_jclass[i_idx[i_idx_sorted]]
                points_jclass[i_idx[0]] = cluster_points_j_class[0]
                points_jclass[i_idx[1]] = cluster_points_j_class[num_points_in_cluster - 1]

                # If the cluster is large, it may make sense to include inner points as well. We start at one end and
                # consider the first point in the cluster which has sufficient distance from the start point, if such a
                # point exists. If it is sufficiently remote from the end of the cluster, we include this point. We
                # proceed until there are no such points left.
                stop = False
                idx_mid = np.array([], dtype=int)
                i_start = 0
                while not stop:
                    # first point which is sufficiently far away from the first in the cluster (The points are sorted
                    # according to the y-coordinate after orthogonal transformation!)
                    idx_point = np.flatnonzero(cluster_points_sorted[:, 0] - cluster_points_sorted[
                        i_start, 0] > min_dist_factor * max_dist_for_surface_points)

                    # There is no such point (may happen when some inner points have been found already).
                    if idx_point.shape[0] == 0:
                        stop = True
                    else:
                        # Test, if the first inner point sufficiently far away from the first point is sufficiently far
                        # away from the last point in the cluster. If so, it is a valid inner point.
                        idx_point = idx_point[0]
                        if cluster_points_sorted[num_points_in_cluster - 1, 0] - cluster_points_sorted[idx_point, 0]\
                                < min_dist_factor * max_dist_for_surface_points:
                            stop = True
                        else:
                            if idx_mid.size:
                                idx_mid = np.concatenate((idx_mid, np.array([idx_point])))
                            else:
                                idx_mid = np.array([idx_point])
                            cluster_points_sorted[i_start:idx_point - 1, 0] = cluster_points_sorted[idx_point, 0]
                            i_start = idx_point
                            num_points_middle += 1

                # Mark all points except of the extremal points and valid inner points for deletion.
                points_iclass[i_idx[2 + num_points_middle:num_points_in_cluster]] = np.nan
                points_iclass[i_idx[2:2 + num_points_middle]] = cluster_points_i_class[idx_mid]

                points_jclass[i_idx[2 + num_points_middle:num_points_in_cluster]] = np.nan
                points_jclass[i_idx[2:2 + num_points_middle]] = cluster_points_j_class[idx_mid]

                n_points += (-1 * num_points_in_cluster + 2 + num_points_middle)

    points_iclass = points_iclass[~np.isnan(points_iclass[:, 0])]
    points_iclass = points_iclass[0:n_points]

    points_jclass = points_jclass[~np.isnan(points_jclass[:, 0])]
    points_jclass = points_jclass[0:n_points]

    return points_iclass, points_jclass, n_points
