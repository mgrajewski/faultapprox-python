import warnings
from math import floor

import numpy as np
import numpy.typing as npt

from geometry import compute_dist_mat

from checkPoints import self_intersection
from entities.Entities import FaultApproxParameters, ProblemDescr


def sort_on_fault(points: npt.ArrayLike, bforce_combine: bool, problem_descr: ProblemDescr,
                  fault_approx_pars: FaultApproxParameters):
    """ This function sorts points on a fault line based on distances. We assume at first that the points on the fault
    belong to one fault line only. We furthermore assume that a fault line is continuous. It may contain kinks, but at
    these, the angle must not exceed acos(fault_approx_pars.cos_angle_max). If we start with the first point on the
    line, the next one is the nearest to the first, unless it is included in the set of already sorted points or if
    the angle between the current line segment and the one with the new point exceeds
    acos(fault_approx_pars.cos_angle_max). Then, we proceed with the second-nearest point, and so on. If this fails
    for all k_nearest_neighbours nearest points, we stop sorting. However, there will be leftover points. We restart the
    algorithm with these leftover points. If there are no leftover points anymore, we are finished.
    Leftover points may occur either if we do not start with first or last points on the fault or if the fault consists
    of several components (this may happen if subdomains are not simply connected). Finding the correct start or end
    point is hard, and we use a heuristic based on the distance to the domain boundary.
    This way, we end up with a bunch of ordered subsets. Ideally, this corresponds to the fact that the fault line
    consists of several components, one subset for each component.
    In the last step, we try to combine these subsets based (again) on distances, if bforceCombine is set to true.
    We made this last step optional, as it may happen that a fault consists of several components. Then it does not
    make sense to sew them together.

Args:
    points (npt.ArrayLike): set of points on the fault line wich needs to be ordered
    bforce_combine (bool): It may happen that a fault line consistens of several components. If not, we combine all
    components to one. This is enforced by bforce_combine = true.
    problem_descr (ProblemDescr):
        class containing all parameters relevant the fault approximation problem
    fault_approx_pars (FaultApproxParameters):
        class containing all parameters relevant for the algorithm for fault approximation

    Returns:
        idx_points_surf_ordered
        sort_success
    """

    num_points, ndim = points.shape

    idx_points_surf_ordered = []

    # initialisation
    itry = 0
    point_found = False
    ncomps = 0

    # if there are two points or even less, there is no need for sorting
    if num_points <= 2:
        idx_points_surf_ordered.append(np.arange(num_points))
        sort_success = True
    else:

        # heuristic number of trials aka the maximal number of sorted subsets. If there are more than about half of
        # the number of points, we must assume that the sorting did not yield any reasonable result.
        ntrials = max(2, floor(num_points * 0.5))

        # number of points per sorted subset
        num_points_per_part = np.zeros(ntrials, dtype=np.int32)

        # compute distance matrix
        dist_mat = compute_dist_mat(points)

        iidx_nearest_neighbour = np.argsort(dist_mat, axis=0)

        idx_aux = np.arange(num_points)

        # as points will be reduced to the number of still unsorted points, we should store the original point set.
        points_orig = np.copy(points)

        for itry in range(num_points):

            # All what follows makes only sense if there is more than one point. If numPoints is one, then there is one
            # orphaned point, and we make it a new segment.
            # Note that num_points is reduced during the number of trials.
            if num_points > 1:

                # Search the points closest to the beginning or end of the fault line. In general, it is impossible to
                # decide which of the point is closest to the beginning/end without knowing the fault line. As
                # heuristic measure, we search the points closest to boundaries of the domain.
                # min_dist_bdry contains the distances to all 4 (in 2D) or 6 (in 3D) lines/planes the domain boundary
                # consists of (even indices of dim for minima, odd indices of dim for maxima).
                min_dist_bdry = np.zeros((2 * ndim, num_points))
                idx_points_surf_ordered.append(np.ones(num_points, dtype=int) * -1)

                for i_dim in range(ndim):
                    ar = points[:, i_dim]
                    min_dist_bdry[2 * i_dim, :] = np.abs(ar - problem_descr.x_min[i_dim])
                    min_dist_bdry[2 * i_dim + 1, :] = np.abs(ar - problem_descr.x_max[i_dim])

                del ar

                # minimal distance of each point to the boundary
                aux_arr2 = np.min(min_dist_bdry, axis=0)

                # index of the boundary part a point is closest to
                min_index = np.argmin(min_dist_bdry, axis=0)

                """
                     _______________________
                    |                       |
                    |                       |
                    |                       |
                    |                       |
                    |                       |
                    |       |               |
                    |       x               |
                    |       |               |
                    |       x--x-x---x      |
                    |_______________________|
                Find out if some points have almost the same distance to the same nearest domain boundary part. In this
                case, we cannot use this distance for finding a reasonable starting point: For the example above, any of
                the four bottom points is nearest, depending on numerical inaccuracies. Therefore, we consider these
                candidates and exclude the minimal distance to the nearest boundary part. Then, we repeat this process.
                
                The loop is necessary, as in 3D, points can be along a line parallel to one of the coordinate axes. In
                this case, we need to exclude 2 planes, as the distance to them of all these points is almost the same
                and therefore meaningless.
                It may even happen, that there are several points with almost the same minimal distance to the boundary,
                but with respect to different lines/planes:
                      _______________________
                     |                       |
                     |                       |
                     |                       |
                     |                       |
                     |                       |
                     |  |                    |
                     |  x                    |
                     |  |                    |
                     |  x--x-x---x           |
                     |_______________________|
                
                This case is however very special and hard to handle: Considering distances, we would end up with the
                corner point, which is not the right one. Moreover, sorting works in most cases even for starting points
                not being the first or last point, so we exclude this special case by
                    size(unique(minIndex(Candidates)),2) == 1.
                That means that for the above case, the starting point is more or less arbitrary. However,
                TestCaseFaultDetection03 shows that even then, sorting works.
                """

                candidates = aux_arr2 < np.min(aux_arr2) + 2 * fault_approx_pars.abstol_bisection

                iskip = 1
                while iskip < ndim and np.unique(min_index[candidates]).shape[0] == 1 and \
                        min_index[candidates].shape[0] > 1:
                    # exclude the distance which is small and almost the same
                    indices = np.concatenate((np.arange(min_index[0]),
                                              np.arange(min_index[0] + 1, 2 * ndim - iskip + 1))).tolist()

                    min_dist_bdry = min_dist_bdry[indices, :]

                    # setting the distance for excluding all points but the candidates to some high value is easier
                    # than index arithmetics, but maybe not so efficient
                    min_dist_bdry[:, ~candidates] = 1e20
                    aux_arr2 = np.min(min_dist_bdry, axis=0)

                    # index of the nearest boundary part
                    min_index = np.argmin(min_dist_bdry, axis=0)

                    candidates = aux_arr2 < np.min(aux_arr2) + 2 * fault_approx_pars.abstol_bisection
                    iskip = iskip + 1

                idx_nearest_to_boundary = np.argsort(aux_arr2, axis=0)
                iidx_point = idx_nearest_to_boundary[0]

                # Ordering points by nearest neighbour search does not work in every case, not even when starting at the
                # beginning or the end of the fault line. However, starting at these points is beneficial, as there are
                # less failure modes. This means that we need a fallback for failed sorting.
                # If the ordering fails for our starting point, we take the second nearest to the boundary and so on.
                idx_points_surf_ordered[itry][0] = iidx_point
                list_of_forbidden_points = np.ones(num_points) * -1
                list_of_forbidden_points[0] = iidx_point

                # start of sorting: take the initial point and its nearest neighbour
                idx_points_surf_ordered[itry][1] = iidx_nearest_neighbour[1, iidx_point]
                list_of_forbidden_points[1] = iidx_nearest_neighbour[1, iidx_point]
                iidx_point = iidx_nearest_neighbour[1, iidx_point]
                num_points_per_part[itry] = 2

                ipoint = 1
                point_found = True
                while ipoint <= num_points - 1 and point_found:

                    # For the current point in the point set, consider the nearest neighbour, if still available,
                    # otherwise the second-nearest neighbour, if still available and so on. If this point passes the
                    # consistency check, we have found the successor of the current point in ordering.
                    # If we tried all k_nearest_neighbours next points and still did not find a successor, we assume
                    # that all points of the same segment are found, and we proceed to the next segment.
                    point_found = False
                    i = 0

                    while i < min(num_points - 1, fault_approx_pars.n_nearest_sort) and not point_found:

                        # i-nearest neighbour is not forbidden
                        if not np.any(list_of_forbidden_points == iidx_nearest_neighbour[i + 1, iidx_point]):
                            # consistency check: compute the angle between the segments: if it exceeds alphaMax, we
                            #  reject the nearest available neighbour and try the next nearest one.
                            seg1 = points[iidx_point, :] - points[
                                                              iidx_nearest_neighbour[i + 1, iidx_point], :]
                            seg2 = points[idx_points_surf_ordered[itry][ipoint - 1], :] - points[iidx_point, :]

                            angle = seg1 @ seg2 / (np.linalg.norm(seg1, 2) * np.linalg.norm(seg2, 2))
                            if angle > fault_approx_pars.cos_alpha_max:
                                idx_points_surf_ordered[itry][ipoint + 1] = iidx_nearest_neighbour[i + 1, iidx_point]
                                iidx_point = iidx_nearest_neighbour[i + 1, iidx_point]
                                list_of_forbidden_points[ipoint + 1] = iidx_point
                                point_found = True
                                num_points_per_part[itry] += 1

                        i = i + 1
                    ipoint = ipoint + 1

                # if there are no points to sort left: Stop sorting
                if not np.any(idx_points_surf_ordered[itry] == -1):
                    idx_points_surf_ordered[itry] = idx_aux[idx_points_surf_ordered[itry]]
                    ncomps = itry + 1

                    # we set point_found = true, even in fact no point has been found (there is no one left), as this
                    # condition is used further below for testing if the search succeeded in case the maximal number of
                    # trials has been exhausted.
                    point_found = True
                    break

                # there are points left to sort: reduce the point set to these leftover points and proceed
                else:

                    # shorten the index array to the correct length
                    idx_points_surf_ordered[itry] = idx_points_surf_ordered[itry][0:num_points_per_part[itry]]

                    # indices with respect to the current point set
                    idx_points_surf_ordered_local = idx_points_surf_ordered[itry]

                    # indices with respect to the original point set
                    idx_points_surf_ordered[itry] = idx_aux[idx_points_surf_ordered[itry]]

                    # idx contains the indices of the points already sorted with
                    # respect to the original point set
                    idx_local = np.setdiff1d(np.arange(num_points), idx_points_surf_ordered_local)

                    # indices of the points already sorted
                    dist_mat = dist_mat[idx_local[:, np.newaxis], idx_local]
                    points = points[idx_local, :]
                    iidx_nearest_neighbour = np.argsort(dist_mat, axis=0)
                    num_points = points.shape[0]
                    idx_aux = idx_aux[idx_local]

            else:
                # one orphaned point left
                idx_points_surf_ordered.append(np.array([idx_aux[0]]))
                num_points_per_part[itry] = 1
                ncomps = itry + 1
                break

        # There are still points which could not be included into the set of sorted points, and we have used all our
        # trials: we failed.
        if itry == ntrials and not point_found:
            sort_success = False

        # if not: continue with the algorithm. Sew together all sorted subsets, if desired
        else:
            # sewing makes sense only if there is more than one ordered subset
            if bforce_combine and ncomps > 1:
                idx_points_surf_ordered = __force_combine(ncomps, points_orig, idx_points_surf_ordered,
                                                          num_points_per_part)

            sort_success = True

        # it may happen that the polygonal line after search intersects, but only if the fault in fact consists of
        # several components. Then, setting bforce_combine = true sews components together in error which do not belong
        # together. However, if combining the subsets is not forced, this must not happen.
        if not bforce_combine:
            for iseg in range(ncomps):
                self_intersects = self_intersection(points_orig[idx_points_surf_ordered[iseg], :])
                if self_intersects:
                    warnings.warn('Sorting leads to a self-intersecting polygonal line.')

        if not sort_success:
            warnings.warn("Sorting of points on fault line failed.")

    return idx_points_surf_ordered, sort_success


def __force_combine(nsegs: int, points_orig: npt.ArrayLike, idx_points_surf_ordered: npt.ArrayLike,
                    num_points_per_part: npt.ArrayLike) -> npt.ArrayLike:
    """SUMMARY

    Args:
        idx_points_surf_ordered ([type]): [description]
        num_points_per_part ([type]): [description]

    Returns:
        idx_points_surf_ordered

    """

    # index of the nearest component
    iseg_min = 0

    # distance between the starting points of the different components
    norm_start_start = np.zeros(nsegs)

    norm_start_end = np.zeros(nsegs)
    norm_end_start = np.zeros(nsegs)

    # distance between the end points of the different components
    norm_end_end = np.zeros(nsegs)

    for isub in range(nsegs - 1):
        aux_old = 1e20

        # start and end points of the isub-th subset
        xstart = points_orig[idx_points_surf_ordered[0][0], :]
        xend = points_orig[idx_points_surf_ordered[0][num_points_per_part[0] - 1], :]

        for jsub in range(1, nsegs - isub):
            # find the segment which is closest
            start_next = points_orig[idx_points_surf_ordered[jsub][0], :]
            end_next = points_orig[idx_points_surf_ordered[jsub][num_points_per_part[jsub] - 1], :]
            norm_start_start[jsub] = np.linalg.norm((xstart - start_next))
            norm_start_end[jsub] = np.linalg.norm((xstart - end_next))
            norm_end_start[jsub] = np.linalg.norm((xend - start_next))
            norm_end_end[jsub] = np.linalg.norm((end_next - xend))

            aux = np.min(
                np.asarray(
                    [norm_start_start[jsub], norm_start_end[jsub], norm_end_start[jsub],
                     norm_end_end[jsub]]))
            if aux < aux_old:
                aux_old = aux
                iseg_min = jsub
        # sew the two subsets together

        merge_success = False

        # the two starting points are closest
        if norm_start_start[iseg_min] <= norm_start_end[iseg_min] and \
                norm_start_start[iseg_min] <= norm_end_start[iseg_min] and \
                norm_start_start[iseg_min] <= norm_end_end[iseg_min]:

            idx_points_surf_ordered[0] = np.concatenate(
                (np.flip(idx_points_surf_ordered[0]), idx_points_surf_ordered[iseg_min]))
            merge_success = True

        # the starting point of the first and the end points of the second subset are closest
        elif norm_start_end[iseg_min] <= norm_start_start[iseg_min] and \
                norm_start_end[iseg_min] <= norm_end_start[iseg_min] and \
                norm_start_end[iseg_min] <= norm_end_end[iseg_min]:

            idx_points_surf_ordered[0] = np.concatenate(
                (np.flip(idx_points_surf_ordered[0]), np.flip(idx_points_surf_ordered[iseg_min])))
            merge_success = True

        elif norm_end_start[iseg_min] <= norm_start_start[iseg_min] and \
                norm_end_start[iseg_min] <= norm_start_end[iseg_min] and \
                norm_end_start[iseg_min] <= norm_end_end[iseg_min]:

            idx_points_surf_ordered[0] = np.concatenate(
                (idx_points_surf_ordered[0], idx_points_surf_ordered[iseg_min]))
            merge_success = True

        elif norm_end_end[iseg_min] <= norm_start_start[iseg_min] and \
                norm_end_end[iseg_min] <= norm_start_end[iseg_min] and \
                norm_end_end[iseg_min] < norm_end_start[iseg_min]:

            idx_points_surf_ordered[0] = np.concatenate(
                (idx_points_surf_ordered[0], np.flip(idx_points_surf_ordered[iseg_min])))
            merge_success = True
        else:
            warnings.warn('Merging of parts failed in sorting')

        # delete the segment just attached and continue
        # combine all subsets in the first one
        if merge_success:
            num_points_per_part[0] = num_points_per_part[0] + num_points_per_part[iseg_min]
            for i in range(iseg_min, nsegs - 1):
                idx_points_surf_ordered[i] = idx_points_surf_ordered[i + 1]
                num_points_per_part[i] = num_points_per_part[i + 1]

    return idx_points_surf_ordered


def main():
    points = np.array([[0.07, 0.04],
                       [0.06, 0.05],
                       [0.08, 0.03],
                       [0.05, 0.06],
                       [0.03, 0.08]])
    bforce = True
    fault_parameters = FaultApproxParameters(eps=1.e-10, err_max=0.002, err_min=0.0001, num_points_local=10)
    problem = ProblemDescr(x_min=np.array([0, 0]), x_max=np.array([1, 1]))
    idx_sorted, sort_success = sort_on_fault(points, bforce_combine=bforce, problem_descr=problem,
                                             fault_approx_pars=fault_parameters)
    print(idx_sorted, sort_success)


if __name__ == '__main__':
    main()
