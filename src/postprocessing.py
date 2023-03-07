"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
import numpy as np
import sys
import numpy.typing as npt
from typing import List, Dict, Tuple
from entities.Entities import FaultApproxParameters, ProblemDescr
from tools.computeIniSet import halton_points
from inpoly import inpoly2
from warnings import warn

import pygmsh
from meshio import CellBlock
from datetime import datetime


def __comp_par_values(edge_end: int, coord_last: npt.ArrayLike) -> npt.ArrayLike:
    """
    Auxiliary function for reconstruct_subdomains. It computes the parameter value of a boundary point in 2D. The
    parametrisation of the rectangle starts and ends in the lower left corner (values 0 and 4).
    Args:
        edge_end (int):
            edge index (1: bottom, 2: right, 3: top, 4: left)
        coord_last (npt.ArrayLike)

    Returns:
        npt.ArrayLike
    """
    if edge_end == 1:
        return coord_last.reshape(-1)
    elif edge_end == 2:
        return 1 + coord_last.reshape(-1)
    elif edge_end == 3:
        return 3 - coord_last.reshape(-1)
    elif edge_end == 4:
        return 4 - coord_last.reshape(-1)
    else:
        return np.array([0])


def reconstruct_subdomains(point_sets_surface: npt.ArrayLike, left_domain_start: npt.ArrayLike,
                           left_domain_end: npt.ArrayLike, problem_descr: ProblemDescr, class_vals: npt.ArrayLike,
                           fault_approx_pars: FaultApproxParameters) -> Tuple[npt.ArrayLike, bool]:
    """The purpose of this function is to combine components of fault lines to polygons which approximate the subdomains
    associated with classification.

    This function uses inpoly2 from inpoly (https://github.com/dengwirda/inpoly-python)

    Args:
        point_sets_surface (npt.ArrayLike):
            Data structure containing the points near the fault lines. PointSetsSurface{iclass, jclass} points to a
            structure of arrays (as many as the segments the fault line consists of) containing the coordinates of the
            points near the fault line between classes iclass and jclass which are themselves belong to iclass.
            Shape: (nclasses, nclasses)
        left_domain_start (npt.ArrayLike):
            Data structure containing the index of the domain boundary from which a segment of a fault line starts (0,
            if it starts in the interior). Indices of the domain edges:
                      3
               _______________
              |               |
              |               |
              |               |
            4 |               | 2
              |               |
              |               |
              |_______________|
                      1

            Shape: (nclasses, nclasses)
        left_domain_end (npt.ArrayLike):
            same as LeftDomainStart, but referring to the domain boundary where a fault line segment ends.
            Shape: (nclasses, nclasses)
        problem_descr (ProblemDescr):
            structure containing all problem-relevant parameters. We refer to its documentation for details.
        class_vals (npt.ArrayLike):
            Array containing the class values. These are not necessarily the class indices. Imagine that
            f(\Omega) = {1,2,5}. Then, class_vals = [1,2,5], whereas the class indices range from 0 to 2.
            Shape: (nclasses,)
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.

    Returns:
    npt.ArrayLike: Structure polygonal approximation to the components of the subdomains associated to the several
        classes
    bool: SUMMARY
    """
    reconstr_succeeded = False

    # number of occurring classes
    num_classes = point_sets_surface.shape[0]

    # find out dimension
    for i_class in range(num_classes - 1):
        for j_class in range(i_class + 1, num_classes):
            point_set = point_sets_surface[i_class, j_class]
            if not point_set.size == 0:
                dim = point_set[0].shape[1]
                break

    max_class_idx = np.max(class_vals)

    # create subdomains-structure
    subdomains = np.empty(max_class_idx, dtype=object)

    if dim == 2:
        # maximal number of trials for completing a subdomain
        max_trials = 100

        point_sets_surface_aux = np.empty((num_classes, num_classes), dtype=object)

        left_domain_start_aux = np.ones(left_domain_start.shape, dtype=int) * -1
        left_domain_end_aux = np.ones(left_domain_end.shape, dtype=int) * -1

        # The arrays point_sets_surface[i_class, j_class] and point_sets_surface[j_class, i_class] both contain points
        # near the fault line between the classes i_class and j_class. The points of the former one belong to i_class,
        # the points of the latter one to j_class. While this is worthwhile for computing approximations of the segments
        # of a fault line, it is not for providing a consistent subdivision of the domain by polygons which we intend
        # here. Therefore, we take the mean of these pairs of points.
        for i_class in range(num_classes):
            for j_class in range(i_class):
                num_segs = point_sets_surface[i_class, j_class].shape[0]
                point_sets_surface_aux[i_class, j_class] = np.empty(num_segs, dtype=object)
                for i_seg in range(num_segs):
                    point_sets_surface_aux[i_class, j_class][i_seg] = 0.5 * (
                            point_sets_surface[i_class, j_class][i_seg] +
                            point_sets_surface[j_class, i_class][i_seg][::-1])
                    left_domain_start_aux[i_class, j_class, i_seg] = left_domain_end[j_class, i_class, i_seg]
                    left_domain_end_aux[i_class, j_class, i_seg] = left_domain_start[j_class, i_class, i_seg]

        for i_class in range(num_classes - 1):
            for j_class in range(i_class + 1, num_classes):
                num_segs = point_sets_surface[i_class, j_class].shape[0]
                point_sets_surface_aux[i_class, j_class] = np.empty(num_segs, dtype=object)
                for i_seg in range(num_segs):
                    point_sets_surface_aux[i_class, j_class][i_seg] = point_sets_surface_aux[j_class, i_class][i_seg][
                                                                      ::-1]
                    left_domain_start_aux[i_class, j_class, i_seg] = left_domain_start[i_class, j_class, i_seg]
                    left_domain_end_aux[i_class, j_class, i_seg] = left_domain_end[i_class, j_class, i_seg]

        # Consistency check: if a segment of a subdomain boundary starts inside the domain but no other segment related
        # to this subdomain end inside the domain, at least one boundary segment must be missing.
        for i_class in range(num_classes):

            # There are segments belonging to i_class ending inside the domain.
            i_end = left_domain_end_aux[i_class]
            i_start = left_domain_start_aux[i_class]

            # Something ends inside but nothing starts inside or something starts inside but nothing ends inside.
            if np.any(i_end == 0) and not np.any(i_start == 0) or not np.any(i_end == 0) and np.any(i_start == 0):
                warn(f"Some components of the boundary of class {class_vals[i_class]} are missing.")
                warn("Reconstruction of subdomains failed. Consider taking a finer initial point set.")
                return subdomains, reconstr_succeeded

        # Compute the maximal number of components per fault line.
        max_num_segs = 0
        for i_class in range(num_classes):
            for j_class in range(i_class + 1, num_classes):
                max_num_segs = np.maximum(max_num_segs, point_sets_surface[i_class, j_class].shape[0])
            subdomains[class_vals[i_class] - 1] = np.empty(max_num_segs, dtype=object)

        par_vals_start = -1 * np.ones((num_classes, num_classes, max_num_segs))
        par_vals_end = -1 * np.ones((num_classes, num_classes, max_num_segs))

        # Allocate and fill num_points_surf with the number of points per component. As the points in [i_class, j_class]
        # are the same as in [j_class, i_class] by construction, we exploit symmetry.
        num_points_surf = np.zeros((point_sets_surface_aux.shape[0], point_sets_surface.shape[1], max_num_segs),
                                   dtype=int)
        for i_class in range(num_classes - 1):
            for j_class in range(i_class + 1, num_classes):
                for i_seg in range(point_sets_surface_aux[i_class, j_class].shape[0]):
                    num_points_surf[i_class, j_class, i_seg] = point_sets_surface_aux[i_class, j_class][i_seg].shape[0]
                    num_points_surf[j_class, i_class, i_seg] = point_sets_surface_aux[j_class, i_class][i_seg].shape[0]

        #    direction of parametrisation of the global domain
        #                <-----------------
        #   |-----------------------------------------|
        #   |            <-----               <-----  |
        #   |       par subdomain I           ---->   |
        #   |                           _   /---------| X
        #   |                           /| /  <----   |
        #   |                          /  / par II    |
        #   |                         /  /         /\ |    /\
        #   |              I            |          |  |    |
        #   |                           |    II,1  |  |    |
        #   |                            \            |    |
        #   |                              \   ---->  |    |
        #   |                                \________| Y
        #   |                                         |
        #   |                                   /-----| Z
        #   |                                  /      |
        #   |                                 / II,2  |
        #   |       ----->                   /________|
        #   |_________________________________________|
        #
        #
        # The surface point sets for [i_class, j_class] are ordered such that subdomain i_class is right when running
        # along the fault line. This, however is not sufficient for assembling the subdomains, if some of them consist
        # of several components: We start assembling subdomain I by starting in X and end with this line segment in Y.
        # Now, we need to continue with the second segment of the fault line starting in Z. To do so, we search for
        # fault line components starting at the right domain boundary and find the one starting in Z (if there are more
        # than one starting at the right domain boundary, we take the closest one). Realising that there are no more
        # components, we close the subdomain and are finished.
        # Now, we assemble both components of subdomain II. We may start in Y and end in X. In contrast to above, we
        # must not consider any additional boundary segments but instead stop and close that first component of
        # subdomain II. Ths same holds for component II,2.
        #
        # The main difference between these two cases is that for subdomain II, the parameter value of the starting
        # point is lower than the parameter value of the end point, both being boundary points. This is why we need
        # parameter values for start and end points of boundary segments if they are boundary points.
        for i_class in range(num_classes):
            for j_class in range(num_classes):
                num_segs = 0
                if point_sets_surface_aux[i_class, j_class] is not None:
                    num_segs = point_sets_surface_aux[i_class, j_class].shape[0]
                for i_seg in range(num_segs):

                    start = left_domain_start_aux[i_class, j_class, i_seg]
                    if start == 1:
                        aux = 1 / (problem_descr.x_max[0] - problem_descr.x_min[0])
                        par_vals_start[i_class, j_class, i_seg] = (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                       0, 0] - problem_descr.x_min[0]) * aux
                    elif start == 2:
                        aux = 1 / (problem_descr.x_max[1] - problem_descr.x_min[1])
                        par_vals_start[i_class, j_class, i_seg] = 1 + (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                           0, 1] - problem_descr.x_min[1]) * aux
                    elif start == 3:
                        aux = 1 / (problem_descr.x_max[0] - problem_descr.x_min[0])
                        par_vals_start[i_class, j_class, i_seg] = 3 - (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                           0, 0] - problem_descr.x_min[0]) * aux
                    elif start == 4:
                        aux = 1 / (problem_descr.x_max[1] - problem_descr.x_min[1])
                        par_vals_start[i_class, j_class, i_seg] = 4 - (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                           0, 1] - problem_descr.x_min[1]) * aux

                    end = left_domain_end_aux[i_class, j_class, i_seg]
                    if end == 1:
                        aux = 1.0 / (problem_descr.x_max[0] - problem_descr.x_min[0])
                        par_vals_end[i_class, j_class, i_seg] = (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                     num_points_surf[i_class, j_class, i_seg] - 1, 0] -
                                                                 problem_descr.x_min[0]) * aux
                    elif end == 2:
                        aux = 1.0 / (problem_descr.x_max[1] - problem_descr.x_min[1])
                        par_vals_end[i_class, j_class, i_seg] = 1.0 + (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                           num_points_surf[
                                                                               i_class, j_class, i_seg] - 1, 1] -
                                                                       problem_descr.x_min[1]) * aux
                    elif end == 3:
                        aux = 1.0 / (problem_descr.x_max[0] - problem_descr.x_min[0])
                        par_vals_end[i_class, j_class, i_seg] = 3.0 - (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                           num_points_surf[
                                                                               i_class, j_class, i_seg] - 1, 0] -
                                                                       problem_descr.x_min[0]) * aux
                    elif end == 4:
                        aux = 1.0 / (problem_descr.x_max[1] - problem_descr.x_min[1])
                        par_vals_end[i_class, j_class, i_seg] = 4.0 - (point_sets_surface_aux[i_class, j_class][i_seg][
                                                                           num_points_surf[
                                                                               i_class, j_class, i_seg] - 1, 1] -
                                                                       problem_descr.x_min[1]) * aux
        corners_next = np.zeros((4, 2), dtype=int)
        corners_next[0, :] = np.array([problem_descr.x_max[0], problem_descr.x_min[1]])
        corners_next[1, :] = np.array([problem_descr.x_max[0], problem_descr.x_max[1]])
        corners_next[2, :] = np.array([problem_descr.x_min[0], problem_descr.x_max[1]])
        corners_next[3, :] = np.array([problem_descr.x_min[0], problem_descr.x_min[1]])

        # Combine the components of the fault lines to polygons which approximate the components of the subdomain
        # associated to the classes.
        for i_class in range(num_classes):

            # We do not know how many components the subdomain associated with i-class consists of, but at most as many
            # as the fault lines consist of.
            for i_seg in range(max_num_segs):

                # true, if all components of a subdomain are found
                finished_sub = False

                # true, if all segments of a component are found
                finished_part = False

                initial = True

                par_start = -1

                # If there are any segments of a fault line associated with i_class (we abuse num_points_surf to
                # indicate that).
                if np.any(np.any(num_points_surf[i_class, :, :] > 0)):
                    # Find the indices of the classes on the other side j_class and the segment index j_seg of all these
                    # fault line segments (maybe several ones).
                    aux_arr = np.squeeze(num_points_surf[i_class, :, :] > 0)
                    if len(aux_arr.shape) == 1 or aux_arr.shape[0] == 1:
                        aux_arr = aux_arr.reshape(-1, 1)
                    j_classes, j_segs = np.where(aux_arr)
                    j_seg = np.min(j_segs)
                    j_class = np.min(j_classes[np.argmin(j_segs)])
                else:
                    warn(f"There are no known points on the boundary of class {class_vals[i_class]}")
                    return subdomains, reconstr_succeeded

                # to avoid that this part is taken once more later on
                left_domain_start_aux_init = left_domain_start_aux[i_class, j_class, j_seg]
                i_try = 0
                while not finished_part:
                    i_try += 1
                    if i_try >= max_trials:
                        warn(f"Reconstruction of subdomain for class {class_vals[i_class]} failed.")
                        return subdomains, reconstr_succeeded

                    if j_class >= 0:

                        # It is not entirely clear why this error case should occur, but it does.
                        if subdomains[class_vals[i_class] - 1].shape[0] >= i_seg + 1:
                            if subdomains[class_vals[i_class] - 1][i_seg] is None:
                                subdomains[class_vals[i_class] - 1][i_seg] = point_sets_surface_aux[i_class, j_class][
                                    j_seg].astype(np.float64)
                            else:
                                subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                                    subdomains[class_vals[i_class] - 1][i_seg],
                                    point_sets_surface_aux[i_class, j_class][j_seg])).astype(np.float64)
                        else:
                            warn(f"Reconstruction of subdomain for class {class_vals[i_class]} failed.")
                            return subdomains, reconstr_succeeded
                        if initial:
                            edge_ini = left_domain_start_aux[i_class, j_class, j_seg]
                            initial = False
                            par_start = par_vals_start[i_class, j_class, j_seg]
                        # todo <LH> -42 ändern?
                        left_domain_start_aux[i_class, j_class, j_seg] = -42
                        edge_end = int(left_domain_end_aux[i_class, j_class, j_seg])
                        num_points_surf[i_class, j_class, j_seg] = 0

                    # If there are no appropriate boundary parts at all, stop reconstructing this subdomain and just
                    # close it.
                    if not np.any(num_points_surf[i_class]):
                        finished_sub = True
                        finished_part = True

                    # There are still unassigned boundary components belonging to the current subdomain.
                    else:
                        j_class = -1

                        # The last assigned boundary part ends on the domain boundary part with index edge_end.
                        if edge_end > 0:

                            # Test if any other still unassigned boundary part starts on the current domain boundary.
                            i_idx = np.squeeze(left_domain_start_aux[i_class] == edge_end)
                            if len(i_idx.shape) == 1 or i_idx.shape[0] == 1:
                                i_idx = i_idx.reshape(-1, 1)

                            # Get x- or y-coordinate from the end point of the current boundary part (the one which ends
                            # on the current edge). This point is the last in the array of boundary points.
                            coord_last = np.array([subdomains[class_vals[i_class] - 1][i_seg][-1, (edge_end + 1) % 2]])
                            par_last = __comp_par_values(edge_end, coord_last)

                            # There might be several boundary parts of that kind. Take the "nearest one" aka the one
                            # with the lowest starting parameter but still larger than the present end parameter.
                            # If there is no such boundary part, take the appropriate part of the domain boundary. Test
                            # before if the component of the subdomain is closed yet.
                            if np.any(i_idx[:]):

                                # Get the indices of the corresponding boundary.
                                i, j = np.unravel_index(np.flatnonzero(i_idx == 1), i_idx.shape)
                                i_idx_2 = np.zeros((i.shape[0], 2), dtype=int)
                                i_idx_2[:, 0] = i
                                i_idx_2[:, 1] = j

                                aux_vals = np.zeros((i_idx_2.shape[0], 1))
                                # Find the nearest part.
                                for i in range(i_idx_2.shape[0]):
                                    aux_vals[i] = point_sets_surface_aux[i_class, i_idx_2[i, 0]][i_idx_2[i, 1]][
                                        0, (edge_end + 1) % 2]

                                aux_pars = __comp_par_values(edge_end, aux_vals)

                                if edge_end == edge_ini and par_start > par_last:
                                    i_idx_2 = i_idx_2[
                                        (aux_pars > np.minimum(par_last, par_start)) &
                                        (aux_pars < np.maximum(par_start, par_last))]
                                    aux_pars = aux_pars[
                                        (aux_pars > np.minimum(par_last, par_start)) &
                                        (aux_pars < np.maximum(par_start, par_last))]
                                else:
                                    i_idx_2 = i_idx_2[aux_pars > par_last]
                                    aux_pars = aux_pars[aux_pars > par_last]

                                # Here, j_class is the index in i_idx2, not actually j_class.
                                j_class = np.argmin(aux_pars) if aux_pars.size != 0 else None

                                # If j_class is not empty, this means that there are fault line components starting on
                                # the current domain boundary. If so, we take the nearest one. If not, we proceed to the
                                # next domain boundary.
                                if j_class is not None:
                                    j_seg = i_idx_2[j_class, 1]
                                    j_class = i_idx_2[j_class, 0]
                                else:
                                    j_class = -1

                            # If there are no other boundary parts leaving the domain at the current domain boundary, go
                            # to the next corner point, if appropriate.
                            if j_class < 0:
                                if edge_ini == edge_end and par_start > par_last:
                                    finished_part = True
                                else:
                                    subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate(
                                        (subdomains[class_vals[i_class] - 1][i_seg],
                                         corners_next[edge_end - 1][np.newaxis]), axis=0)
                                    # No further boundary components on current domain boundary: proceed to next edge.
                                    edge_end = int(edge_end) % 4 + 1

                        # The current component ends inside the domain.
                        else:

                            # Search all other boundary parts which start inside.
                            i_idx = (left_domain_start_aux[i_class, :, :] == 0) & (num_points_surf[i_class, :, :] > 0)
                            i_idx = np.squeeze(i_idx)

                            if len(i_idx.shape) == 1 or i_idx.shape[0] == 1:
                                i_idx = i_idx.reshape(-1, 1)

                            if np.any(i_idx):
                                # Get the indices of the corresponding boundary parts.
                                i, j = np.unravel_index(np.flatnonzero(i_idx == 1), i_idx.shape)
                                i_idx_2 = np.zeros((i.shape[0], 2), dtype=int)
                                i_idx_2[:, 0] = i
                                i_idx_2[:, 1] = j

                                aux_vals = np.zeros((i_idx_2.shape[0], 2))
                                # Collect all starting points and compute their distance to the end point of the current
                                # part.
                                for i in range(i_idx_2.shape[0]):
                                    aux_vals[i] = point_sets_surface_aux[i_class, i_idx_2[i, 0]][i_idx_2[i, 1]][0]
                                aux_vals = aux_vals - subdomains[class_vals[i_class] - 1][i_seg][-1]
                                aux_vals = np.power(aux_vals, 2)
                                aux_vals[:, 0] = np.sum(aux_vals, axis=1)

                                min_dist, j_class = np.min(aux_vals[:, 0:1]), np.argmin(aux_vals[:, 0:1])

                                min_dist = np.sqrt(min_dist)

                                # If there are candidates which are reasonably close, continue with them.
                                if min_dist < fault_approx_pars.max_dist_for_surface_points:
                                    # up to here, j_class is the index in i_idx_2, not actually j_class
                                    j_seg = i_idx_2[j_class, 1]
                                    j_class = i_idx_2[j_class, 0]
                                else:
                                    finished_part = True

                            else:
                                if not np.any(num_points_surf[i_class]):
                                    finished_sub = True
                                    finished_part = True
                                else:
                                    warn('Contradicting information about subdomains')
                                    reconstr_succeeded = False
                                    return subdomains, reconstr_succeeded

                # There are no boundary segments left: close polygon.

                # We start at the left domain boundary.
                if left_domain_start_aux_init == 4:

                    # We end at the bottom.
                    if edge_end == 1:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_max[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])), axis=0)

                    # We end at the right.
                    elif edge_end == 2:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_max[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])), axis=0)

                    # We end at the top.
                    elif edge_end == 3:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_min[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])), axis=0)

                    # We end at the left.
                    elif edge_end == 4:
                        num_points = subdomains[class_vals[i_class] - 1][i_seg].shape[0]
                        if subdomains[class_vals[i_class] - 1][i_seg][0, 1] > \
                                subdomains[class_vals[i_class] - 1][i_seg][num_points - 1, 1]:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate(
                                (subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[problem_descr.x_min[0], problem_descr.x_min[1]],
                                     [problem_descr.x_max[0], problem_descr.x_min[1]],
                                     [problem_descr.x_max[0], problem_descr.x_max[1]],
                                     [problem_descr.x_min[0], problem_descr.x_max[1]],
                                     [problem_descr.x_min[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])),
                                axis=0)
                        else:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                                subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[problem_descr.x_min[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])),
                                axis=0)

                # We start at the right domain boundary.
                elif left_domain_start_aux_init == 2:

                    # We end at the bottom.
                    if edge_end == 1:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_max[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])), axis=0)

                    # We end at the right.
                    elif edge_end == 2:
                        num_points = subdomains[class_vals[i_class] - 1][i_seg].shape[0]
                        if subdomains[class_vals[i_class] - 1][i_seg][0, 1] < \
                                subdomains[class_vals[i_class] - 1][i_seg][num_points - 1, 1]:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate(
                                (subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[problem_descr.x_max[0], problem_descr.x_min[1]],
                                     [problem_descr.x_min[0], problem_descr.x_min[1]],
                                     [problem_descr.x_min[0], problem_descr.x_max[1]],
                                     [problem_descr.x_max[0], problem_descr.x_max[1]],
                                     [problem_descr.x_max[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])),
                                axis=0)
                        else:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                                subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[problem_descr.x_max[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])),
                                axis=0)

                    # We end at the top.
                    elif edge_end == 3:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_min[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])), axis=0)

                    # We end at the left.
                    elif edge_end == 4:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_min[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], subdomains[class_vals[i_class] - 1][i_seg][0, 1]]])), axis=0)

                # We start at the bottom domain boundary.
                elif left_domain_start_aux_init == 1:

                    # We end at the bottom.
                    if edge_end == 1:
                        num_points = subdomains[class_vals[i_class] - 1][i_seg].shape[0]
                        if subdomains[class_vals[i_class] - 1][i_seg][0, 0] < \
                                subdomains[class_vals[i_class] - 1][i_seg][num_points - 1, 0]:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate(
                                (subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[problem_descr.x_max[0], problem_descr.x_min[1]],
                                     [problem_descr.x_max[0], problem_descr.x_max[1]],
                                     [problem_descr.x_min[0], problem_descr.x_max[1]],
                                     [problem_descr.x_min[0], problem_descr.x_min[1]],
                                     [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_min[1]]])),
                                axis=0)
                        else:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                                subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_min[1]]])),
                                axis=0)

                    # We end at the right.
                    elif edge_end == 2:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_max[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], problem_descr.x_min[1]],
                                 [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_min[1]]])), axis=0)

                    # We end at the top.
                    elif edge_end == 3:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_min[0], problem_descr.x_max[1]],
                                 [problem_descr.x_min[0], problem_descr.x_min[1]],
                                 [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_min[1]]])), axis=0)

                    # We end at the left.
                    elif edge_end == 4:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_min[0], problem_descr.x_min[1]],
                                 [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_min[1]]])), axis=0)

                # We start at the top domain boundary.
                elif left_domain_start_aux_init == 3:
                    # we end at the bottom
                    if edge_end == 1:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_max[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], problem_descr.x_max[1]],
                                 [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_max[1]]])), axis=0)

                    # We end at the right.
                    elif edge_end == 2:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_max[0], problem_descr.x_max[1]],
                                 [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_max[1]]])), axis=0)

                    # We end at the top.
                    elif edge_end == 3:
                        num_points = subdomains[class_vals[i_class] - 1][i_seg].shape[0]
                        if subdomains[class_vals[i_class] - 1][i_seg][0, 0] > \
                                subdomains[class_vals[i_class] - 1][i_seg][
                                    num_points - 1, 0]:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate(
                                (subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[problem_descr.x_min[0], problem_descr.x_max[1]],
                                     [problem_descr.x_min[0], problem_descr.x_min[1]],
                                     [problem_descr.x_max[0], problem_descr.x_min[1]],
                                     [problem_descr.x_max[0], problem_descr.x_max[1]],
                                     [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_max[1]]])),
                                axis=0)
                        else:
                            subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                                subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                    [[subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_max[1]]])),
                                axis=0)

                    # We end at the left.
                    elif edge_end == 4:
                        subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                            subdomains[class_vals[i_class] - 1][i_seg], np.array(
                                [[problem_descr.x_min[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], problem_descr.x_min[1]],
                                 [problem_descr.x_max[0], problem_descr.x_max[1]],
                                 [subdomains[class_vals[i_class] - 1][i_seg][0, 0], problem_descr.x_max[1]]])), axis=0)

                # We start somewhere in the middle.
                elif left_domain_start_aux_init == 0:
                    subdomains[class_vals[i_class] - 1][i_seg] = np.concatenate((
                        subdomains[class_vals[i_class] - 1][i_seg],
                        np.array([subdomains[class_vals[i_class] - 1][i_seg][0, :]])), axis=0)
                if finished_sub:
                    n_parts = i_seg + 1
                    subdomains[class_vals[i_class] - 1] = subdomains[class_vals[i_class] - 1][0:n_parts]
                    break

            if np.any(num_points_surf[i_class, :] > 0):
                warn(f'There are leftover boundary parts from subdomain {class_vals[i_class]}.'
                     f'Reconstruction of subdomains failed.')
                reconstr_succeeded = False
                return subdomains, reconstr_succeeded

        # Last part: Is class i_class inside or outside the polygon? We know that the segments of a polygon are ordered
        # such that class i_class is left of the segment. If not inside, flip the subdomain aka taking its complement.
        # Of course, this is correct only if the current subdomain is the only one.
        # However, testing for the need of flipping is necessarily heuristic. We choose the point to test with as
        # follows: For two consecutive boundary points, we take the mean and add to that a scaled (intended) inner
        # normal vector. If this point is in fact inside the domain, class i_class is inside.
        # However, testing like this with one point near the boundary and testing with it, it may happen that the point
        # near the boundary which is intended to be inside the domain is in fact outside:
        #
        #              testing point  O
        #                             /\ scaled inner normal vector
        #                             |
        #                           __|____________---------------x
        #  ___________--------------  |                            |
        # x---------------------------o---------------------------x
        # This failure mode is quite rare and does occur merely in the presence of cusps. Making the normal vector too
        # small to prevent this failure mode may lead to numerical instabilities. To minimize the chance for failure, we
        # consider two points instead of one. In the case of contradicting results (one indicates to flip, the other
        # not to flip), we consider a third point and decide by majority.
        # Note that we test with respect to the discrete approximations of a class; therefore, no additional function
        # calls are necessary, only some point-in-polygon-tests.
        for i_class in range(num_classes):
            for i_part in range(subdomains[class_vals[i_class] - 1].shape[0]):
                idx_1 = 0
                # number of point per subdomain
                idx_2 = subdomains[class_vals[i_class] - 1][i_part].shape[0]
                idx_2 = int(np.minimum(idx_2 - 1, np.ceil(0.5 * idx_2)) - 1)
                aux_vec = subdomains[class_vals[i_class] - 1][i_part][np.array([idx_1, idx_2]), :] - \
                    subdomains[class_vals[i_class] - 1][i_part][np.array([idx_1 + 1, idx_2 + 1]), :]
                if aux_vec.ndim == 1:
                    aux_vec = aux_vec[np.newaxis, :]
                test_point = subdomains[class_vals[i_class] - 1][i_part][np.array([idx_1, idx_2]), :] \
                    - 0.5 * aux_vec - 0.01 * np.concatenate((-aux_vec[:, 1].reshape(-1, 1),
                                                             aux_vec[:, 0].reshape(-1, 1)), axis=1)
                inside, _ = inpoly2(test_point, subdomains[class_vals[i_class] - 1][i_part])
                flip = False

                # Both two points are outside.
                if np.all(~inside):
                    flip = True
                # in case of ambiguity: let a third point decide
                elif np.any(inside) and np.any(~inside):
                    idx_3 = subdomains[class_vals[i_class] - 1][i_part].shape[0] - 1

                    # The following heuristic works if the subdomain consists of more than three points.
                    if idx_3 >= 3:
                        idx_3 = np.minimum(idx_3 - 1, np.ceil(0.75 * (idx_3 + 1)).astype(int) - 1)
                        idx_3_next = idx_3 + 1
                    else:
                        idx_3 = 2
                        idx_3_next = 0
                    aux_vec = subdomains[class_vals[i_class] - 1][i_part][idx_3] - \
                        subdomains[class_vals[i_class] - 1][i_part][idx_3_next]
                    if aux_vec.ndim == 1:
                        aux_vec = aux_vec[np.newaxis, :]
                    test_point = subdomains[class_vals[i_class] - 1][i_part][
                                     idx_3] - 0.5 * aux_vec - 0.05 * np.concatenate(
                        (-aux_vec[:, 1].reshape(-1, 1), aux_vec[:, 0].reshape(-1, 1)), axis=1)
                    inside, _ = inpoly2(test_point, subdomains[class_vals[i_class] - 1][i_part])
                    flip = ~inside
                if flip:
                    subdomains[class_vals[i_class] - 1][i_part] = np.concatenate((np.array(
                        [problem_descr.x_min, [problem_descr.x_max[0], problem_descr.x_min[1]], problem_descr.x_max,
                         [problem_descr.x_min[0], problem_descr.x_max[1]], [np.nan, np.nan]]),
                                                                                  subdomains[class_vals[i_class] - 1][
                                                                                      i_part]))
        # Here, any subdomain must consist of at least three points. Check this.
        for i_class in range(num_classes):
            if subdomains[i_class] is None:
                num_segs = 0
            else:
                num_segs = subdomains[i_class].shape[0]
            for i_seg in range(num_segs):
                if subdomains[i_class][i_seg].shape[0] < 3:
                    reconstr_succeeded = False
                    warn(f"Component {i_seg} of Subdomain for class {class_vals[i_class]} consists of two points only.")
                    return subdomains, reconstr_succeeded
        reconstr_succeeded = True

        # Another consistency check: test, that subdomains do not overlap. As heuristic, we take test points and test
        # if these are included in more than one part of a subdomain.
        test_points = halton_points(200, 2)

        # affine transformation from [0,1]^2 to [Xmin, Xmax]^2
        for i_dim in range(2):
            test_points[:, i_dim] = np.multiply((problem_descr.x_max[i_dim] - problem_descr.x_min[i_dim]),
                                                test_points[:, i_dim]) + problem_descr.x_min[i_dim]

        inside = np.zeros((test_points.shape[0], 1))

        for i_class in range(num_classes):
            for i_part in range(subdomains[class_vals[i_class] - 1].size):
                temp, _ = inpoly2(test_points, subdomains[class_vals[i_class] - 1][i_part])
                inside += temp[:, np.newaxis]
        if np.any(inside > 1):
            warn('Overlapping parts of subdomains detected, reconstruction of subdomains failed.')
            reconstr_succeeded = False
        return subdomains, reconstr_succeeded

    elif dim == 3:
        # evil hack, only for two classes yet
        for i_class in range(num_classes - 1):
            for j_class in range(i_class + 1, num_classes):
                pass
    return subdomains, reconstr_succeeded


def vis_subdomain(subdomains: npt.ArrayLike, class_vals: npt.ArrayLike) -> Tuple[npt.ArrayLike, List[CellBlock],
                                                                                 Dict[str, list]]:
    """
    Args:
        subdomains (npt.ArrayLike): Structure polygonal approximation to the components of the subdomains associated to
            the several classes
        class_vals (npt.ArrayLike):
            Array containing the class values. These are not necessarily the class indices. Imagine that
            f(\Omega) = {1,2,5}. Then, class_vals = [1,2,5], whereas the class indices range from 0 to 2.
            Shape: (nclasses,)

    Returns:
        npt.ArrayLike: points
        List[CellBlock]: cells
        Dict[str,list]: cell_data
    """
    points = None
    cell_data = {"class": []}
    cells = []
    for i_class in range(subdomains.shape[0]):
        if subdomains[i_class] is None:
            continue
        for i_comp in range(subdomains[i_class].shape[0]):
            with pygmsh.occ.Geometry() as geo:

                # In the 2d case: add zeros as third component, since vtu is for visualisation of 3D data.
                if subdomains[i_class][i_comp].shape[1] == 2:
                    zeros = np.zeros((subdomains[i_class][i_comp].shape[0], 1))
                    subdomains[i_class][i_comp] = np.concatenate((subdomains[i_class][i_comp], zeros), axis=1)

                # If the domain components are not simply connected, there are several boundary components.
                # They are separated by NaNs.
                nan_idx = np.flatnonzero(np.isnan(subdomains[i_class][i_comp][:, :1]))
                if nan_idx.size:

                    # Split subdomains into their boundaries.
                    boundaries = [subdomains[i_class][i_comp][:nan_idx[0]]]

                    # Save the maximal x-value of the corresponding boundary.
                    max_x = [np.max(boundaries[0][:, 1])]
                    for i in range(1, nan_idx.size):
                        boundaries.append(subdomains[i_class][i_comp][nan_idx[i - 1] + 1:nan_idx[i]])
                        max_x.append(np.max(boundaries[i][:, 1]))
                    boundaries.append(subdomains[i_class][i_comp][nan_idx[-1] + 1:])
                    max_x.append(np.max(boundaries[-1][:, 1]))

                    # The outer boundary has the maximal x-value of all boundaries.
                    outer_idx = np.argmax(max_x)
                    outer = assure_geo_tolerance(boundaries[outer_idx])
                    outer = geo.add_polygon(outer)
                    boundaries.pop(outer_idx)
                    # the inner boundaries
                    inner = [geo.add_polygon(assure_geo_tolerance(boundary)) for boundary in boundaries]

                    # boolean difference between outer boundary and the union of all inner boundaries
                    if len(inner) > 1:
                        geo.boolean_difference(outer, geo.boolean_union(inner))
                    else:
                        geo.boolean_difference(outer, inner)
                else:
                    geo.add_polygon(assure_geo_tolerance(subdomains[i_class][i_comp]))
                mesh = geo.generate_mesh(dim=3)
                # get cell_data
                for cell in mesh.cells:
                    cell_data["class"] += [[[i_class] for _ in range(len(cell.data))]]
                # get cells
                for cell in mesh.cells:
                    cell.data += len(points) if points is not None else 0
                    cells.append(cell)
                # get points
                if points is None:
                    points = mesh.points
                else:
                    points = np.concatenate((points, mesh.points), axis=0)
    return points, cells, cell_data


def assure_geo_tolerance(points: npt.ArrayLike) -> npt.ArrayLike:
    """
    This function makes sure, that the first and last point satisfy the geometrical tolerance of pygmsh and removes the
    last point if that's not the case.

    Args:
        points (npt.ArrayLike):
            points to adapt

    Returns:
        npt.ArrayLike: adapted points
    """
    eps = 1e-9
    dist_first_to_last = np.abs(points[-1] - points[0])
    if np.all(dist_first_to_last < eps):
        return points[:-1]
    else:
        return points


def export2vtu(points: npt.ArrayLike, triang: npt.ArrayLike, scalar_data: npt.ArrayLike, scalar_data_names: list,
               vec_data: npt.ArrayLike = None, vec_data_names: list = None, filename: str = None) -> None:
    """
    export2vtu exports point wise data to a vtu-file which can be read in by e.g. Paraview. The function works in 2D
    or 3D. In 2D, triangular and quadrilateral meshes are supported, in 3D tetrahedral and hexahedral meshes for volumes
    and triangular meshes for surfaces. The mesh may be unstructured. Mixed meshes are supported.

    Args:
        points (npt.ArrayLike):
            The set of points of the triangulation. n_verts stands for the number of data points and ndim \in {2, 3} for
            the dimension of the data. Shape: (n_verts, ndim)
        triang (npt.ArrayLike):
            represents the triangulation corresponding to points. The indices of the points are 0-based; the last
            column in this array contains the number of points of the current cell. Shape: (n_cells, MaxPointsPerCell+1)
        scalar_data (npt.ArrayLike):
            n_verts stands for the number of data; points and n_scalar_data_sets represents the number of data sets in
            the vtu-file. Shape: (n_verts x n_scalar_data_sets)
        scalar_data_names:
            list containing the names of the data sets which appear in the vtu-file
        filename (str):
            the filename of the vtu-file. Appends the ending .vtu if it's missing.
            Defaults to export2vtu_ with datetime appended.
        vec_data (n_scalar_data_sets x n_verts x 3, optional): data sets for vector-valued quantities.
            Paraview requires 3D vectors even if the vector field is 2D.
            Note: This order of the shape happens due tue the numpy representation. The first shape parameter
            represents the depth
        vec_data_names (optional): name of the optional vector-valued quantities

    Returns:
        None
    """
    np.set_printoptions(threshold=20, edgeitems=10, linewidth=140,
                        formatter=dict(float=lambda x: "%.3g" % x))  # float arrays %.3g
    if filename is None:
        filename = f"export2VTU_{datetime.now().strftime('%d%m%y_%H%M%S')}.vtu"

    # default: no vector data sets
    n_vec_data_sets = 0

    if vec_data is not None and vec_data_names is not None:
        # determine the number of vector data sets to write
        dim_vec_data_set = len(vec_data.shape)
        n_vec_data_sets = 1 if dim_vec_data_set == 2 else vec_data.shape[0]

        if vec_data_names.shape[0] != n_vec_data_sets:
            print(
                f"The number of vec data sets ({n_vec_data_sets}) does not match the number of names for data "
                f"sets.")
            exit(-1)

    if len(triang.shape) == 1:
        triang = triang.reshape(1, -1)
        print("triang is a 1D array. It gets converted into a 2D (n x 1) array to continue")

    if len(scalar_data.shape) == 1:
        scalar_data = scalar_data.reshape(scalar_data.shape[0], -1)
        print("PointData is a 1D array. It gets converted into a 2D (n x 1) array to continue")

    # determine the dimension of data  space
    ndim = points.shape[1]

    # determine the number of point data sets to write
    n_scalar_data_sets = scalar_data.shape[1]

    if len(scalar_data_names) != n_scalar_data_sets:
        print(
            f"The number of point data sets ({n_scalar_data_sets}) does not match the number of names for data sets.")
        exit(-1)

    # number of vertices
    n_verts = len(scalar_data[:, 0])

    # number of cells in the triangulation
    n_cells, aux = triang.shape

    # points per cell in triangulation
    points_per_cell = triang[:, aux - 1]
    max_points_per_cell = aux

    zcomponent = np.zeros((n_verts, 1)) if ndim == 2 else points[:, 2]

    to_write = ""
    # header of the vtu-file
    to_write += f'<?xml version="1.0"?>\n' \
                f'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" ' \
                f'compressor="vtkZLibDataCompressor">\n' \
                f'  <UnstructuredGrid>\n' \
                f'    <Piece NumberOfPoints="{n_verts}" NumberOfCells="{n_cells}">\n'

    # data points
    to_write += f'      <Points>\n' \
                f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'

    for p0, p1, zc in zip(points[:, 0], points[:, 1], zcomponent.reshape(1, -1).ravel()):
        to_write += '          {:.4e} {:.4e} {:.4e}\n'.format(p0, p1, zc)

    to_write += f'        </DataArray>\n' \
                f'      </Points>\n'

    # cell definitions
    to_write += f'      <Cells>\n' \
                f'        <DataArray type="Int32" Name="connectivity" format="ascii">\n'

    polygons = [[] for _ in range(max_points_per_cell)]
    offsets = np.zeros((n_cells, 1))
    poly_types = np.zeros((n_cells, 1))
    iidx_start = 1
    abs_offset = 0

    format_spec_start = '           {:d} {:d} {:d}'
    for i_points_per_cell in range(3, max_points_per_cell):
        # the following two lines are pretty dirty, I know
        polygons[i_points_per_cell] = triang[points_per_cell == i_points_per_cell, :i_points_per_cell]
        n_cells_current_type = polygons[i_points_per_cell].shape[0]
        iidx_end = iidx_start + n_cells_current_type - 1
        if list(range(iidx_start, iidx_end)):
            offsets[iidx_start - 1:iidx_end] = abs_offset + np.array(
                [[i] for i in range(1, n_cells_current_type + 1)]) * i_points_per_cell
            abs_offset = offsets[iidx_end - 1]
        elif iidx_start == iidx_end:
            offsets[iidx_start - 1] = abs_offset + np.array(
                [i + 1 for i in range(n_cells_current_type)]) * i_points_per_cell
            abs_offset = offsets[iidx_start - 1]

        format_spec = format_spec_start + "\n"
        if ndim == 2:
            # VTK_Triangle
            if i_points_per_cell == 3:
                gtype = 5
            # VTK_Line
            elif i_points_per_cell == 2:
                gtype = 3
            # VTK_Quad
            elif i_points_per_cell == 4:
                gtype = 9
            # VTK_Polygon
            else:
                gtype = 7
        else:
            # VTK_Triangle (for surface meshes)
            if i_points_per_cell == 3:
                gtype = 5
            # VTK_Line
            elif i_points_per_cell == 2:
                gtype = 3
            # VTK_Tetra
            elif i_points_per_cell == 4:
                gtype = 10
            # VTK_Hexahedron
            elif i_points_per_cell == 8:
                gtype = 12
            else:
                gtype = 7

        poly_types[iidx_start - 1:iidx_end] = gtype * np.ones((n_cells_current_type, 1))
        if iidx_start == iidx_end:
            poly_types[iidx_start - 1] = gtype * np.ones((n_cells_current_type, 1))
        iidx_start = iidx_end + 1

        if np.any(polygons[i_points_per_cell]):
            for row in polygons[i_points_per_cell]:
                to_write += format_spec.format(*row.astype(np.int).ravel())
        format_spec_start += " {:d}"
    to_write += f'        </DataArray>\n'

    # offsets
    to_write += f'        <DataArray type="Int32" Name="offsets" format="ascii">\n'
    for offset in offsets:
        to_write += '           {:d}\n'.format(*offset.astype(np.int).ravel())
    to_write += f'        </DataArray>\n'

    # cell types
    to_write += f'        <DataArray type="UInt8" Name="types" format="ascii">\n'
    for polyType in poly_types:
        to_write += '           {:d}\n'.format(*polyType.astype(np.int).ravel())
    to_write += f'        </DataArray>\n' \
                f'      </Cells>\n'

    # point data
    to_write += f'      <PointData>\n'
    for i in range(n_scalar_data_sets):
        to_write += f'        <DataArray type="Float32" Name="{scalar_data_names[i]}" NumberOfComponents="1" ' \
                    f'format="ascii">\n'
        for pd in scalar_data[:, i].ravel():
            to_write += '          {:.6e}\n'.format(pd)
        to_write += f"        </DataArray>\n"
    for i in range(n_vec_data_sets):
        to_write += f'        <DataArray type="Float32" Name="{vec_data_names[i]}" NumberOfComponents="3" ' \
                    f'format="ascii">\n'
        for vd in vec_data[i]:
            to_write += '          {:.4e} {:.4e} {:.4e}\n'.format(*vd)
        to_write += '        </DataArray>\n'
    to_write += f'      </PointData>\n'

    # footer section
    to_write += f'    </Piece>\n' \
                f'  </UnstructuredGrid>\n' \
                f'</VTKFile>\n'

    # write the created vtu string to a file
    filename = filename if filename.endswith(".vtu") else filename + ".vtu"
    with open(filename, "w") as f:
        f.write(to_write)
