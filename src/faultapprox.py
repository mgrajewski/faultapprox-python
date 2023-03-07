"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
from warnings import warn

import numpy as np
import numpy.typing as npt
from typing import Tuple
import meshio
import logging

from computeClassification import compute_classification
from getBarycentres import get_barycentres
from getTripletsNearFault import get_triplets_near_fault
from entities.Entities import FaultApproxParameters, ProblemDescr
from postprocessing import reconstruct_subdomains, vis_subdomain
import tools.statistics as stats
import tools.ncalls as ncalls


def fault_approx(point_set: npt.ArrayLike, problem_descr: ProblemDescr, fault_approx_pars: FaultApproxParameters) -> \
        Tuple[npt.ArrayLike, bool]:
    """
    We consider piecewise smooth a function f: \Omega -> {1,..., n} such that there are lines or more generally
    manifolds where the function is discontinuous, and \Omega is an axis-parallel rectangle or a cuboid.
    We assume that for \Omega_i = f^{-1}(i),
       \Omega = \overline{\Omega_i} \cup \hdots \overline\Omega_n}.
    The purpose of this function is to represent such lines/surfaces by sufficiently many points in their near vicinity.
    We will call a curve or surface \overline{\Omega_i} \cap \overline{\Omega_j} a fault in what follows.

    We proceed as follows:
    1) Starting from an initial sample set, we detect how many of such subdomains exist.
    2) step initialise: We detect all sample points in the vicinity of a fault surface and utilise them to find
       additional points nearer to the faults based on barycentres. We repeat that process once. We classify these
       points accordingly and use for them for finding additional ones extremely close to the decision line/surface with
        a bisection approach (Building block iniapprox).
    3) step fill: We detect gaps or holes in the representing point set, which we fill in this step.
    4) step expand: We ensure that the set of points represents the fault in its entire extent.
    5) step adapt: We adaptively add or remove (2D only) points in the representing sets depending on geometric
       properties like curvature. We end up with a point cloud representing the faults sufficiently accurately.
    6) step reconstruct: In 2D, we describe each of the subdomains \Omega_i by a closed polygonal approximation of
       its boundary based on the corresponding faults. The points the polygon consists of are ordered counter-clockwise
       such that the subdomain appears on the right side when following the points.
    7) We provide a visualisation as vtu-file if desired.

    Args:
        point_set (npt.ArrayLike):
            Initial set of points where to sample f; Dimension is number of points x ndim, where ndim is the dimension
            of \Omega, 2 or 3. Shape: (npoints, ndim)
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.
        fault_approx_pars (FaultApproxParameters):
            Object containing all parameters relevant for the fault detection algorithm. We refer to its documentation
            in Entities.py for details.

    Returns:
        npt.ArrayLike: array containing all subdomain information
        boolean: reconstr_succeeded True, if (in 2D), a closed polygonal approximation of any \Omega_i could be obtained
"""

    ndim = point_set.shape[1]

    #  Remove points which are obviously not inside the domain to consider.
    for idim in range(ndim):
        point_set = point_set[point_set[:, idim] <= problem_descr.x_max[idim], :]
        point_set = point_set[point_set[:, idim] >= problem_descr.x_min[idim], :]

    # Classification of the initial points.
    class_of_points = compute_classification(point_set, problem_descr)

    # class_vals contains the class values found so far. These are not necessarily the class indices: Imagine that
    # f(\Omega) = {1,2,5} and all corresponding subdomains have been covered by points. Then, ClassValues = [1,2,5],
    # whereas the class indices range from 0 to 2.
    class_vals = np.unique(class_of_points)

    # If there is more than one class present in the initial point set, we start reconstructing the subdomains.
    if class_vals.size > 1:

        logging.debug('- compute second set of barycentres M')

        # First rough approximation of the faults by means of barycentres.
        means_barys = get_barycentres(point_set, class_of_points, fault_approx_pars)

        # classes of the means of barycentres
        class_means_barys = compute_classification(means_barys, problem_descr)

        if problem_descr.extended_stats:
            stats.pos_in_code.append('after_M')
            stats.means_barys = means_barys
            stats.class_means_barys = class_means_barys
            stats.ncalls.append(ncalls.total)

        # Second slightly less rough approximation of the faults by means of barycentres of means of barycentres.
        logging.debug('- compute second set of barycentres M2')

        means_barys_2 = get_barycentres(means_barys, class_means_barys, fault_approx_pars)
        class_means_barys_2 = compute_classification(means_barys_2, problem_descr)

        if problem_descr.extended_stats:
            stats.pos_in_code.append('after_M2')
            stats.means_barys_2 = means_barys_2
            stats.class_means_barys_2 = class_means_barys_2
            stats.ncalls.append(ncalls.total)

        barycentres = np.concatenate((means_barys, means_barys_2), axis=0)
        class_of_barys = np.concatenate((class_means_barys, class_means_barys_2), axis=0)
        point_set = np.concatenate((point_set, barycentres), axis=0)
        class_of_points = np.concatenate((class_of_points, class_of_barys), axis=0)

        # It may happen that not all classes are detected in the initial point set, but only later in the extended point
        # set classified so far. Therefore, we recompute class_vals here.
        class_vals = np.unique(class_of_points)

        logging.debug('- compute points on subdomain boundaries')
        # get_points_near_surf contains the steps iniapprox, fill, expand and adapt.
        point_sets_surface, left_domain_start, left_domain_end, success, normals_surface = get_triplets_near_fault(
            barycentres, point_set, class_of_barys, class_of_points, problem_descr, fault_approx_pars)

        if not success:
            reconstr_succeeded = False
            subdomains = 0
            return subdomains, reconstr_succeeded

        logging.debug('- reconstruct subdomains')
        # Step reconstruct: Describe the subdomains approximately as closed polygons according to the classes using
        # the points near the faults (only for 2D).
        subdomains, reconstr_succeeded = reconstruct_subdomains(point_sets_surface, left_domain_start,
                                                                left_domain_end, problem_descr, class_vals,
                                                                fault_approx_pars)

        if problem_descr.output_file_vtu != '' and reconstr_succeeded:
            # prepare visualisation
            logging.debug('- compute visualisation of subdomains')
            point_set, cells, cell_data = vis_subdomain(subdomains, class_vals)

            logging.debug('- write visualisation file')

            # export as vtu - file for further analysis in e.g. paraview
            meshio.write_points_cells(problem_descr.output_file_vtu, point_set, cells, cell_data=cell_data)

        return subdomains, reconstr_succeeded

    # All points in the initial point set belong to the same class: display a warning and return dummy values.
    else:
        warn('All points in the initial point set belong to the same class.')
        warn('Consider employing an enriched set of sampling points.')
        return np.array([0]), False
