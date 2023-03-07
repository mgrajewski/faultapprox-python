"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import norm

from entities.Entities import FaultApproxParameters
from geometry import compute_dist_mat


def get_barycentres(point_set: npt.ArrayLike, class_of_points: npt.ArrayLike,
                    fault_approx_parameters: FaultApproxParameters) -> npt.ArrayLike:
    """
    Compute a rough approximation of fault lines based upon barycentres. We follow Allasia et al.: Adaptive detection
    and approximation of unknown surface discontinuities from scattered data, Simulation in Modelling Practice and
    Theory, July 2009
    For a given sampling point, we consider the n_nearest_points the nearest ones. If this subset contains points from
    more than one class, we compute the barycentres of the points in one class. As approximation to the fault line,
    we consider the arithmetic mean between all combinations of barycentres within the n_nearest_points nearest points
    and store these new points in barycentres.
    In Allasia et al., the means of the barycentres are called barycentres as well. We deviate from this terminology.

    Args:
        point_set (npt.ArrayLike):
            Set of points. Shape: (num_points, ndim).
        class_of_points (npt.ArrayLike): classes of the points in points.
            We assume that the classes are ordered according to PointSet such that class_of_points[i] represents the
            class of the i-th point in points. Shape: (num_points,)
        fault_approx_parameters (FaultApproxParameters): Class containing all relevant parameters for our method of
            detecting fault lines.

    Returns:
        means_of_barycentres (npt.ArrayLike): (see description above)
    """

    point_set = point_set.astype(float)

    # number of points and dimension
    num_points, ndim = point_set.shape

    # number of the nearest points to consider
    n_nearest_points = fault_approx_parameters.n_nearest_points

    # When removing duplicates, points closer than eps are considered identical.
    eps = fault_approx_parameters.eps

    # We do not know how many barycentres exist, but at most num_points ones.
    # We shorten the vector later on.
    means_of_barycentres = np.zeros((num_points, ndim), dtype=float)

    # total number of the new means of barycentres computed from point_set
    n_means_of_barycentres = 0

    # lower n_nearest_points if necessary
    n_nearest_points = min(n_nearest_points, num_points)

    # Compute distance matrix for finding the nearest points.
    dist_mat = compute_dist_mat(point_set)

    # matrix containing the indices of the nearest points for all points in point_set
    idx_nearest_neighbours = np.argsort(dist_mat, axis=0, kind='stable')

    # loop over points
    for ipoint in range(num_points):

        # These are the n_nearest_points nearest neighbours to the given point.
        idx_current_nearest_neighbours = idx_nearest_neighbours[0:n_nearest_points, ipoint]
        nearest_neighbours = point_set[idx_current_nearest_neighbours]

        # A point is a fault point if in its neighborhood are points of several classes.
        class_of_nearest_neighbours = class_of_points[idx_current_nearest_neighbours]

        # Find the values and the number of classes represented in nearest_neighbours.
        class_vals = np.unique(class_of_nearest_neighbours)
        nclasses = class_vals.shape[0]

        # nearest_neighbours contains points from more than one class.
        if nclasses > 1:

            # For each class, compute the barycentres of the points in nearest_neighbours which belong to that class.
            barycentres_in_class = np.zeros((nclasses, ndim), dtype=float)

            # Add barycentres for improved approximation of the fault.
            for iclass in range(nclasses):
                points_in_class = nearest_neighbours[class_of_nearest_neighbours == class_vals[iclass]]

                # arithmetic mean of the points within one class (aka barycentres)
                barycentres_in_class[iclass] = np.sum(points_in_class, axis=0) / points_in_class.shape[0]

            # Compute means of barycentres for any reasonable combination of classes and store them in
            # barycentres_in_class.
            for iclass in range(nclasses):
                for jclass in range(iclass + 1, nclasses):
                    n_means_of_barycentres += 1
                    if n_means_of_barycentres > means_of_barycentres.shape[0]:
                        means_of_barycentres = np.concatenate((means_of_barycentres, 0.5 * (
                                barycentres_in_class[iclass] + barycentres_in_class[jclass])[np.newaxis, :]), axis=0)
                    else:
                        means_of_barycentres[n_means_of_barycentres - 1] = 0.5 * (
                                barycentres_in_class[iclass] + barycentres_in_class[jclass])

    # Shorten vector to its appropriate length.
    means_of_barycentres = means_of_barycentres[0:n_means_of_barycentres]

    # Last part: remove duplicates. Two points are considered duplicate if closer than eps.
    for i in range(n_means_of_barycentres - 1):
        for j in range(i + 1, n_means_of_barycentres):
            if np.abs(norm(means_of_barycentres[i] - means_of_barycentres[j])) < eps:
                means_of_barycentres[j] = -42

    means_of_barycentres = means_of_barycentres[np.logical_not(means_of_barycentres[:, 0] == -42)]

    return means_of_barycentres
