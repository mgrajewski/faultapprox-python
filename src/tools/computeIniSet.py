"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
import numpy as np
import numpy.typing as npt

import src.utils

from entities.Entities import ProblemDescr


def comp_ini_set(npoints: int, mode: str, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    This function generates an initial set of npoints sampling points.

    Args:
        npoints (int):
            number of points in the initial set
        mode (string):
            mode is in {tensor, halton} and describes what type of set shall be generated
        problem_descr (ProblemDescr):
            Object containing all problem-relevant parameters. We refer to its documentation in Entities.py for details.

    Returns:
        npt.ArrayLike: initial point set
    """
    ndim = problem_descr.x_min.shape[0]

    if mode == 'tensor':

        npoints_per_side = np.power(npoints, 1/ndim).astype(int)
        npoints = npoints_per_side**ndim

        point_set = np.linspace(0, 1, npoints_per_side)
        if ndim == 2:
            x, y = np.meshgrid(point_set, point_set)
            x = x.reshape((npoints, 1))
            y = y.reshape((npoints, 1))
            point_set = np.concatenate((x, y), axis=1)
        elif ndim == 3:
            x, y, z = np.meshgrid(point_set, point_set, point_set)
            x = x.reshape((npoints, 1))
            y = y.reshape((npoints, 1))
            z = z.reshape((npoints, 1))
            point_set = np.concatenate((x, y, z), axis=1)

    elif mode == 'halton':

        point_set = halton_points(npoints, ndim, 1)

    else:
        raise Exception("Invalid choice of method.")
    # transform to actual domain
    for idim in range(ndim):
        point_set[:, idim] = (problem_descr.x_max[idim] - problem_descr.x_min[idim])*point_set[:, idim] \
                             + problem_descr.x_min[idim]

    return point_set


def halton_points(npoints: int, ndim: int, seed=0):
    """
        This function creates a set of points using the Halton sequence for
        each coordinate in [0,1]^dim. For details, see e.g.
        https://en.wikipedia.org/wiki/Halton_sequence

        Args:
            npoints (int):
                desired number of sampling points
            ndim (int):
                dimension of hypercube [0,1]^dim to sample
            seed (default = 0):
                seed for Halton sequence (useful if one wants to continue with an existing Halton sequence)

        Returns:
            npt.ArrayLike: array of shape (num_samples, dim) containing the cartesian coordinates of the sampling points
        """
    # primes as starting point for each coordinate (here: implicitly hard-coded to 8, as I do not see realistic
    # scenarios with dim > 8)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19])
    # allocate the point set
    point_set = np.zeros((npoints, ndim))

    for i_dim in range(ndim):
        basis = primes[i_dim]
        for i_sample in range(npoints):
            n0 = i_sample + seed + 1
            halton_number = 0
            f = 1 / basis
            while n0 > 0:
                n1 = np.floor(n0 / basis)
                r = n0 - n1 * basis
                halton_number += f * r
                f /= basis
                n0 = n1
            point_set[i_sample, i_dim] = halton_number
    return point_set
