"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
import numpy as np
import numpy.typing as npt

from inpoly import inpoly2


def classify_points(points: npt.ArrayLike, subdomains: npt.ArrayLike) -> npt.ArrayLike:
    """The classification of points induces a subdivision of the domain into subdomains which themselves may consist of
     several components. fault_approx is to compute approximations to these sets.
    This subroutine returns the class and component index of the points in the array points with respect to the
    approximations provided by fault_approx.

    This function uses inpoly2 from inpoly (https://github.com/dengwirda/inpoly-python)

    Args:
        points (npt.ArrayLike):
            N x 2-array of point coordinates.
        subdomains (npt.ArrayLike): Cell array containing polygonal approximations of the
            subdomains and their components.

    Returns:
        n_classes (array): (length: N) containing the classes of the points in
            Points (-1, if the point could not be assigned to any class).
        component (array): (length: N) containing the component indices of the
            classes the points in Points belong to (-1, if no component could
            be found).

    """

    dim = points.shape[1]

    assert dim == 2, "This function works in 2D only."

    classes = -1 * np.ones((points.shape[0], 1))
    component = -1 * np.ones((points.shape[0], 1))
    n_classes = subdomains.shape[0]
    for i_class in range(n_classes):
        for i_comp in range(subdomains[i_class].shape[0]):
            index = inpoly2(points, subdomains[i_class, i_comp])
            classes[index] = i_class
            component[index] = i_comp

    return [n_classes, component]
