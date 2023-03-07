"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""

import types
import numpy as np
import numpy.typing as npt
import tools.ncalls as ncalls

from entities.Entities import ProblemDescr


def compute_classification(points: npt.ArrayLike, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """ We assume that we classify points according to some function. For maximal flexibility, this function is provided
    as a function pointer in problem_descr. Its behaviour can be controlled by an object function_pars which is
    part of the list of function parameters and provided with problem_descr.
    We do not specify what this object contains nor how its content acts on function behaviour but delegate this to
    the user. We however specify the parameter list of our function: f(points, function_pars).

    Args:
        points (npt.ArrayLike):
            points to classify. Shape: (npoints, dim)
        problem_descr (ProblemDescr):
            this object contains all parameters and information related to the fault approximation problem

    Returns:
        class_of_points (npt.ArrayLike):
            array containing the class indices. Shape: (npoints,)

    Raises:
        Exception: If no function is given in problem_descr.
    """

    if len(points.shape) == 1:
        points = points[np.newaxis, :]
    num_points, dim = points.shape

    # test, if some test function is provided in problem_descr
    error_message = f"on {compute_classification.__name__}No function given in ProblemDescr"
    assert not (problem_descr.test_func == '' or problem_descr.test_func is None), error_message

    # there is a function given: check, if this function really exists
    if isinstance(problem_descr.test_func, (types.FunctionType, types.BuiltinFunctionType)):
        if num_points > 0:
            class_of_points = problem_descr.test_func(points, problem_descr)

            # update number of function evaluations
            ncalls.total += num_points
            ncalls.acc += 1

        else:
            class_of_points = np.zeros(0, dtype=int)

    else:
        # there is no test function given: skip computation
        raise Exception("No function given in ProblemDescr")

    if dim >= 2:
        class_of_points[points[:, 0] < problem_descr.x_min[0]] = -1
        class_of_points[points[:, 1] < problem_descr.x_min[1]] = -1
        class_of_points[points[:, 0] > problem_descr.x_max[0]] = -1
        class_of_points[points[:, 1] > problem_descr.x_max[1]] = -1

    if dim == 3:
        class_of_points[points[:, 2] < problem_descr.x_min[2]] = -1
        class_of_points[points[:, 2] > problem_descr.x_max[2]] = -1

    return class_of_points
