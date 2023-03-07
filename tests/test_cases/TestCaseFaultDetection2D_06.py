"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
import numpy as np
import logging

import src.utils
from faultapprox import fault_approx
from entities.Entities import FaultApproxParameters, ProblemDescr

from tests.test_funcs.TestFunc2D import func_fd_2d_cl2_c0_02
import tools.ncalls as ncalls


def main(log_level):
    my_settings = FaultApproxParameters()
    my_settings.max_dist_for_surface_points = 0.05
    my_settings.abstol_bisection = 0.001
    my_settings.num_points_local = 10
    my_settings.err_max = 0.002
    my_settings.err_min = 1e-4

    my_prob = ProblemDescr()
    my_prob.output_file_vtu = "testFaultDetection2D_06.vtu"
    my_prob.test_func = func_fd_2d_cl2_c0_02
    my_prob.x_min = np.array([0, 0])
    my_prob.x_max = np.array([1, 1])

    ncalls.acc = 0
    ncalls.total = 0

    n_point_per_side = 10

    point_set = np.arange(0, 1 + 0.01, 1 / (n_point_per_side - 1)).reshape(1, -1)
    x = np.tile(point_set, (10, 1)).reshape(1, n_point_per_side ** 2)
    y = np.tile(point_set.T, (1, 10)).reshape(1, n_point_per_side ** 2)
    point_set = np.concatenate((x.T, y.T), axis=1)

    logging.basicConfig(format='%(message)s', level=log_level)
    subdomains, successful = fault_approx(point_set, my_prob, my_settings)

    if successful:
        logging.debug('- reconstruction of subdomains successful')

    logging.info(str(ncalls.acc) + ', ' + str(ncalls.total))

if __name__ == '__main__':
    main(logging.DEBUG)
