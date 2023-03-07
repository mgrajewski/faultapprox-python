"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
import numpy as np
import numpy.typing as npt

import src.utils
from entities.Entities import ProblemDescr


def func_fd_2d_cl3_c0_01(point_set: npt.ArrayLike, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
        former test case 01

    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)
    class_of_points[np.arctan(5 * point_set[:, 0] + 5 * point_set[:, 1]) < 0.5] = 2
    class_of_points[point_set[:, 1] > point_set[:, 0] * 0.7 + 0.5] = 3

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c0_01_random(a: float, b: float):
    def __func_fd_2d_cl3_c0_01(point_set: npt.ArrayLike, problem_descr: ProblemDescr) -> npt.ArrayLike:
        class_of_points = np.ones(point_set.shape[0], dtype=np.int)
        class_of_points[np.arctan(5 * point_set[:, 0] + 5 * point_set[:, 1]) < a] = 2
        class_of_points[point_set[:, 1] > point_set[:, 0] * b + 0.5] = 3

        class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

        return class_of_points

    return __func_fd_2d_cl3_c0_01


def func_fd_2d_cl2_c0_01(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 02
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)
    class_of_points[point_set[:, 0] > 0.55] = 2
    class_of_points[point_set[:, 1] > 0.78] = 2
    class_of_points[point_set[:, 1] < 0.33] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl2_c0_01_random(a: float, b: float, c: float):
    def __func_fd_2d_cl2_c0_01(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
        class_of_points = np.ones(point_set.shape[0], dtype=np.int)
        class_of_points[point_set[:, 0] > a] = 2
        class_of_points[point_set[:, 1] > b] = 2
        class_of_points[point_set[:, 1] < c] = 2

        class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

        return class_of_points

    return __func_fd_2d_cl2_c0_01


def func_fd_2d_cl4_c0_01(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 03
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[(point_set[:, 1] <= point_set[:, 0]) & (point_set[:, 1] >= 1 - point_set[:, 0])] = 2
    class_of_points[(point_set[:, 1] >= point_set[:, 0]) & (point_set[:, 1] >= 1 - point_set[:, 0])] = 4
    class_of_points[
        (point_set[:, 0] >= 0.3) & (point_set[:, 0] < 0.7) & (point_set[:, 1] > 0.3) & (point_set[:, 1] < 0.7)] = 3

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c1_01(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 04
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[point_set[:, 1] > 0.7 + 0.1 * np.sin(12.0 * point_set[:, 0])] = 2
    class_of_points[point_set[:, 1] < 0.3 + 0.1 * np.sin(12.0 * point_set[:, 0])] = 2
    class_of_points[(point_set[:, 1] - 0.5) ** 2 + (point_set[:, 0] - 1.0) ** 2 >= 0.2] = 3

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c1_01_random(a: float, b: float, c: float):
    def __func_fd_2d_cl3_c1_01(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
        """
        former test case 04
        """
        class_of_points = np.ones(point_set.shape[0], dtype=np.int)

        class_of_points[point_set[:, 1] > a + 0.1 * np.sin(b * point_set[:, 0])] = 2
        class_of_points[point_set[:, 1] < (1.0 - a) + 0.1 * np.sin(b * point_set[:, 0])] = 2
        class_of_points[(point_set[:, 1] - 0.5) ** 2 + (point_set[:, 0] - 1.0) ** 2 >= c] = 3

        class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

        return class_of_points

    return __func_fd_2d_cl3_c1_01


def func_fd_2d_cl3_c1_02(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 05
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[point_set[:, 0] < point_set[:, 1] - 0.4] = 2
    class_of_points[point_set[:, 0] > point_set[:, 1] - 0.2] = 2
    class_of_points[point_set[:, 0] > point_set[:, 1] + 0.3] = 3

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c1_02_random(a: float, b: float, c: float):
    def __func_fd_2d_cl3_c1_02(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
        """
        former test case 05
        """
        class_of_points = np.ones(point_set.shape[0], dtype=np.int)

        class_of_points[point_set[:, 0] < a*point_set[:, 1] - (1-a)] = 2
        class_of_points[point_set[:, 0] > b*point_set[:, 1] - 0.2] = 2
        class_of_points[point_set[:, 0] > point_set[:, 1] + c] = 3

        class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

        return class_of_points
    return __func_fd_2d_cl3_c1_02


def func_fd_2d_cl2_c0_02(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 06
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[(point_set[:, 0] - 0.5) ** 2 + (point_set[:, 1] - 0.5) ** 2 < 0.2] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl2_c1_03(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 08
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    # Ball with radius 0.15 around (1, 0.7)
    class_of_points[(point_set[:, 0] - 1.0) ** 2 + (point_set[:, 1] - 0.7) ** 2 < 0.0225] = 2

    # Ball with radius 0.15 around (1, 0.3)
    class_of_points[(point_set[:, 0] - 1.0) ** 2 + (point_set[:, 1] - 0.3) ** 2 < 0.0225] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl2_c1_04(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 09
    """
    class_of_points = 2 * np.ones(point_set.shape[0], dtype=np.int)

    # ball with radius 0.15 around (1,1)
    class_of_points[(point_set[:, 0] - 1.0) ** 2 + ((point_set[:, 1] - 1.0) ** 2) < 0.0225] = 1

    # second component of subdomain 1 is a kind of stripe
    class_of_points[
        (point_set[:, 1] < 0.5 * point_set[:, 0] - 0.1) & (point_set[:, 1] > 0.5 * point_set[:, 0] - 0.3)] = 1

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c1_03(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 10
    """
    class_of_points = 3 * np.ones(point_set.shape[0], dtype=np.int)

    # subdomain 1 essentially consists of two vertical stripes above 0.7 and below 0.3
    class_of_points[point_set[:, 1] > 0.7] = 1
    class_of_points[point_set[:, 1] < 0.3] = 1

    # subdomain 2 consists of two triangles starting in (0.85, 0.7) and (0.85, 0.3) to the right domain boundary
    class_of_points[(point_set[:, 1] < 0.5 * point_set[:, 0] + 0.275) &
                    (point_set[:, 1] > -0.5 * point_set[:, 0] + 1.125)] = 2
    class_of_points[(point_set[:, 1] < 0.5 * point_set[:, 0] - 0.125) &
                    (point_set[:, 1] > -0.5 * point_set[:, 0] + 0.725)] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c1_04(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 10a (triangles to the left instead of to the right domain boundary)
    """
    class_of_points = 3 * np.ones(point_set.shape[0], dtype=np.int)

    # subdomain 1 essentially consists of two vertical stripes above 0.7 and below 0.3
    class_of_points[point_set[:, 1] > 0.7] = 1
    class_of_points[point_set[:, 1] < 0.3] = 1

    # subdomain 2 consists of two triangles starting in (0.85, 0.7) and (0.85, 0.3) to the right domain boundary
    class_of_points[(point_set[:, 1] > 0.5 * point_set[:, 0] + 0.625) &
                    (point_set[:, 1] < -0.5 * point_set[:, 0] + 0.775)] = 2
    class_of_points[(point_set[:, 1] > 0.5 * point_set[:, 0] + 0.225) &
                    (point_set[:, 1] < -0.5 * point_set[:, 0] + 0.375)] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl2_c1_05(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 11 (three half circles
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    # Ball with radius 0.15 around (0, 0.7)
    class_of_points[point_set[:, 0] ** 2 + (point_set[:, 1] - 0.7) ** 2 < 0.0225] = 2

    # Ball with radius 0.15 around (0, 0.3)
    class_of_points[point_set[:, 0] ** 2 + (point_set[:, 1] - 0.3) ** 2 < 0.0225] = 2

    # Ball with radius 0.2 around (0.5, 0)
    class_of_points[(point_set[:, 0] - 0.5) ** 2 + point_set[:, 1] ** 2 < 0.04] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl2_c0_03(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 12 (one half circle at upper domain boundary)
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    # Ball with radius sqrt(1/10) around (0.5, 1)
    class_of_points[(point_set[:, 0] - 0.5) ** 2 + (point_set[:, 1] - 1.0) ** 2 < 0.1] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl4_c1_01(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 13
    """
    class_of_points = 3 * np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[(point_set[:, 1] > 0.7) | (point_set[:, 1] < 0.3)] = 2

    class_of_points[(point_set[:, 0] < 0.2) & (0.7 < point_set[:, 1]) & (point_set[:, 1] < 0.9)] = 1
    class_of_points[(point_set[:, 0] < 0.2) & (0.1 < point_set[:, 1]) & (point_set[:, 1] < 0.3)] = 1

    class_of_points[(point_set[:, 1] >= 0.9) & (point_set[:, 1] > point_set[:, 0] + 0.7)] = 4
    class_of_points[(point_set[:, 1] <= 0.1) & (point_set[:, 1] < -point_set[:, 0] + 0.3)] = 4

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl2_c1_06(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 14
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[((point_set[:, 1] - 0.28) ** 2 + (point_set[:, 0] - 0.0) ** 2 < 0.03) &
                    ((point_set[:, 1] - 0.28) ** 2 + (point_set[:, 0] - 0.0) ** 2 > 0.01)] = 2

    class_of_points[((point_set[:, 1] - 0.72) ** 2 + (point_set[:, 0] - 0.0) ** 2 < 0.03) &
                    ((point_set[:, 1] - 0.72) ** 2 + (point_set[:, 0] - 0.0) ** 2 > 0.01)] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c0_02(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    former test case 21
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[(point_set[:, 1] > 0.4) & (point_set[:, 0] + point_set[:, 1] < 1.2)] = 2

    class_of_points[(point_set[:, 1] > 1 / 6 * np.sin(12.0 * point_set[:, 0]) + 0.7) &
                    (point_set[:, 0] + point_set[:, 1] >= 1.2)] = 3

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_fd_2d_cl3_c1_05(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
    paper test problem 1
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)

    class_of_points[point_set[:, 1] > 0.6 + 0.1 * np.sin(10.0 * np.pi*np.power(point_set[:, 0], 1.5))] = 2
    #class_of_points[(points[:, 1] - 0.5) ** 2 + (points[:, 0] - 1.0) ** 2 <= 0.16] = 3
    class_of_points[(point_set[:, 1] - 0.5) ** 6 + (point_set[:, 0] - 1.0) ** 6 <= 0.005] = 3

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_Allasia_example1(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
        Example 1 from Allasia, Giampietro et al., Efficient approximation algorithms. Part I: approximation of unknown
        fault lines from scattered data, Dolomites Research Notes On Approximation, Vol. 3, (2010), p. 7-38
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)
    point_set_aux = (point_set[:, 0] - 0.5)**2 + (point_set[:, 1] - 0.5)** 2 <= 0.16
    class_of_points[point_set_aux] = 1 + 2 * np.floor(3.5 * np.sqrt(
        point_set[point_set_aux, 0] * point_set[point_set_aux, 0] + point_set[point_set_aux, 1] * point_set[point_set_aux, 1]))
    class_of_points[class_of_points == 3] = 2
    class_of_points[class_of_points == 5] = 3
    class_of_points[class_of_points == 7] = 4

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_Allasia_example4(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
        inspired by Example 4 from Allasia, Giampietro et al., Efficient approximation algorithms. Part I: approximation
         of unknown fault lines from scattered data, Dolomites Research Notes On Approximation, Vol. 3, (2010), p. 7-38
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)
    class_of_points[point_set[:, 0] > 0.5] = 2
    class_of_points[point_set[:, 0] > 0.6] = 3

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def func_Allasia_example5(point_set: np.ndarray, problem_descr: ProblemDescr) -> npt.ArrayLike:
    """
        inspired by Example 5 from Allasia, Giampietro et al., Efficient approximation algorithms. Part I: approximation
         of unknown fault lines from scattered data, Dolomites Research Notes On Approximation, Vol. 3, (2010), p. 7-38
    """
    class_of_points = np.ones(point_set.shape[0], dtype=np.int)
    class_of_points[(point_set[:, 1] > 0.4) & (point_set[:,0] > 0.4) & (point_set[:,1] < point_set[:,0] + 0.2)] = 2

    class_of_points = __indicate_outside_domain(class_of_points, point_set, problem_descr)

    return class_of_points


def __indicate_outside_domain(class_of_points, point_set, problem_descr):
    class_of_points[point_set[:, 0] < problem_descr.x_min[0]] = -1
    class_of_points[point_set[:, 1] < problem_descr.x_min[1]] = -1
    class_of_points[point_set[:, 0] > problem_descr.x_max[0]] = -1
    class_of_points[point_set[:, 1] > problem_descr.x_max[1]] = -1

    return class_of_points
