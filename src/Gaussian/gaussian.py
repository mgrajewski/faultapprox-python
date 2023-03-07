"""
Author: Matthias Grajewski (grajewski@fh-aachen.de) and Luis Hasenauer
This file is part of faultapprox-python (https://github.com/mgrajewski/faultapprox-python)
"""
import numpy as np
import numpy.typing as npt
from numba import jit, float64


@jit(float64[:](float64[:, :], float64[:], float64), nopython=True)
def gaussian(x: npt.ArrayLike, xc: npt.ArrayLike, scale: float) -> npt.ArrayLike:
    """gaussian computes the values of the Gaussian RBF centered around xc with scaling factor scale in the set of
    points x.

    Gaussian(x) = :math:`psi(||x-xc||//scale)`
    :math:`psi(y) = e^(-y^2)`

    Args:
        x (npt.ArrayLike):
            array containing the cartesian coordinates of points to evaluate the RBF at. Shape: (npoints, dim)
        xc (npt.ArrayLike):
            array containing the cartesian coordinates of the center. Shape: (dim,)
        scale (float): Scaling factor for the RBF.

    Returns:
        array: array containing the values for the RBF in x. Shape: (npoints,)
    """

    # ||x-xc||^2 stored in func_vals; the numpy-version with matrix-vector product is slightly faster than
    # np.sum with axis=1
    func_vals = (x - xc)**2 @ np.ones(xc.shape[0])

    aux_scale = 1. / (scale * scale)

    func_vals = np.exp(-func_vals * aux_scale)
    return func_vals


@jit(float64[:, :](float64[:, :], float64[:], float64), nopython=True)
def gaussian_first_der(x: npt.ArrayLike, xc: npt.ArrayLike, scale: float) -> npt.ArrayLike:
    """gaussian_first_der computes the first derivatives aka the gradient of the Gaussian RBF centered around xc with
        scaling factor scale in the set of points x.

        Gaussian(x) = psi(||x-xc||/scale)
            psi(y) = e^(-y^2)

        Gaussian_first_der(x) = -2/(scale^2)*exp(-||x-xc||/scale) * (x-xc)

    Args:
        x (npt.ArrayLike):
            array containing the cartesian coordinates of points to evaluate the RBF at. Shape: (npoints, dim)
        xc (npt.ArrayLike):
            array containing the cartesian coordinates of the center. Shape: (dim,)
        scale (float): Scaling factor for the RBF.

    Returns:
        derivatives(npt.ArrayLike): array containing the gradients aka derivatives. Shape: (npoints, ndim)

    """
    npoints, ndim = x.shape

    diff_vec = x-xc

    # ||x-xc||^2 stored in derivatives
    # slightly faster than np.sum with axis=1; the numpy-version with matrix-vector product is slightly faster than
    # np.sum with axis=1
    derivatives = (x - xc)**2 @ np.ones(xc.shape[0])

    aux_scale = 1. / (scale * scale)

    aux = (-2 * aux_scale * np.exp(-derivatives * aux_scale))
    derivatives = np.outer(aux, np.ones(ndim)) * diff_vec

    return derivatives


@jit(float64[:, :, :](float64[:, :], float64[:], float64), nopython=True)
def gaussian_second_der(x: npt.ArrayLike, xc: npt.ArrayLike, scale: float) -> npt.ArrayLike:
    """gaussian_second_der computes the values of the second derivative of the Gaussian RBF centered around xc with
    scaling factor scale in the set of points x.
        
    Gaussian_second_der(x) =
        :math:`psi(||x-xc||/scale)*( 4/(scale^4)*(x-xc)*(x-xc)' - 2/(scale^2)*I)`
        
    Args:
        x (npt.ArrayLike):
            array containing the cartesian coordinates of points to evaluate the RBF at. Shape: (npoints, dim)
        xc (npt.ArrayLike):
            array containing the cartesian coordinates of the center. Shape: (dim,)
        scale (float): Scaling factor for the RBF.

    Returns:
        hessian(npt.ArrayLike): array containing the Hessians aka 2nd derivatives. Shape: (npoints, ndim, ndim)

    """

    npoints, ndim = x.shape

    # difference vector
    diff_vec = x - xc

    # ||x-xc||^2 stored in norms_sq. The variant with matrix-vector multiplication seems to be slightly faster than
    # np.sum with axis=1
    norms_sq = diff_vec**2 @ np.ones(xc.shape[0])

    hessian = np.zeros((npoints, ndim, ndim))
    aux_scale = 1. / (scale * scale)

    aux = 2 * aux_scale * np.eye(ndim)

    #  psi(||x-xc||/scale)*( 4/(scale^4)*(x-xc)*(x-xc)' - 2/(scale^2)*I)
    for ipoint in range(npoints):
        hessian[ipoint, :, :] = (4 * aux_scale * aux_scale * np.outer(diff_vec[ipoint], diff_vec[ipoint]) - aux) * \
                                np.exp(-norms_sq[ipoint] * aux_scale)
    return hessian


def to_arrays(x, xc):
    """This function makes single numbers or even 1d-numpy arrays 2d-numpy arrays for consistency reasons. This
    simplifies the computation in gaussian itself and enables to numba-compile the actual RBF functions.

    Args:
        x (double or npt.ArrayLike): evaluation point(s)
        xc (double or npt.ArrayLike): center point

    Returns:
        x (npt.ArrayLike) : evaluation points as 2d-numpy array. Shape: (number of points, ndim)
        xc (npt.Arraylike) : center point as 1d-numpy array. Shape: (dim,)
    """    
    #  If x is just a number: make it an array
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    #  If xc is just a number: make it an array
    if not isinstance(xc, np.ndarray):
        xc = np.asarray(xc)

    if xc.ndim == 0:
        xc = np.array([xc], dtype=np.float64)
    elif xc.ndim >= 2:
        raise Exception('Only one center point per function call supported.')

    # If x is a 1d-array
    if x.ndim == 1:
        x = x.reshape((1, x.shape[0]))

        # special case 1d: list of points as 1d array
        if xc.shape[0] == 1:
            x = x.T

    elif x.ndim == 0:
        x = np.array([[x]], dtype=np.float64)
    elif x.ndim > 2:
        raise Exception('Arrays with two dimensions or more are not supported.')

    x = x.astype(np.float64)
    xc = xc.astype(np.float64)

    return x, xc


if __name__ == '__main__':
    # -- 1D test case --
    # one point; point and center as ordinary numbers
    x_test = 1
    xc_test = 1
    scale_test = 0.5

    # reference value: 1.0
    ref_val = np.array([1.0])
    print('1d, one point: deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian(*to_arrays(x_test, xc_test), scale_test), 2)))

    # reference value: 0.0
    ref_val = np.array([[0.0]])
    print('1d, one point, derivative: deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian_first_der(*to_arrays(x_test, xc_test), scale_test), 2)))

    # reference value
    ref_val = np.array([[[-8.0]]])
    print('1d, one point, 2nd derivative: deviation is %1.2e' %
          (ref_val - gaussian_second_der(*to_arrays(x_test, xc_test), scale_test)))

    # three points, center as ordinary number
    x_test = np.asarray([0.5, 1, 2])
    xc_test = 1
    scale_test = 0.5

    # reference values
    ref_val = np.array([0.367879441171442, 1.0, 0.018315638888734])
    print('1d, three points: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian(*to_arrays(x_test, xc_test), scale_test), 2)))

    # reference values
    ref_val = np.array([[1.471517764685769], [0.0], [-0.146525111109873]])
    print('1d, three points, derivative: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian_first_der(*to_arrays(x_test, xc_test), scale_test), 2)))

    # reference values
    ref_val = np.array([[[2.943035529371539]], [[-8.0]], [[1.025675777769114]]])
    print('1d, three points, 2nd derivative: norm of deviation is %1.2e' %
          (np.linalg.norm((ref_val - gaussian_second_der(*to_arrays(x_test, xc_test), scale_test)).flatten(), 2)))

    print()
    # -- 2D test case --
    # one point
    x_test = np.array([3, 1])
    xc_test = np.array([1, 2])
    scale_test = 2

    # reference value
    ref_val = np.array([0.286504796860190])
    print('2d, one point: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian(*to_arrays(x_test, xc_test), scale_test), 2)))

    # reference value
    ref_val = np.array([[-0.286504796860190, 0.143252398430095]])
    print('2d, one point, derivative: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian_first_der(*to_arrays(x_test, xc_test), scale_test), 2)))

    # reference value
    ref_val = np.array([[[0.143252398430095, -0.143252398430095], [-0.143252398430095, -0.071626199215048]]])
    print('2d, one point, 2nd derivative: norm of deviation is %1.2e' %
          (np.linalg.norm((ref_val - gaussian_second_der(*to_arrays(x_test, xc_test), scale_test))[0], 2)))

    # two points
    x_test = np.array([[2, 3], [3, 1]], dtype=np.float64)
    xc_test = np.array([1, 2], dtype=np.float64)
    scale_test = 2

    # reference values
    ref_val = np.array([0.606530659712633, 0.286504796860190])
    print('2d, two points: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian(x_test, xc_test, scale_test), 2)))

    # reference values:
    #
    ref_val = np.array([[-0.303265329856317, -0.303265329856317], [-0.286504796860190, 0.143252398430095]])
    print('2d, two points, derivative: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian_first_der(x_test, xc_test, scale_test), 2)))

    ref_val = np.array([[[-0.151632664928158, 0.151632664928158], [0.151632664928158,  -0.151632664928158]],
                        [[0.143252398430095, -0.143252398430095], [-0.143252398430095, -0.071626199215048]]])
    deviation = ref_val - gaussian_second_der(x_test, xc_test, scale_test)
    norm_dev = 0
    for i in range(deviation.shape[0]):
        norm_dev = norm_dev + np.linalg.norm(deviation[i], 2)

    print('2d, two points, 2nd derivative: norm of deviation is %1.2e' % norm_dev)

    # three points
    x_test = np.array([[2, 3], [3, 1], [1, 3]], dtype=np.float64)
    xc_test = np.array([1, 2], dtype=np.float64)
    scale_test = 2

    # reference values
    ref_val = np.array([0.606530659712633, 0.286504796860190, 0.778800783071405])
    print('2d, three points: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian(x_test, xc_test, scale_test), 2)))

    # reference values: -0.303265329856317  -0.303265329856317
    #                   -0.286504796860190   0.143252398430095
    #                    0                  -0.389400391535702
    ref_val = np.array([[-0.303265329856317, -0.303265329856317],
                        [-0.286504796860190, 0.143252398430095],
                        [0, -0.389400391535702]])
    print('2d, three points, derivatives: norm of deviation is %1.2e' %
          (np.linalg.norm(ref_val - gaussian_first_der(x_test, xc_test, scale_test), 2)))
    print('2d, three points, 2nd derivatives:', gaussian_second_der(x_test, xc_test, scale_test))
