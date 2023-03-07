import types

import numpy.typing as npt


class FaultApproxParameters:
    """The class initializes the necessary parameters of the class  FaultApproximation.
    
    Attributes:
        eps (float): points closer as eps are regarded as identical
        n_nearest_points: number of the nearest points to consider when computing barycentres
        max_dist_for_surface_points (float):
            desired maximum distance of a point on the fault line to the next one
        alpha (float):
            for safeguarding in computing valid point pairs as starting values for bisection
        min_dist_factor (float):
            points closer than min_dist_factor*max_dist_for_surface_points are removed if appropriate
        cos_alpha_max (float):
            cosine of maximal admissible angle between consecutive line segments for sorting
        n_nearest_sort (int):
        number of nearest points to consider for sorting
        num_points_to_add (int):
            number of points to add per prolongation sweep
        max_iter_adapt (int): maximal number of adaptive refinement/coarsening steps for adaptively placing points on
            the fault line/surface
        num_points_local (int): number of points for computing a^local coordinate system and normal vectors in case of
            a fault line
        err_min (float): line segments with error lower than err_min are coarsened
        err_max (float): line segments with error larger than err_max are refined
        abstol_bisection (float): stopping criterion for bisection algorithm
        max_iter_bisection (int): maximum number of bisection iterations
        max_trials_for_filling_gaps_in_lines (int): There may be a large distance between neighbouring points near a
            fault line. We try to fill such gaps by adding points. However, to the additional information provided, the
            (assumed) shape of the fault line may changes considerably. Due to that, new gaps may emerge.
            Therefore, we put the filling process in a loop and perform at most max_trials_for_filling_gaps_in_lines
            of such steps.
            In 3D, it takes sometimes several passes to really fill all holes in the representation of a surface. This
            is the number of passes.
    """

    def __init__(self, eps: float = 1e-10,
                 n_nearest_points: int = 10,
                 max_dist_for_surface_points: float = 0.05,
                 alpha: float = 0.25,
                 min_dist_factor: float = 0.2,
                 num_points_to_add: int = 1,
                 cos_alpha_max: float = -0.9,
                 n_nearest_sort: int = 5,
                 max_iter_adapt: int = 4,
                 num_points_local: int = 3,
                 err_min: float = 0.001,
                 err_max: float = 0.15,
                 abstol_bisection: float = 1e-3,
                 max_iter_bisection: int = 15,
                 max_trials_for_filling_gaps_in_lines: int = 3):
        self.eps = eps
        self.n_nearest_points = n_nearest_points

        # Get PointsNearSurface
        self.max_dist_for_surface_points = max_dist_for_surface_points
        self.alpha = alpha
        self.min_dist_factor = min_dist_factor
        self.cos_alpha_max = cos_alpha_max
        self.n_nearest_sort = n_nearest_sort
        self.num_points_to_add = num_points_to_add
        self.max_iter_adapt = max_iter_adapt
        self.num_points_local = num_points_local
        self.err_min = err_min
        self.err_max = err_max

        # parameters for computeSingleSurfacePoint
        self.abstol_bisection = abstol_bisection
        self.max_iter_bisection = max_iter_bisection
        self.max_trials_for_filling_gaps_in_lines = max_trials_for_filling_gaps_in_lines

    def __str__(self):
        result = (f'The parameters of class FaultApproxParameters have the following values:\n'
                  f'eps = {self.eps}\n'
                  f'alpha = {self.alpha}\n'
                  f'min_dist_factor = {self.min_dist_factor}\n'
                  f'cos_alpha_max = {self.cos_alpha_max}\n'
                  f'n_nearest_sort = {self.n_nearest_sort}\n'
                  f'num_points_to_add = {self.num_points_to_add}\n'
                  f'maxiter_adapt = {self.max_iter_adapt}\n'
                  f'num_points_local = {self.num_points_local}\n'
                  f'err_min = {self.err_min}\n'
                  f'err_max = {self.err_max}\n'
                  f'abstol_bisection = {self.abstol_bisection}\n'
                  f'maxiter_bisection = {self.max_iter_bisection}\n'
                  f'max_trials_for_filling_gaps_in_lines = {self.max_trials_for_filling_gaps_in_lines}\n')
        return result

    def __repr__(self):
        report = f' FaultApproxParameters(eps={self.eps}, alpha={self.alpha}, min_dist_factor={self.min_dist_factor}' \
                 f', num_points_to_add={self.num_points_to_add}, maxiter_adapt={self.max_iter_adapt}, ' \
                 f'num_points_local' \
                 f'={self.num_points_local}, err_min={self.err_min}, err_max={self.err_max}, abstol_bisection=' \
                 f'{self.abstol_bisection}, maxiter_bisection={self.max_iter_bisection}, \
                 max_trials_for_filling_gaps_in_lines=' f'{self.max_trials_for_filling_gaps_in_lines}) '
        return report


class ProblemDescr:
    """The reason for collecting all test problem-related data in an object is reproducibility.
    
    In many situations, one forgets to document all parameters and test settings properly such that after some time,
    this information is lost and one cannot reproduce the test case and its results. In such a case, the experiment and
    all related effort was pointless.
    We decided to separate problem-related data and approximation-related data in order to perform parameter studies
    conveniently. In this case, all problem-related data remain unchanged whereas the approximation-related data may
    change.

    Attributes:
        comments (string):
            string for saving some comments
        input_file_excel (string):
            name for Excel input file (can be helpful to read in data from actual measurements or simulations)
        input_file_csv (string):
            name for CSV input file (same reason)
        test_func (function):
            function pointer to test function (NULL if not provided)
        function_parameters (object):
            description object for external models (for maximal flexibility, we just incorporate an object the
            user may define and implement according to his needs)
        output_file_vtu (string):
            file name of VTU output file (no output is written if not specified)
        name_of_data_in_vtu (string):
            name of data field inside VTU
        x_min (int):
            lower boundaries of the domain (necessary if a function pointer only is provided). For explicitly given
            point sets, x_min is ignored.
        x_max (int):
            upper boundaries of the domain (necessary if a function pointer only is provided). For explicitly given
            point sets, x_max is ignored.
        domain_polygon (int): domain for RBF approximation. If not given, take either the hypercube [x_min, x_max] if no
            point set is provided or the bounding box of the point set if such is provided.
        point_vec (int):
            vector with coordinates of the data points. Shape: (npoints, ndim)
        point_set (int):
            function values at the data points. Shape: (npoints,)
        scale_vec (int):
            scaling values at the data points
        extended_stats (bool):
            if true, more internal information is logged
    """

    def __init__(self,
                 comments: str = "",
                 input_file_excel: str = "",
                 input_file_csv: str = "",
                 test_func: types.FunctionType = None,
                 function_parameters: dict = None,
                 output_file_vtu: str = "",
                 name_of_data_in_vtu: str = "func",
                 x_min: npt.ArrayLike = None,
                 x_max: npt.ArrayLike = None,
                 domain_polygon: npt.ArrayLike = None,
                 point_vec: npt.ArrayLike = None,
                 point_set: npt.ArrayLike = None,
                 scale_vec: npt.ArrayLike = None,
                 extended_stats: bool = False):
        self.comments = comments
        self.input_file_excel = input_file_excel
        self.input_file_csv = input_file_csv
        self.test_func = test_func
        self.function_parameters = function_parameters
        self.output_file_vtu = output_file_vtu
        self.name_of_data_in_vtu = name_of_data_in_vtu
        self.x_min = x_min
        self.x_max = x_max
        self.domain_polygon = domain_polygon
        self.point_vec = point_vec
        self.point_set = point_set
        self.scale_vec = scale_vec
        self.extended_stats = extended_stats

    def __str__(self) -> str:
        result = (f'The parameters of class ProblemDescr have the following values:\n'
                  f'comments = {self.comments}\n'
                  f'input_file_excel = {self.input_file_excel}\n'
                  f'input_file_csv = {self.input_file_csv}\n'
                  f'test_func = {self.test_func}\n'
                  f'function_parameters = {self.function_parameters}\n'
                  f'output_file_vtu = {self.output_file_vtu}\n'
                  f'name_of_data_in_vtu = {self.name_of_data_in_vtu}\n'
                  f'x_min = {self.x_min}\n'
                  f'x_max = {self.x_max}\n'
                  f'domain_polygon = {self.domain_polygon}\n'
                  f'point_vec = {self.point_vec}\n'
                  f'points_set = {self.point_set}\n'
                  f'scale_vec = {self.scale_vec}\n'
                  f'extended_stats = {self.extended_stats}\n')
        return result

    def __repr__(self) -> str:
        report = (f' ProblemDescr({self.comments}, {self.input_file_excel}, {self.input_file_csv}, {self.test_func}, '
                  f'{self.function_parameters}, {self.output_file_vtu}, {self.name_of_data_in_vtu},{self.x_min},'
                  f' {self.x_max}, {self.domain_polygon}, {self.point_vec}, {self.point_set}, {self.scale_vec}, '
                  f'{self.extended_stats}) ')
        return report
