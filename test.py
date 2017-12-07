
from os.path import join as pjoin

import unittest

from problem import BALBundleAdjustmentProblem
from scipy_ba import solve, TransformMode


class TestBundleAdjustment(unittest.TestCase):

    def test_reparameterization(self):
        """Ensure the BA works when the camera pose is reparameterized.

        Notes
            Most of the literature assumes a camera is expressed as its own
            pose, not as a point transformation. That is, to transform a 3D
            world point X into a point P in the camera coordinate frame, one
            would need to compute
                P = R'(X - t).

            However, the conventions used in the BAL dataset assume a model
            which transforms points as
                P = RX + t.

            For solving the problem elegantly using analytical (read:
            hand-coded), we need to reparameterize the camera vectors. This test
            ensures the problem is still solved OK when this reparameterization
            is applied.

            To this end, we leverage numerical jacobians estimated using finite
            differences, instead of the analytical formulation, which is only
            available for the reparameterized (aka canonical), and not for the
            BAL dataset formulation.
        """
        # TODO(andrei): Consider looping over, say, 1--20 frame windows in the dataset.
        data_fpath = pjoin("data", "small", "problem-49-7776-pre.txt")

        # TODO(andrei): Make these parameteric tests to track progress better.
        for max_frames in [3, 5, 10, 15]:
            print("Testing on subset of {} frames.".format(max_frames))
            problem_bal = BALBundleAdjustmentProblem(
                "LadyBug",
                data_fpath,
                load_params={'max_frames': max_frames, 'canonical_rots': False})
            problem_bal_reparam = BALBundleAdjustmentProblem(
                "LadyBug",
                data_fpath,
                load_params={'max_frames': max_frames, 'canonical_rots': True})

            solver_args = {
                'plot_results': False,
                'analytic_jacobian': False
            }

            result = solve(problem_bal, transform_mode=TransformMode.BAL, **solver_args)
            result_reparam = solve(problem_bal_reparam,
                                   transform_mode=TransformMode.CANONICAL, **solver_args)
            delta_error = abs(result.cost - result_reparam.cost)

            # TODO(andrei): Maybe set based on frame count?
            error_eps = 1e+2

            self.assertLess(delta_error, error_eps,
                            "The gap between the final cost of the optimized "
                            "original problem and the optimized reparameterized "
                            "problem should be less than episilon = {}.".format(
                                error_eps))
