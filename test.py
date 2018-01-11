
from os.path import join as pjoin

import unittest

from problem import BALBundleAdjustmentProblem
from scipy_ba import solve, TransformMode

# TODO(andrei): Test to error out when non-canonical formulation + analytic J.
# TODO(andrei): Also check actual final parameters.
# TODO(andrei): Option to check loss over time for both compared methods.



ladybug_49_data_fpath = pjoin("data", "small", "problem-49-7776-pre.txt")
trafalgar_21_data_fpath = pjoin("data", "small", "problem-21-11315-pre.txt.bz2")

def get_ladybug(fpath, **kw):
    return BALBundleAdjustmentProblem("LadyBug", fpath, load_params=kw)


# Using explicit 3D rotation matrix for make_canonical:
# 3: 0.0030
# 5: 0.0292
# 10: 0.0006
# 15: 0.4046
# -1 (still fails with big gap)
#
# Using 'rotate':
#
# 3: 0.0287
# 5: 0.0223
# 10: 2597 (!)
# 15: 98
# 25: 122 (but the second version (reparam'd) was better in cost!)

class TestReparameterization(unittest.TestCase):

    def test_reparameterization_03(self):
        self._test_reparameterization_impl(3)

    def test_reparameterization_05(self):
        self._test_reparameterization_impl(5)

    def test_reparameterization_10(self):
        self._test_reparameterization_impl(10)

    def test_reparameterization_15(self):
        self._test_reparameterization_impl(15)

    def test_reparameterization_25(self):
        self._test_reparameterization_impl(25)

    def test_reparameterization_full(self):
        self._test_reparameterization_impl(-1)

    def _test_reparameterization_impl(self, max_frames):
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
            hand-coded), we need to reparametrise the camera vectors. This test
            ensures the problem is still solved OK when this reparametrisation
            is applied.

            To this end, we leverage numerical jacobians estimated using finite
            differences, instead of the analytical formulation, which is only
            available for the reparameterized (aka canonical), and not for the
            BAL dataset formulation.
        """
        # TODO(andrei): Consider looping over, say, 1--20 frame windows in the dataset.
        print("Testing on subset of {} frames.".format("all" if max_frames == -1
                                                       else max_frames))
        problem_bal = get_ladybug(ladybug_49_data_fpath,
                                  max_frames=max_frames,
                                  canonical_rots=False)
        problem_bal_reparam = get_ladybug(ladybug_49_data_fpath,
                                          max_frames=max_frames,
                                          canonical_rots=True)
        solver_args = {
            'plot_results': False,
            'analytic_jacobian': False,
            # Ensures an upper bound on the test run time.
            # TODO(andrei): Reparameterized values take longer to converge? Or
            # at least more fevals?
            'max_nfev': 50,
        }
        result = solve(problem_bal, transform_mode=TransformMode.BAL, **solver_args)
        result_reparam = solve(problem_bal_reparam,
                               transform_mode=TransformMode.CANONICAL, **solver_args)
        delta_error = abs(result.cost - result_reparam.cost)

        # This essentially translates to, on average 0.001 pixels of error per
        # 2D point coordinate.
        error_eps = 1e-3 * problem_bal.get_2d_point_count()

        self.assertLess(delta_error, error_eps,
                        "The gap between the final cost of the optimized "
                        "original problem and the optimized reparametrized "
                        "problem should be less than episilon = {}.".format(
                            error_eps))
        print("(no-rep vs. rep) OK with error {:.4f} <= {:.4f} (epsilon)".format(
            delta_error, error_eps))


class TestAnalyticalJacobian(unittest.TestCase):
    # TODO(andrei): Write dedicated test which should work with the scipy
    # numerical differentiation tools.

    def test_small(self):
        # As of December 08 (evening), this seems to work ok for 10 frames or
        # so, i.e., the solutions of analytic and numeric jacobians are similar,
        # but for 20 frames the ana solution starts falling behind, failing the
        # test by a large margin, being substantially worth than the numerical
        # approximation.
        max_frames = 5

        def get_dataset():
            # Rotation errors much bigger on this one as of Dec 14 2017.
            # return get_ladybug(ladybug_49_data_fpath,
            #                    max_frames=max_frames,
            #                    canonical_rots=True)
            # Non-sequential datasets are bad because it means most keypoints
            # are seen even from very few frames, so numerically estimating the
            # Jacobian is very slow.
            return BALBundleAdjustmentProblem("Trafalgar 21",
                                              trafalgar_21_data_fpath,
                                              load_params={
                                                  'max_frames': max_frames,
                                                  'canonical_rots': True
                                              })

        args = {
            'plot_results': False,
            'transform_mode': TransformMode.CANONICAL,
            # TODO(andrei): Make this lower!
            'max_nfev': 15,
        }

        result_ana = solve(get_dataset(), analytic_jacobian=True, **args)
        result_num = solve(get_dataset(), analytic_jacobian=False, **args)

        delta_error = abs(result_num.cost - result_ana.cost)
        error_eps = 1e-3 * get_dataset().get_2d_point_count()

        self.assertLess(delta_error, error_eps,
                        "The gap between the final cost of the problem "
                        "optimized using analytical Jacobians and the that of "
                        "the problem optimized using numerically approximated "
                        "Jacobians should be less than epsilon = {}.".format(
                            error_eps))
        print("(ana vs. num) OK with error {:.4f} <= {:.4f} (epsilon)".format(
            delta_error, error_eps))



