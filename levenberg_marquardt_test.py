
import unittest

import numpy as np

from levenberg_marquardt import powell, levenberg_marquardt, scipy_lm, \
    model_fitting_wrapper


# TODO(andrei): Consider unifying with your mega-benchmark which includes TF
# and Ceres.
class TestCustomLM(unittest.TestCase):

    def _compare(self, f, J, x0, lm_args=None):
        if lm_args is None:
            # Ensure you don't use a mutable default argument, because Python.
            lm_args = {}
        x_star, steps, _ = levenberg_marquardt(
            f, J, x0,
            custom_solver=False,
            **lm_args)
        x_star_custom_solver, steps_custom_sovler, _ = levenberg_marquardt(
            f, J, x0,
            custom_solver=False,
            **lm_args)
        sp_opt_result = scipy_lm(f, J, x0, max_nfev=100)
        x_star_scipy = sp_opt_result.x
        steps_scipy = sp_opt_result.nfev

        print(sp_opt_result)

        self.assertLess(np.linalg.norm(x_star - x_star_custom_solver),
                        1e-10,
                        "Using the custom cholesky_solve implementation "
                        "should not significantly impact accuracy.")
        self.assertLess(np.linalg.norm(x_star - x_star_scipy),
                        1e-2,   # TODO fix this; it's TOO lenient!
                        "The stock cholesky_solve method should not be very "
                        "different from the scipy solution.")
        self.assertLess(np.linalg.norm(x_star_custom_solver - x_star_scipy),
                        1e-2,
                        "The custom cholesky solve method should not be very "
                        "different from the scipy solution.")
        # TODO-LOW(andrei): check number of steps, too

    def test_toy_powell(self):
        f, J, name = powell()
        x0 = np.array([3, 1])
        self._compare(f, J, x0, lm_args={
            'convergence_epsilon_g': 1e-15,
            'convergence_epsilon_h': 1e-15,
            'tau': 1.0
        })

    def test_simple_residual(self):
        (f_fit, J_fit, name), x_gt = model_fitting_wrapper()
        x0 = np.array([-1.0, -2.0, 1.0, -1.0])
        self._compare(f_fit, J_fit, x0, lm_args={
            'convergence_epsilon_g': 1e-8,
            'convergence_epsilon_h': 1e-8,
            'tau': 1e-3
        })

    # def test_bundle_adjustment(self):
    #     self.fail("Not yet implemented!")
