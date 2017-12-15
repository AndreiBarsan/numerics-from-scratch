import os

import numpy as np

from scipy import io as sio

from lie import rotate, SO3
from numpy_test import NumpyTestCase


class SO3Test(NumpyTestCase):
    """
    Notes
        Ground truth results generated using MATLAB's 'vrrotvec2mat' function.
    """

    def test_axis_aligned_twists(self):
        samples = [
            ("x twist",
             [1, 0, 0],
             [[1.0, 0, 0],
             [0.0, 0.5403, -0.8415],
             [0.0, 0.8415, 0.5403]]
            ),
            ("y twist",
             [0, 1, 0],
             [[0.5403, 0, 0.8415],
              [0, 1.0000, 0],
              [-0.8415, 0, 0.5403]]
             ),
            ("z twist",
             [0, 0, 1],
             [[0.5403, -0.8415, 0],
              [0.8415, 0.5403, 0],
              [0, 0, 1.0000]
             ])
        ]

        for name, twist, expected_rotation_matrix in samples:
            self.assertArrayEqual(SO3.exp(twist),
                                  expected_rotation_matrix,
                                  msg=name)

    def test_random_twists(self):
        test_data_root = os.path.join('.', 'data', 'test')
        twists = sio.loadmat(os.path.join(test_data_root,
                                            'so3_twist.mat'))['test_twists']
        rots = sio.loadmat(os.path.join(test_data_root, 'so3_rot.mat'))[
            'test_matrices']

        for i, (twist, expected_rotation_matrix) in enumerate(zip(twists, rots)):
            # print("Sample SO conversion {}/{}...".format(i+1, len(twists)), end='')
            self.assertArrayEqual(SO3.exp(twist),
                                  expected_rotation_matrix,
                                  msg="random-example-{:04d}".format(i))
            # print(" OK!")

    def test_rotate_helper(self):
        """Compare the 'rotate' utility from the SciPy cookbook to computing
        an explicit matrix exponentiation and using that to rotate a vector."""
        np.random.seed(12345632)

        cases = 20000

        for i in range(cases):
            # TODO(andrei): Is the range sane?
            vec = np.random.uniform(-100, 100, (3, 1))
            rot_axis = np.random.uniform(-100, 100, (3, 1))
            rot_axis /= np.linalg.norm(rot_axis)

            rot_angle = np.random.uniform(-np.pi, np.pi)
            rot = rot_axis * rot_angle

            result_direct = rotate(vec.T, rot.T).T

            R = SO3.exp(rot.ravel())
            result_via_R = np.dot(R, vec)

            self.assertArrayEqual(
                result_direct,
                result_via_R,
                msg="Rotating a vector using the implemented Rodrigues formula "
                    "should be the same as computing the rotation matrix of "
                    "axis-angle representation and using that to transform the "
                    "vector.")

            res_inv_direct = rotate(vec.T, -rot.T).T
            result_inv_via_R = np.dot(R.transpose(), vec)
            self.assertArrayEqual(res_inv_direct, result_inv_via_R)

            self.assertArrayEqual(
                SO3.exp(rot.ravel()).transpose(),
                SO3.exp(-rot.ravel())
            )

        print("Tested {} random rotation cases.".format(cases))




if __name__ == '__main__':
    import unittest
    unittest.main()
