import os

from scipy import io as sio

from lie import SO3
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


if __name__ == '__main__':
    import unittest
    unittest.main()
