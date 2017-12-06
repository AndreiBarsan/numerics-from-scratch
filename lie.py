"""
Most of these utilities are from the pysfm project.
"""
import os

import numpy as np
from scipy import io as sp_io
from numpy_test import NumpyTestCase

def skew(m):
    """Compute the skew-symmetric matrix for m.
    From the pysfm package.
    """
    m = np.asarray(m)
    assert m.shape == (3,)
    return np.array([[  0,    -m[2],  m[1] ],
                     [  m[2],  0,    -m[0] ],
                     [ -m[1],  m[0],  0.   ]])


class SO3(object):
    """Used to compute the mapping from a twist to SO(3) """
    @classmethod
    def exp(cls, twist):
        twist = np.asarray(twist)
        assert np.shape(twist) == (3,), 'shape was ' + str(np.shape(twist))

        t = np.linalg.norm(twist)
        if t < 1e-8:
            return np.eye(3)   # exp(0) = I

        twist_cross = skew(twist)
        A = np.sin(t)/t
        B = (1. - np.cos(t)) / (t*t)
        I = np.eye(3)
        return I + A * twist_cross + B * np.dot(twist_cross, twist_cross)

    # Compute jacobian of exp(m)*x with respect to m, evaluated at
    # m=[0,0,0]. x is assumed constant with respect to m.
    @classmethod
    def J_expm_x(cls, x):
        return skew(-x)

    # Return the generators times x
    @classmethod
    def generator_field(cls, x):
        return skew(x)


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
        twists = sp_io.loadmat(os.path.join(test_data_root,
                                            'so3_twist.mat'))['test_twists']
        rots = sp_io.loadmat(os.path.join(test_data_root, 'so3_rot.mat'))[
            'test_matrices']

        for i, (twist, expected_rotation_matrix) in enumerate(zip(twists, rots)):
            print("Sample SO conversion {}/{}...".format(i+1, len(twists)), end='')
            self.assertArrayEqual(SO3.exp(twist),
                                  expected_rotation_matrix,
                                  msg="random-example-{:04d}".format(i))
            print(" OK!")


if __name__ == '__main__':
    import unittest
    unittest.main()
