"""
Most of these utilities are from the pysfm project.
"""

import numpy as np

from algebra import skew


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


