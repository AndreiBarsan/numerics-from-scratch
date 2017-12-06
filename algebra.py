import numpy as np


def skew(m):
    """Compute the skew-symmetric matrix for m.
    From the pysfm package.
    """
    m = np.asarray(m)
    assert m.shape == (3,)
    return np.array([[  0,    -m[2],  m[1] ],
                     [  m[2],  0,    -m[0] ],
                     [ -m[1],  m[0],  0.   ]])
