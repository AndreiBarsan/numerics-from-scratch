import sys
from abc import ABCMeta, abstractmethod

import numpy as np


class C2Function:
    """Represents a twice continuously differentiable function."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def val(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def hessian(self, x):
        pass


def rosenbrock_sep(x, y):
    """Used for plotting the contour lines."""
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


class Rosenbrock(C2Function):
    """The Rosenbrock "banana" function.

    A classic benchmark for optimization algorithms.
    """

    def val(self, x):
        """Returns the scalar value of the function."""
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def __call__(self, *args, **kwargs):
        return self.val(*args)

    def gradient(self, x):
        """Returns a [2x1] gradient vector."""
        return np.array([
            [-400 * x[0] * x[1] + 400 * x[0] * x[0] * x[0] - 2 + 2 * x[0]],
            [ 200 * x[1] - 200 * x[0] * x[0]]
        ]).reshape(2, 1)

    def hessian(self, x):
        """Returns a [2x2] Hessian matrix."""
        res = np.array([
            [-400 * x[1] + 1200 * x[0] * x[0] + 2, -400 * x[0]],
            [-400 * x[0],                           [200.0]]
        ]).reshape(2, 2)

        sys.stdout.flush()
        return res
