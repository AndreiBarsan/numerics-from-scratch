"""Simple gradient descent implementation."""

import numpy as np
import scipy.linalg as spla

import matplotlib.pyplot as plt
import sys

from abc import ABCMeta, abstractmethod


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


def l2_norm(x):
    return np.linalg.norm(x, ord=2)


class OptimizationResults:
    """Logs detailed information about an optimization procedure.

    Attributes
        known_optimum: The a priori known value of the global optimum of the
                       function being optimized.
        iterates: The parameter values over time.
        values: The function values over time.
        ratios: This and quad_ratios correspond to Equation (1) from the CSC2305
                Assignment 03 handout.
        quad_ratios: See above.
    """

    def __init__(self, known_optimum, x_0, f_0, norm=l2_norm):
        self.known_optimum = known_optimum
        self.norm = norm
        self.iterates = [x_0]
        self.values = [f_0]
        self.ratios = []
        self.quad_ratios = []

    def record(self, iterate, value):
        if len(self.iterates) == 0:
            raise ValueError("record() must be called after x_0 and f_0 are "
                             "recorded.")

        previous = self.iterates[-1]
        current_gap = self.norm(iterate - self.known_optimum)
        previous_gap = self.norm(previous - self.known_optimum)

        ratio = current_gap / previous_gap
        ratio_quad = current_gap / (previous_gap ** 2)

        self.ratios.append(ratio)
        self.quad_ratios.append(ratio_quad)
        self.iterates.append(iterate)
        self.values.append(value)


# TODO(andreib): Document.
def backtracking_line_search(func, x, alpha_0, p, c, rho):
    alpha = alpha_0
    f_x = func(x)
    dot_grad_p = np.dot(func.gradient(x).T, p)[0, 0]

    it = 0
    while True:
        lhs = func(x + alpha * p)
        rhs = f_x + c * alpha * dot_grad_p
        if lhs[0] <= rhs[0]:
            break
        alpha = rho * alpha
        it += 1
        if it > 100:
            raise ValueError("Exceeded max number of BTLS iterations")

    return alpha

# TODO(andreib): Organize code into "optimizer" class, and have a generic
# "optimize" method. This


def steepest_descent(func, x_0, alpha_0, f_star_gt):
    iteration = 0
    x = x_0
    convergence_epsilon = 1e-8
    step_size = alpha_0

    results = OptimizationResults(
        x_0=x_0,
        f_0=func(x_0),
        known_optimum=f_star_gt)

    while True:
        iteration += 1

        grad = func.gradient(x)
        direction = -grad
        # Note: c=0.6 works well for Newton.
        # Note: c=0.1, rho=0.5 is fast for SD with cold-start bt!! (should try a grid search,
        # comparing different param values with/wo cold start in terms of it
        # count *and* of wall time).
        step_size = backtracking_line_search(func, x, alpha_0, direction, c=0.1, rho=0.5)
        # step_size = 0.001

        x_next = x + step_size * direction

        # TODO(andreib): Gradient and hessian check!
        if np.linalg.norm(x - x_next) < convergence_epsilon:
            print("Converged in {} iterations.".format(iteration))
            break

        f_next = func(x_next)[0]
        if np.isnan(f_next):
            raise ValueError("Something blew up!")
        results.record(x_next, f_next)

        if (iteration + 1) % 1 == 0:
            print("Iteration {} | fval = {:.4f}".format(iteration, f_next))

        x = x_next

    return x, results


def newton(func, x_0, alpha_0, f_star_gt):
    iteration = 0
    x = x_0
    convergence_epsilon = 1e-8
    step_size = alpha_0

    results = OptimizationResults(
        x_0=x_0,
        f_0=func(x_0),
        known_optimum=f_star_gt)

    while True:
        iteration += 1

        # TODO compute with line search
        hval = func.hessian(x)
        gval = func.gradient(x)

        direction = -np.dot(np.linalg.inv(hval), gval)
        step_size = backtracking_line_search(func, x, step_size, direction, c=0.6, rho=0.9)

        x_next = x + step_size * direction

        # TODO(andreib): Gradient and hessian check!
        if np.linalg.norm(x - x_next) < convergence_epsilon:
            print("Converged in {} iterations.".format(iteration))
            break

        f_next = func(x_next)[0]
        results.record(x_next, f_next)

        if (iteration + 1) % 1 == 0:
            print("Iteration {} | fval = {:.4f}".format(iteration, f_next))

        x = x_next

    return x, results


def f(x, y):
    # return rosenbrock(x, y)
    pass
    # return np.cos(np.hypot(x, y)) + np.sin(np.hypot(x + 5, y + 5))


def main():
    samples = 500
    contour_count = 25
    xlim = [-2.0, 2.0]
    ylim = [-2.0, 4.0]
    x = np.linspace(xlim[0], xlim[1], samples)
    y = np.linspace(ylim[0], ylim[1], samples)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_sep(X, Y)
    func = Rosenbrock()
    x_star_gt = np.array([1.0, 1.0]).reshape((2, 1))
    f_star_gt = func(x_star_gt)
    x_0_easy = np.array([1.2, 1.2]).reshape(2, 1)
    x_0_hard = np.array([-1.2, 1.0]).reshape(2, 1)
    x_0 = x_0_hard

    contour_vals = np.linspace(0.0, np.max(Z) * 0.25, contour_count)
    # contour_vals = np.linspace(1.0, np.sqrt(np.max(Z) * 0.75), contour_count) ** 2
    cont = plt.contour(X, Y, Z, contour_vals)
    plt.colorbar(cont)
    plt.scatter(x_star_gt[0], x_star_gt[1], s=100, marker='*')

    # TODO(andreib): Pretty plot starting point with label (easy/hard).

    x_final_n, results_n = newton(func, x_0=x_0_easy, alpha_0=1.0, f_star_gt=f_star_gt)
    plot_iterates(results_n, color='k', label='Newton, easy')
    x_final_n, results_n = newton(func, x_0=x_0_hard, alpha_0=1.0, f_star_gt=f_star_gt)
    plot_iterates(results_n, color='y', label='Newton, hard')

    x_final_sd_e, results_sd_e = steepest_descent(func, x_0=x_0_easy, alpha_0=1.0, f_star_gt=f_star_gt)
    plot_iterates(results_sd_e, color='b', stride=25, label='SD, easy (subsampled iterates)')
    x_final_sd_h, results_sd_h = steepest_descent(func, x_0=x_0_hard, alpha_0=1.0, f_star_gt=f_star_gt)
    plot_iterates(results_sd_h, color='r', stride=25, label='SD, hard (subsampled iterates)')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.show()


def plot_iterates(results, label, color, stride=1):
    """Plots the iterates from the result onto the current plot."""
    if len(results.iterates) <= 1:
        raise ValueError("Insufficient iterates to plot!")
    its_np = np.squeeze(np.array(results.iterates))
    x_prev, y_prev = its_np[0, 0], its_np[0, 1]
    for (x, y) in its_np[1::stride, :]:
        plt.arrow(x_prev, y_prev, x - x_prev, y - y_prev,
                  head_width=0.05, head_length=0.10, color=color)
        x_prev, y_prev = x, y
    # Hack to ensure there is a legend entry for the manually drawn iterates.
    plt.scatter([-1000], [-1000], color=color, label=label)


if __name__ == '__main__':
    # mayavi_demo()
    main()
