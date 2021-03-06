"""Steepest descent, Newton, and BFGS implementations with line search methods.

References
    [NW] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization: Springer
    Series in Operations Research and Financial Engineering. Springer.
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np

# List of things which could be improved:
#
# TODO(andrei): XXX This is important! There is a bug in the ratio computation
# code. You are passing f* instead of x*, meaning that your ratios don't go to
# infinity as you're approaching the solution. (I lost 2 marks in the assignment
# because of this.)
# TODO-LOW(andrei): Organize code into "optimizer" class, and have a generic
# "optimize" method. (TF and friends model)
# TODO-LOW(andrei): Fewer gradient evaluations in the BFGS function.
from csc2305.functions import Rosenbrock
from csc2305.optimization_results import OptimizationResults
from csc2305.utils import is_spd, plot_rosenbrock_contours, \
    plot_iterates, output_table


def backtracking_line_search(func, x, alpha_0, p, c, rho):
    """Implements Algorithm 3.1 from [NW]."""
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


def get_phi(func, x, p):
    def phi(alpha):
        return func.val(x + alpha * p)

    def phi_prime(alpha):
        f_grad = func.gradient(x + alpha * p)
        assert f_grad.shape == (2, 1)
        res = np.dot(f_grad.ravel(), p)
        assert res.shape == (1,)
        return res

    return phi, phi_prime


def poor_mans_interpolate(a_lo, a_hi):
    """Confirmed by the Professor as also acceptable.

    And easier to debug...
    """
    return a_lo + (a_hi - a_lo) / 2.0


def interpolate(func, x, p, a_lo, a_hi, c1):
    """Finds a trial step length in [a_lo, a_hi].
    Not in use at the moment, since the simpler version was flagged as acceptable.

    Based on the methods from Chapter 3 of [NW].
    """
    phi, phi_prime = get_phi(func, x, p)

    phi_prime_0 = phi_prime(a_lo)
    phi_alpha_0 = phi(a_hi)
    phi_0 = phi(a_lo)

    def check_alpha(aaa):
        return phi(aaa) <= phi_0 + c1 * aaa * phi_prime_0

    if check_alpha(a_hi):
        print("No interpolation needed. Returning alpha_0 = {}.".format(a_hi))
        return a_hi

    # Perform quadratic interpolation and check
    num = phi_prime_0 * a_hi * a_hi
    denom = 2 * (phi_alpha_0 - phi_0 - phi_prime_0 * a_hi)
    a_1 = -num / denom

    if check_alpha(a_1):
        print("Quadratic interpolation passed.")
        return a_1
    else:
        # we may be able to get by with just this
        print("Quadratic interpolation failed, but returning a_1 as a trial anyway.")
        return a_1

    # Perform cubic interpolation check
    # coef = 1.0 / (a_hi * a_hi * a_1 * a_1 * (a_1 - a_hi))
    # m_1 = np.array([
    #     [a_hi * a_hi, -(a_1 * a_1)],
    #     [-(a_hi * a_hi * a_hi), a_1 * a_1 * a_1]
    # ])
    # m_2 = np.array([
    #     phi(a_1) - phi_0 - phi_prime_0 * a_1,
    #     phi_alpha_0 - phi_0 - phi_prime_0 * a_hi
    # ])
    # res = coef * np.dot(m_1, m_2)
    # assert res.shape == (2,)
    #
    # a = res[0]
    # b = res[1]
    # assert c1.shape == (2, 2)
    # a_2 = (-b + math.sqrt(b * b - 3 * a * phi_prime_0)) / (3 * a)
    #
    # if check_alpha(a_2):
    #     print("Cubic interpolation passed.")
    #     return a_2
    # else:
    #     print("Cubic interpolation FAILED. Iterating.")
    #     raise ValueError("Looping not supported.")


def zoom(func, x, p, a_lo, a_hi, c1, c2):
    """Used by 'wolfe_search' to find an acceptable range for alpha.

    Algorithm 3.6 from [NW]."""
    print("zoom()")
    assert 0.0 < c1 < c2 < 1.0, "Wolfe constant validation"

    max_it = 100
    it = 0

    while True:
        print("Zoom iteration {:04d}: alpha in [{:.6f}, {:.6f}]".format(it, a_lo, a_hi))
        it += 1
        if it >= max_it:
            raise ValueError("Maximum zoom() iteration ({}) reached!".format(max_it))

        phi, phi_prime = get_phi(func, x, p)
        a_j = poor_mans_interpolate(a_lo, a_hi)

        if phi(a_j) > phi(0.0) + c1 * a_j * phi_prime(0.0) or phi(a_j) >= phi(a_lo):
            a_hi = a_j
        else:
            phi_prime_a_j = phi_prime(a_j)
            if math.fabs(phi_prime_a_j) <= -c2 * phi_prime(0.0):
                print("Zoom found good alpha = {:.4f} (check A, it = {})".format(a_j, it))
                return a_j

            if phi_prime_a_j * (a_hi - a_lo) >= 0.0:
                a_hi = a_lo

            a_lo = a_j


def wolfe_search(func, x, p, alpha_0, c1, c2, **kw):
    """Implements the Wolfe Conditions-based line search method from [NW].

    Algorithm 3.5 from [NW].
    """
    a_0 = 0.0
    max_its = 10        # As suggested on p.62 from [NW]
    phi, phi_prime = get_phi(func, x, p)
    phi_0 = phi(0.0)
    phi_prime_0 = phi_prime(0.0)

    i = 1
    a_growth_factor = kw.get('a_growth_factor', 10.0)
    a_i = a_1 = kw.get('a_1', 0.01)
    a_prev = a_0
    verbose = kw.get('verbose', False)

    while True:
        if verbose: print("Wolfe search it {}. a_{} = {:.4f}".format(i, i, a_i))
        phi_a_i = phi(a_i)

        if phi_a_i > phi_0 + c1 * a_i * phi_prime_0 or (phi(a_i) >= phi(a_prev) and i > 1):
            if verbose: print("wolfe_search: returning zoom (A) at iteration {}".format(i))
            return zoom(func, x, p, a_prev, a_i, c1, c2)

        phi_prime_a_i = phi_prime(a_i)

        if math.fabs(phi_prime_a_i) <= -c2 * phi_prime_0:
            if verbose: print("wolfe_search: returning a_i = {} at iteration {}".format(a_i, i))
            return a_i

        if phi_prime_a_i >= 0.0:
            if verbose: print("wolfe_search: returning zoom (B) at iteration {}".format(i))
            return zoom(func, x, p, a_i, a_prev, c1, c2)

        i += 1
        if i >= max_its:
            if verbose: print("wolfe_search: WARNING maximum number of "
                              "iterations reached. Returning possibly "
                              "suboptimal a_i = {}".format(a_i))
            return a_i

        a_prev = a_i
        a_i *= a_growth_factor
        # Implicitly, a_max is a_i * 2^{max_its}.


def steepest_descent(func, x_0, alpha_0, f_star_gt):
    """Implements the steepest descent method introduced in [NW] Chapter 2."""
    iteration = 0
    x = x_0
    convergence_epsilon = 1e-8
    step_size = alpha_0
    print_every_k = 500

    results = OptimizationResults(
        x_0=x_0,
        f_0=func(x_0)[0],
        known_optimum=f_star_gt)

    while True:
        iteration += 1

        grad = func.gradient(x)
        direction = -grad
        # Note: c=0.6 works well for Newton.
        # Note: c=0.1, rho=0.5 is fast for SD with cold-start bt!! (should try a grid search,
        # comparing different param values with/wo cold start in terms of it
        # count *and* of wall time).
        # This is a conservative version which always works and computes the
        # step size from scratch.
        # step_size = backtracking_line_search(func, x, alpha_0, direction, c=0.1, rho=0.5)
        # This version is hand-tuned and works with warm step size values.
        step_size = backtracking_line_search(func, x, min(2.0, step_size * 10.0), direction, c=0.5, rho=0.5)

        x_next = x + step_size * direction

        if np.linalg.norm(x - x_next) < convergence_epsilon:
            if not np.allclose(grad, np.zeros_like(grad), atol=1e-4):
                raise ValueError("Converged to non-critical point!")
            if not is_spd(func.hessian(x)):
                raise ValueError("Converged to non-minimum!")

            print("Steepest descent converged in {} iterations.".format(iteration))
            break

        f_next = func(x_next)[0]
        if np.isnan(f_next):
            raise ValueError("Something blew up!")
        results.record(x_next, f_next, step_size)

        if (iteration + 1) % print_every_k == 0:
            print("Iteration {} | fval = {:.4f}".format(iteration, f_next))

        x = x_next

    return x, results


def newton(func, x_0, alpha_0, f_star_gt):
    """Implements the Newton method introduced in [NW] Chapter 2."""
    iteration = 0
    x = x_0
    convergence_epsilon = 1e-8
    step_size = alpha_0

    results = OptimizationResults(
        x_0=x_0,
        f_0=func(x_0)[0],
        known_optimum=f_star_gt)

    while True:
        iteration += 1

        hval = func.hessian(x)
        gval = func.gradient(x)

        direction = -np.dot(np.linalg.inv(hval), gval)
        step_size = backtracking_line_search(func, x, step_size, direction, c=0.6, rho=0.9)

        x_next = x + step_size * direction

        # TODO-LOW(andreib): Gradient and hessian check for extra sanity.
        if np.linalg.norm(x - x_next) < convergence_epsilon:
            print("Converged in {} iterations.".format(iteration))
            break

        f_next = func(x_next)[0]
        results.record(x_next, f_next, step_size)

        # if (iteration + 1) % 1 == 0:
        #     print("Newton iteration {:03d} | fval = {:.4f} ".format(iteration, f_next))

        x = x_next

    return x, results


def bfgs(func, x_0, alpha_0, f_star_gt):
    """Implements the quasi-newton BFGS optimization method from [NW]."""
    iteration = 0
    x = x_0
    convergence_epsilon = 1e-8

    # The approximate Hessian
    B = np.eye(x_0.shape[0])

    results = OptimizationResults(
        x_0=x_0,
        f_0=func(x_0)[0],
        known_optimum=f_star_gt)

    while True:
        iteration += 1

        gval = func.gradient(x)
        direction = -np.dot(np.linalg.inv(B), gval)
        step_size = wolfe_search(func, x, direction, alpha_0, c1=0.1, c2=0.9)
        x_next = x + step_size * direction

        if np.linalg.norm(x - x_next) < convergence_epsilon:
            print("Converged in {} iterations.".format(iteration))
            break

        f_next = func(x_next)[0]
        results.record(x_next, f_next, step_size)

        # Update the Hessian approximation
        s = (x_next - x).ravel()
        y = (func.gradient(x_next) - gval).ravel()
        B_s = np.dot(B, s)
        # Implements equation (2.19) from [NW].
        T_1 = np.outer(B_s, B_s) / np.dot(s, np.dot(B, s))
        T_2 = np.outer(y, y) / np.dot(y, s)
        B = B - T_1 + T_2
        assert T_1.shape == (2, 2)
        assert T_2.shape == (2, 2)
        assert B.shape == (2, 2)

        # Check one of the necessary conditions for B's positive definitiveness.
        if np.dot(y, s) <= -1e-4:
            raise ValueError("Secant condition check failed at iteration {}.".format(iteration))

        # Explicitly check B is SPD (redundant, for extra safety).
        if not is_spd(B):
            raise ValueError("Hessian approximation became non-SPD at iteration {}!".format(iteration))

        if (iteration + 1) % 1 == 0:
            print("Iteration {} | fval = {:.6f}".format(iteration, f_next))
            # fro_residual = np.linalg.norm(B - hessian, ord='fro')
            # l2_residual = np.linalg.norm(B - hessian, ord=2)
            # hessian = func.hessian(x)
            # check = np.linalg.norm(np.dot((B - hessian), direction)) / np.linalg.norm(direction)
            # # print("Frobenius norm of (B - H) = {:.6f}".format(fro_residual))
            # # print("l2 norm of (B - H)        = {:.6f}".format(l2_residual))
            # print("ratio               = {:.6}".format(check))

        # At this point we are confident that the step is correct, so we finally
        # update the iterate.
        x = x_next

    # Returns the final value of x and the summary object.
    return x, results


def main():
    samples = 500
    contour_count = 25
    xlim = [-2.0, 2.0]
    ylim = [-0.5, 3.0]
    plt.figure(0)
    plot_rosenbrock_contours(contour_count, samples, xlim, ylim)

    x_0_easy = np.array([1.2, 1.2]).reshape(2, 1)
    x_0_hard = np.array([-1.2, 1.0]).reshape(2, 1)
    func = Rosenbrock()
    x_star_gt = np.array([1.0, 1.0]).reshape((2, 1))
    f_star_gt = func(x_star_gt)
    plt.scatter(x_star_gt[0], x_star_gt[1], s=100, marker='*')

    # Steepest Descent
    f0 = plt.figure(0)
    plt.xlim(xlim)
    plt.ylim(ylim)

    start = time.time()
    x_final_sd_e, results_sd_e = steepest_descent(func, x_0=x_0_easy, alpha_0=1.0, f_star_gt=f_star_gt)
    delta_s = time.time() - start
    print("Table for steepest descent easy")
    output_table(results_sd_e)
    plot_iterates(results_sd_e, delta_s, color='b', stride=25, label='SD, easy')
    start = time.time()
    x_final_sd_h, results_sd_h = steepest_descent(func, x_0=x_0_hard, alpha_0=1.0, f_star_gt=f_star_gt)
    print("Table for steepest descent hard")
    output_table(results_sd_h)
    delta_s = time.time() - start
    plot_iterates(results_sd_h, delta_s, color='r', stride=25, label='SD, hard')
    plt.legend()
    plt.savefig('conv.eps')

    # TODO(andreib): Pretty plot starting point with label (easy/hard).
    # Newton and Quasi-Newton
    f1 = plt.figure(1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plot_rosenbrock_contours(contour_count, samples, xlim, ylim)
    start = time.time()
    x_final_n, results_n = newton(func, x_0=x_0_easy, alpha_0=1.0, f_star_gt=f_star_gt)
    delta_s = time.time() - start
    plot_iterates(results_n, delta_s, color='k', label='Newton, easy')
    print("Table for newton easy")
    output_table(results_n)

    start = time.time()
    x_final_n, results_n = newton(func, x_0=x_0_hard, alpha_0=1.0, f_star_gt=f_star_gt)
    delta_s = time.time() - start
    plot_iterates(results_n, delta_s, color='y', label='Newton, hard')
    print("Table for newton hard")
    output_table(results_n)

    start = time.time()
    x_final_bfgs, results_bfgs = bfgs(func, x_0=x_0_easy, alpha_0=1.0, f_star_gt=f_star_gt)
    delta_s = time.time() - start
    plot_iterates(results_bfgs, delta_s, color='g', stride=5, label="BFGS, easy")
    print("Table for BFGS easy")
    output_table(results_bfgs)

    start = time.time()
    x_final_bfgs, results_bfgs = bfgs(func, x_0=x_0_hard, alpha_0=1.0, f_star_gt=f_star_gt)
    delta_s = time.time() - start
    plot_iterates(results_bfgs, delta_s, color='m', stride=5, label="BFGS, hard")
    plt.legend()
    plt.savefig('conv_second.eps')
    print("Table for BFGS hard")
    output_table(results_bfgs)

    # Iterates, hard
    plt.figure(30)
    # q_sd =
    q_n = results_n.ratios
    q_bfgs = results_bfgs.ratios
    plt.plot(np.arange(len(q_n)), q_n, 'k-', label="Ratios for Newton")
    plt.plot(np.arange(len(q_bfgs)), q_bfgs, 'r-', label="Ratios for BFGS")
    plt.xlabel("iteration (k)")
    plt.ylabel("ratio")
    plt.legend()
    plt.savefig('ratios.eps')

    plt.figure(31)
    plt.plot(np.arange(len(q_n)), results_n.quad_ratios, 'k--', label="Quadratic ratios for Newton")
    plt.plot(np.arange(len(q_bfgs)), results_bfgs.quad_ratios, 'r--', label="Quadratic ratios for BFGS")
    plt.xlabel("iteration (k)")
    plt.ylabel("quadratic ratio")
    plt.legend()
    plt.savefig('ratios_quad.eps')

    # plt.show()


if __name__ == '__main__':
    main()
