"""Implementation of the dogleg trust region optimization method.

References
    [NW] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization: Springer
    Series in Operations Research and Financial Engineering. Springer.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from csc2305.functions import Rosenbrock
from csc2305.optimizaion_results import OptimizationResults
from csc2305.utils import plot_rosenbrock_contours, plot_iterates, \
    l2_norm


def continuous_binary_search(tau_start, tau_end, p_fun, delta, max_steps):
    # Uses Lemma 4.2 to do a simple search for the optimal tau in (4.16), such
    # that the step is still within the trust region of radius delta.
    # func decreases as tau increases
    # Want smallest value of func with argument under delta.
    # I.e., since m(p_tilde(tau)) function decreases with tau, want the biggest
    # tau s.t., p(tau) < delta.

    tau_mid = (tau_end - tau_start) / 2.0
    if max_steps == 0:
        # TODO(andreib): Maybe return the min, dunno.
        return p_fun(tau_mid)

    p_mid = p_fun(tau_mid)
    if l2_norm(p_mid) < delta:
        return continuous_binary_search(tau_mid, tau_end, p_fun, delta, max_steps-1)
    else:
        return continuous_binary_search(tau_start, tau_mid, p_fun, delta, max_steps-1)


def naive_search(tau_start, tau_end, p_fun, delta, max_steps):
    # Uses Lemma 4.2 to do a simple search for the optimal tau in (4.16), such
    # that the step is still within the trust region of radius delta.

    tau_cur = tau_end
    if l2_norm(p_fun(tau_start)) >= delta:
        # TODO custom exception
        raise ValueError("Invalid dog leg selected.")

    p_start_norm = l2_norm(p_fun(tau_start))
    print("\t\tp(tau_start = {}) = {}".format(tau_start, p_start_norm))

    step = 0.05
    while True:
        p_cur = p_fun(tau_cur)
        p_cur_norm = l2_norm(p_cur)
        if p_cur_norm <= delta:
            return p_cur

        tau_cur -= step

        # TODO sometimes this fails when the optimum is riiight past tau=1.0
        if tau_cur < tau_start:
            print("\t\tStep adjustment inside the naive search.")
            tau_cur += step
            step /= 2.0
            # return p_fun(tau_start)
            # raise ValueError("You have a bug in your code.")



def dogleg_search(f, g, B, D):
    """Computes an update direction using the dogleg method.

    Solves Equation (4.3) using the dogleg method
    """
    search = naive_search
    max_steps = 10      # TODO argfy and actually use if the naive search is bad.

    # p_U is the point corresponding to the standard steepest-descent step.
    alpha = np.dot(g.T, g) / np.dot(g.T, np.dot(B, g))
    assert alpha.shape == (1, 1)
    p_U = -alpha * g
    # Note: solves with gesv, not potrs, but that's more general, so we don't
    # have to assume B is PD.
    p_B = -np.linalg.solve(B, g)

    if l2_norm(p_B) <= D:
        print("\tp_B is inside the TR. Returning directly!")
        return p_B

    if l2_norm(p_U) > D:
        # First half of the dogleg
        # p_U already outside TR. Should search for tau along the [0, 1] range.
        print("\tp_U already outside TR. Searching on first half.")

        def p_fun(tau):
            return tau * p_U
        return search(
            tau_start=0.0,
            tau_end=1.0,
            p_fun=p_fun,
            delta=D,
            max_steps=max_steps)
    else:
        # Second half of the dogleg
        # p_U inside TR. Should compute p_B and search for tau along the [1, 2]
        # range.
        print("\tp_U inside TR. Searching on second half.")

        def p_fun(tau):
            return p_U + (tau - 1) * (p_B - p_U)
        return search(
            tau_start=1.0,
            tau_end=2.0,
            p_fun=p_fun,
            delta=D,
            max_steps=max_steps)


def trust_region(func, x_0, x_gt, **kwargs):
    """Dogleg-based trust region optimization of 'func' starting from 'x_0'.

    Based on Algorithm 4.1 from [NW].

    Problem statement (exercise 4.2 from [NW]):
        Write a program that implements the dogleg method. Choose Bk to be the
        exact Hessian. Apply it to solve Rosenbrock’s function (2.22).

        Experiment with the update rule for the trust region by changing the
        constants in Algorithm 4.1, or by designing your own rules.
    """
    convergence_epsilon = kwargs.get('convergence_epsilon', 1e-4)
    iteration = 0
    max_iterations = 5000
    poor_model_threshold = 0.25
    great_model_threshold = 0.75
    assert poor_model_threshold < 1
    f_0 = func(x_0)

    # TODO experiment with these values a LOT
    D_max = 0.75
    D_0 = 0.25
    # eta in the textbook
    update_threshold = 0.20

    x = x_0
    D = D_0
    results = OptimizationResults(x_gt, x_0, f_0)

    while True:
        iteration += 1
        if iteration >= max_iterations:
            print("Maximum number of iterations ({}) reached.".format(max_iterations))
            break
        print("\nIteration {}".format(iteration))

        f = func(x)
        g = func.gradient(x)
        H = func.hessian(x)
        # From the problem text: we can use the full Hessian. For more complex
        # problems, one could also consider using a BFGS approximation of H.
        B = H

        def model(model_p):
            return f + np.dot(g.T, model_p) + 0.5 * np.dot(model_p.T, np.dot(B, model_p))

        p_k = dogleg_search(f, g, B, D)
        print("\tp_k = {}".format(p_k.ravel()))

        f_prev = f
        f_new = func(x + p_k)

        model_prev = model(np.zeros_like(p_k))
        model_new = model(p_k)

        gain_ratio = ((f_prev - f_new) / (model_prev - model_new))[0, 0]
        print("\tGain ratio: {:.8f}".format(gain_ratio))

        if gain_ratio < poor_model_threshold:
            # Poor model performance: Reduce the size of the trust region
            print("\nPoor model, reducing TR size!")
            D *= poor_model_threshold
        else:
            # Good model performance
            if gain_ratio > great_model_threshold and abs(l2_norm(p_k) - D) < 1e-5:
                print("\nGood model, increasing TR size!")
                # Great update. Increase the size of the trust region!
                D = min(2 * D, D_max)

        if gain_ratio > update_threshold:
            print("\tGood update, adding p_k to x.")
            x = x + p_k
            results.record(x, f_new, alpha=1.0)
            print("\tx = {}".format(x.ravel()))

            if l2_norm(p_k) < convergence_epsilon:
                print("\nStep size small to have converged. Stopping.")
                break

    return x, results



def main():
    samples = 500
    contour_count = 25
    xlim = [-2.0, 2.0]
    ylim = [-0.5, 2.0]
    x_0_easy = np.array([1.2, 1.2]).reshape(2, 1)
    x_0_hard = np.array([-1.2, 1.0]).reshape(2, 1)
    plt.figure(figsize=(8, 8))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plot_rosenbrock_contours(contour_count, samples, xlim, ylim)
    func = Rosenbrock()
    x_star_gt = np.array([1.0, 1.0]).reshape((2, 1))

    iterate_stride = 1
    start_t = time.time()
    x_final_sd_e, results_sd_e = trust_region(func, x_0=x_0_easy, x_gt=x_star_gt)
    delta_s = time.time() - start_t
    # print("Table for dogleg easy")
    # output_table(results_sd_e)
    print(results_sd_e.iterates)
    plot_iterates(results_sd_e, delta_s, color='b', stride=iterate_stride, label='Dogleg, easy')

    start_t = time.time()
    x_final_sd_h, results_sd_h = trust_region(func, x_0=x_0_hard, x_gt=x_star_gt)
    # print("Table for dogleg hard")
    # output_table(results_sd_h)
    delta_s = time.time() - start_t
    plot_iterates(results_sd_h, delta_s, color='r', stride=iterate_stride, label='Dogleg, hard')

    plt.scatter(x_star_gt[0], x_star_gt[1], s=100, marker='*')
    plt.scatter(x_0_easy[0], x_0_easy[1], s=75, c='b', marker='^')
    plt.scatter(x_0_hard[0], x_0_hard[1], s=75, c='r', marker='^')

    plt.annotate("Easy start",
                 xy=(x_0_easy[0], x_0_easy[1]),
                 xytext=(x_0_easy[0] + 0.2, x_0_easy[1] - 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05,
                                 width=0.25, headwidth=0.25))#, length_includes_head=True))

    plt.legend()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/dogleg-iterates.eps')
    plt.show()


if __name__ == '__main__':
    main()
