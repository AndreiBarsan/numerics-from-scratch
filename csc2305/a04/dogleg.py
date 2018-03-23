"""Implementation of the dogleg trust region optimization method.

References
    [NW] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization: Springer
    Series in Operations Research and Financial Engineering. Springer.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

# This allows us to run some little experiments in parallel!
from sklearn.externals import joblib

from csc2305.functions import Rosenbrock
from csc2305.optimization_results import OptimizationResults
from csc2305.utils import plot_rosenbrock_contours, plot_iterates, \
    l2_norm


class OptimizationError(Exception):
    pass


def naive_search(tau_start, tau_end, p_fun, trust_radius, step_size):
    """Searches the dogleg path for an optimal tau in [0, 2].

    Uses Lemma 4.2 to do a simple search for the optimal tau in (4.16), such
    that the step is still within the trust region of radius 'trust_radius'.
    """
    tau_cur = tau_end
    if l2_norm(p_fun(tau_start)) >= trust_radius:
        raise ValueError("Invalid dog leg start point.")

    while True:
        p_cur = p_fun(tau_cur)
        if l2_norm(p_cur) <= trust_radius:
            return p_cur

        tau_cur -= step_size

        if tau_cur < tau_start:
            tau_cur += step_size
            step_size /= 2.0
            print("\t\tStep adjustment inside the naive search [new step size = {}].".format(step_size))


def dogleg_search(g, B, trust_radius, **kwargs):
    """Computes an update direction using the dogleg method.

    Solves Equation (4.3) using the dogleg method
    """
    search_method = naive_search

    # p_U is the point corresponding to the standard steepest-descent step.
    alpha = np.dot(g.T, g) / np.dot(g.T, np.dot(B, g))
    assert alpha.shape == (1, 1)
    p_U = -alpha * g

    # Important: using 'solve' is much more efficient than naively inverting B.
    # While it is not a huge issue in our small 2D problem, for larger-scale
    # problems, e.g., least-squares problems with thousands or hundreds of
    # thousands of variables, it can make a huge diference.
    #
    # Technical note: solves with gesv, not potrs, so that we don't have to
    # assume B is PD. If we were to know for sure that B is PD, like in
    # Gauss-Newton, for instance, then we can use potrf+potrs (Cholesky
    # factorization, followed by substitution) for increased efficiency.
    p_B = -np.linalg.solve(B, g)

    if l2_norm(p_B) <= trust_radius:
        # The tip of the dogleg is inside the trust region, so return it
        # directly.
        return p_B

    search_step_size = kwargs.get('step_size', 0.05)
    if l2_norm(p_U) > trust_radius:
        # First half of the dogleg
        #
        # p_U already outside TR. Should search for tau along the [0, 1] range.
        def p_fun(tau):
            return tau * p_U
        return search_method(
            tau_start=0.0,
            tau_end=1.0,
            p_fun=p_fun,
            trust_radius=trust_radius,
            step_size=search_step_size)
    else:
        # Second half of the dogleg
        #
        # p_U inside TR. Should compute p_B and search for tau along the [1, 2]
        # range.
        def p_fun(tau):
            return p_U + (tau - 1) * (p_B - p_U)
        return search_method(
            tau_start=1.0,
            tau_end=2.0,
            p_fun=p_fun,
            trust_radius=trust_radius,
            step_size=search_step_size)


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
    max_iterations = kwargs.get('max_iterations', 5000)
    poor_model_threshold = 0.25
    great_model_threshold = 0.75
    assert poor_model_threshold < 1
    f_0 = func(x_0)

    max_trust_radius = kwargs.get('max_trust_radius', 0.75)
    initial_trust_radius = kwargs.get('initial_trust_radius', 0.25)
    # eta in the textbook
    update_threshold = kwargs.get('update_threshold', 0.20)

    x = x_0
    trust_radius = initial_trust_radius
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

        p_k = dogleg_search(g, B, trust_radius, **kwargs)
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
            trust_radius *= poor_model_threshold
        else:
            # Good model performance
            if gain_ratio > great_model_threshold and abs(l2_norm(p_k) - trust_radius) < 1e-5:
                print("\nGood model, increasing TR size!")
                # Great update. Increase the size of the trust region!
                trust_radius = min(2 * trust_radius, max_trust_radius)

        if gain_ratio > update_threshold:
            print("\tGood update, adding p_k to x.")
            x = x + p_k
            results.record(x, f_new, alpha=1.0)
            print("\tx = {}".format(x.ravel()))

            if l2_norm(p_k) < convergence_epsilon:
                print("\nStep size small => converged. Stopping.")
                break

    return x, results


def experiment(row, col, func, x_0, x_gt, max_radius, initial_radius):
    """Wrapper around 'trust_region' used by 'heatmap_experiment'."""
    max_iterations = 500
    x_tr, res = trust_region(func, x_0, x_gt,
                             max_trust_radius=max_radius,
                             initial_trust_radius=initial_radius,
                             max_iterations=max_iterations
                             )
    return row, col, len(res.iterates) if np.allclose(x_tr, x_gt) else max_iterations


def heatmap_experiment(func, x_0, x_gt):
    max_trust_radii = np.arange(0.05, 3.05, 0.05)
    initial_trust_radius_factors = np.arange(0.05, 1.05, 0.05)[::-1]

    heatmap = np.zeros((len(initial_trust_radius_factors),
                        len(max_trust_radii)))
    total_runs = len(max_trust_radii) * len(initial_trust_radius_factors)
    print("Will run the optimization procedure {} times.".format(total_runs))

    results = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(experiment)(row, col, func,
                                   x_0, x_gt,
                                   max_trust_radius,
                                   max_trust_radius * factor)
            for col, max_trust_radius in enumerate(max_trust_radii)
            for row, factor in enumerate(initial_trust_radius_factors)
    )
    print("Results processed. Aggregating...")
    for row, col, its in results:
        heatmap[row, col] = its

    plt.figure(figsize=(12, 4))
    im = plt.imshow(heatmap, cmap='viridis')
    plt.colorbar(im)
    plt.xlabel(r"Maximum Trust Region Radius ($\hat{\Delta}$)")
    plt.ylabel(r"$\Delta_0$ fraction of $\hat{\Delta}$")

    X_TICK_RATIO = 4
    def ff(_, tick_number):
        if tick_number >= 0 and tick_number * X_TICK_RATIO < len(max_trust_radii):
            return max_trust_radii[tick_number * X_TICK_RATIO]
        else:
            return None

    plt.xticks(np.arange(0, len(max_trust_radii), X_TICK_RATIO))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(ff))
    plt.yticks(range(len(initial_trust_radius_factors)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(
        lambda _, i: initial_trust_radius_factors[i]
            if 0 <= i < len(initial_trust_radius_factors)
            else None
    ))
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'heatmap.eps'))
    plt.savefig(os.path.join('figures', 'heatmap.png'))
    plt.show()


def main():
    samples = 500
    contour_count = 25
    xlim = [-2.0, 2.0]
    ylim = [-0.5, 2.0]
    x_0_easy = np.array([1.2, 1.2]).reshape(2, 1)
    x_0_hard = np.array([-1.2, 1.0]).reshape(2, 1)
    plot_dogleg = False
    if plot_dogleg:
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
    if plot_dogleg:
        plot_iterates(results_sd_e, delta_s, color='b', stride=iterate_stride, label='Dogleg, easy')

    start_t = time.time()
    x_final_sd_h, results_sd_h = trust_region(func, x_0=x_0_hard, x_gt=x_star_gt)
    # print("Table for dogleg hard")
    # output_table(results_sd_h)
    delta_s = time.time() - start_t

    if plot_dogleg:
        plot_iterates(results_sd_h, delta_s, color='r', stride=iterate_stride, label='Dogleg, hard')

        plt.scatter(x_star_gt[0], x_star_gt[1], s=100, marker='*')
        plt.scatter(x_0_easy[0], x_0_easy[1], s=75, c='b', marker='^')
        plt.scatter(x_0_hard[0], x_0_hard[1], s=75, c='r', marker='^')

        arrowprops = dict(facecolor='black', shrink=0.05, width=0.25, headwidth=4.0)
        plt.annotate("Easy start",
                     xy=(x_0_easy[0], x_0_easy[1]),
                     xytext=(x_0_easy[0] + 0.2, x_0_easy[1] - 0.2),
                     arrowprops=arrowprops)
        plt.annotate("Hard start",
                     xy=(x_0_hard[0], x_0_hard[1]),
                     xytext=(x_0_hard[0] - 0.5, x_0_hard[1] - 0.2),
                     arrowprops=arrowprops)
        plt.legend()
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/dogleg-iterates.eps')
        plt.savefig('figures/dogleg-iterates.png')
        plt.show()

    heatmap_experiment(func, x_0_hard, x_star_gt)


if __name__ == '__main__':
    main()


# def continuous_binary_search(tau_start, tau_end, p_fun, delta, max_steps):
#     # Uses Lemma 4.2 to do a simple search for the optimal tau in (4.16), such
#     # that the step is still within the trust region of radius delta.
#     # func decreases as tau increases
#     # Want smallest value of func with argument under delta.
#     # I.e., since m(p_tilde(tau)) function decreases with tau, want the biggest
#     # tau s.t., p(tau) < delta.
#
#     tau_mid = (tau_end - tau_start) / 2.0
#     if max_steps == 0:
#         # TODO(andreib): Maybe return the min, dunno.
#         return p_fun(tau_mid)
#
#     p_mid = p_fun(tau_mid)
#     if l2_norm(p_mid) < delta:
#         return continuous_binary_search(tau_mid, tau_end, p_fun, delta, max_steps-1)
#     else:
#         return continuous_binary_search(tau_start, tau_mid, p_fun, delta, max_steps-1)
