"""Data fitting with Levenberg-Marquardt.

The implementation is not optimized, and does not employ many of the tricks
employed by industry-grade libraries like MINPACK, but works well on simple
examples (see 'main' for a couple).
"""

import numpy as np
import matplotlib.pyplot as plt


from scipy import optimize
from sklearn.utils import shuffle as sk_shuffle


def model_fitting_M(x, t):
    E1 = np.exp(x[0] * t)
    E2 = np.exp(x[1] * t)

    return x[2] * E1 + x[3] * E2


def model_fitting(t, y):
    """Fits an exponential model (four parameters) to the given data.

    Example 1.1 from "Methods for Non-Linear Least Squares Problems"

    TODO-LOW(andrei): Support arbitrary models. The Jacobian should just have a
    flipped sign.

    Data fitting for the model:
      $$ M(x, t_i) = x_3 e^{x_1 t_i} + x_4 e^{x_2 t_i} $$
    """

    def f(x):
        return model_fitting_M(x, t) - y

    def f_jacobian(x):
        # wrt x, so y does not appear
        E1 = np.exp(x[0] * t)
        E2 = np.exp(x[1] * t)

        col_1 = t * x[2] * E1
        col_2 = t * x[3] * E2
        col_3 = E1
        col_4 = E2

        return np.array([
            col_1, col_2, col_3, col_4
        ]).T

    return f, f_jacobian, "Exponential model"


def simple_model_fitting_M(x, t):
    return x[0] * x[1] * t + x[2] * x[2] * (t ** 2) + x[3] * (t ** 3)


def simple_model_fitting(t, y):
    def f(x):
        return simple_model_fitting_M(x, t) - y

    def f_jacobian(x):
        return np.array([
            x[1] * t,
            x[0] * t,
            2 * x[2] * (t ** 2),
            (t ** 3)
        ]).T

    return f, f_jacobian, "Simple model"


def model_fitting_wrapper():
    """Generates dummy data for the model fitting function."""
    x_ground_truth = [-4.0, -5.0, 4.0, -4.0]
    # WARNING: this function "blows up" for negative inputs; this does not
    # affect fitting usually, but the very large discrepancy between the
    # labels' magnitudes may induce numerical instability.
    SAMPLE_COUNT = 200
    NOISE_STD = 0.005
    ts = np.linspace(-0.1, 2.0, SAMPLE_COUNT)
    ys = model_fitting_M(x_ground_truth, ts)
    epsilon = np.random.randn(len(ys)) * NOISE_STD
    ys += epsilon

    model_domain = np.linspace(-0.1, 2.0, 1000)
    model_plot = model_fitting_M(x_ground_truth, model_domain)
    plt.figure()
    plt.plot(model_domain, model_plot)
    plt.scatter(ts, ys, marker='x')

    ts, ys = sk_shuffle(ts, ys)
    return model_fitting(ts, ys), x_ground_truth


def powell():
    """A simple function which is challenging to optimize.

    Source: M. J. D. Powell (1970): A Hybrid Method for Non-Linear Equations
    """
    def f(x):
        return np.array([
            x[0],
            (10 * x[0]) / (x[0] + 0.1) + 2 * x[1] * x[1]
        ]).T

    def f_jacobian(x):
        # Note that this becomes singular at (0, 0), the function's global
        # optimum, making it tricky to optimize.
        return np.array([
            [1.0,                          0.0],
            [(x[0] + 0.1) ** -2.0,  4.0 * x[1]]
        ])

    return f, f_jacobian, "Powell Function"


def modified_powell():
    """Reparameterized to make the Jacobian non-singular."""
    def f(x):
        return np.array([
            x[0],
            (10 * x[0]) / (x[0] + 0.1) + 2 * x[1]
        ]).T

    def f_jacobian(x):
        return np.array([
            [1.0,                   0.0],
            [(x[0] + 0.1) ** -2.0,  2.0]
        ])

    return f, f_jacobian, "Modified Powell Function"


def modified_powell_harder():
    """A single extra quadratic term make the function much harder to optimize."""
    def f(x):
        return np.array([
            x[0] * x[0],
            (10 * x[0]) / (x[0] + 0.1) + 2 * x[1]
        ]).T

    def f_jacobian(x):
        return np.array([
            [2 * x[0],              0.0],
            [(x[0] + 0.1) ** -2.0,  2.0]
        ])

    return f, f_jacobian, "Modified Powell Function with Quadratic"


# TODO-LOW(andrei): This could be simplified where it's used.
def L(h, fx, Jx):
    return np.dot(fx.T, fx) + \
           2 * np.dot(h.T, np.dot(Jx.T, fx)) + \
           np.dot(h.T, np.dot(Jx.T, np.dot(Jx, h)))


def F(f, xx, N):
    return 1.0 / N * np.dot(f(xx).T, f(xx))


def levenberg_marquardt(f, J, x0, **kw):
    """Minimizes the function f using the Levenberg-Marquardt method.
    The implementation closely follows Algorithm 3.16. from [Madsen, 2004].

    Args:
        f: The function to minimize.
        J: The Jacobian of f.
        x0: The initial value of f's parameters, x.
        model: If fitting a residual function, the model part of it.
               Optional, and used only for visualization.

    Keyword Args:
        ground_truth: The ground truth value of x. Used for visualizing a
                      comparison between the true model and the estimated ones.
        plot: Whether convergence plots should be shown.

    Returns:
        x*: The value of the optimal parameters.
        steps: The number of steps taken to reach the optimum.
        opt_result: The OptimizationResult computed by scipy on the same
                    problem. Useful for comparing this implementation to a
                    strong baseline.
        trace: A dictionary containing the values of F, g's l-infinity norm,
               and mu over time.

    References:
        [Madsen, 2004]: Madsen, Kaj, Hans Bruun Nielsen, and Ole Tingleff.
        "Methods for non-linear least squares problems." (2004).

    """
    convergence_epsilon_g = kw.get('convergence_epsilon_g', 1e-8)
    convergence_epsilon_h = kw.get('convergence_epsilon_h', 1e-8)
    tau = kw.get('tau', 1e-3)
    max_steps = 100
    verbose = kw.get('verbose', True)

    # Rows in J, i.e., number of data points
    N = J(x0).shape[0]
    # Columns in J, i.e., number of model parameters
    M = J(x0).shape[1]

    print("The Jacobian is {}x{}.".format(N, M))
    step = 0
    # A factor used when scaling mu after poor updates
    nu = 2
    # The parameter(s) which get optimized
    x = x0

    # Keep track of things for e.g., plotting
    trace = {
        'F_vals': [],
        'g_infty_norms': [],
        'mus': []
    }

    # Initialize the variables used in the iteration
    A = np.dot(J(x).T, J(x))
    g = np.dot(J(x).T, f(x))
    found = (np.linalg.norm(g, np.inf) <= convergence_epsilon_g)
    mu = tau * np.max(A[np.diag_indices(A.shape[0])])

    def log(msg, *args, **kw):
        if verbose: print(msg, *args, **kw)

    # The main optimization loop
    while not found and step < max_steps:
        step += 1

        # Solve for the optimal increment h
        I = np.eye(M)
        h_lm = np.linalg.solve(A + mu * I, -g)

        g_infty_norm = np.linalg.norm(g, np.inf)
        F_current = F(f, x, N)
        trace['F_vals'].append(F_current)
        trace['g_infty_norms'].append(g_infty_norm)
        trace['mus'].append(mu)
        threshold = (convergence_epsilon_h * (np.linalg.norm(x) + convergence_epsilon_h))
        log("Current value of objective:", F_current)

        if g_infty_norm <= threshold:
            # Convergence test passed. We're done!
            log("Converged.")
            found = True
        else:
            # Adjust the (Gradient Descent) <-> (Gauss-Newton) balance
            gr_num = np.dot(f(x), f(x)) - \
                     np.dot(f(x + h_lm), f(x + h_lm))
            fx = f(x)
            Jx = J(x)
            gr_denom = L(np.zeros(h_lm.shape), fx, Jx) - L(h_lm, fx, Jx)
            gain_ratio = (gr_num) / (gr_denom)

            if gr_denom < 0:
                raise ValueError(
                    "Error: Solving for h_lm yielded a WORSE result. "
                    "This is Likely an implementation bug.")

            log("New value of objective:", F(f, x + h_lm, N))
            log("Gain ratio:", gain_ratio)

            if gain_ratio > 0:
                log("Good update: should decrease mu")
                # The update was good! Make a step in the determined direction.
                x = x + h_lm

                A = np.dot(J(x).T, J(x))
                g = np.dot(J(x).T, f(x))
                found = (np.max(np.abs(g)) <= convergence_epsilon_g)
                mu = mu * max(1.0 / 3.0, 1 - (2 * gain_ratio - 1) ** 3)
                nu = 2

                if found:
                    log("Converged!")
            else:
                log("Poor update. Adjusting  mu and nu.")
                # The update was too small or even increased f!
                # In this case, we want to "shift" LM more towards the "coarser"
                # gradient descent mode, which is done by increasing mu.
                mu = mu * nu
                nu = nu * 2

    log("Levenberg-Marquardt optimization complete.")
    if not found:
        log("Warning: terminated without finding optimum ({} steps)", end='')
    else:
        log("Found an optimum ", end='')

    log("in {} steps.".format(step))
    log("Final x:", x)
    log("Final optimal value: F = ", F(f, x, N))

    log("Now doing the same thing with the built-in scipy LM implementation.")
    opt_result = optimize.least_squares(f, x0, J, method='lm', verbose=1,
                                        max_nfev=max_steps)
    log("Scipy LM result:")
    log("\t[Scipy] Solution found:", opt_result.x)
    log("\t[Scipy] Final optimal value:", opt_result.cost)
    # log(opt_result)

    return x, step, opt_result, trace


def plot_trace(f, J, x, model, scipy_opt_result, steps_taken, trace,
               **kw):
    # First, plot the values of F, ||g||, and mu over time.
    plt.figure()
    plt.plot(np.arange(0, steps_taken), trace['F_vals'], label='F')
    plt.plot(np.arange(0, steps_taken), trace['g_infty_norms'],
             label=r"$\left\|| g \right\||$")
    plt.plot(np.arange(0, steps_taken), trace['mus'], label='$\mu$')
    plt.xlabel("Iteration")
    plt.semilogy()
    plt.legend()
    plt.title("Final optimal value: us = {:.8f}, scipy = {:.8f}".format(
        F(f, x, J(x).shape[0]), scipy_opt_result.cost
    ))

    # Second, if applicable, visually compare our model's fit to scipy's
    # (and, optionally, the ground truth).
    if model is not None:
        # If we are fitting an actual model, plot the result.
        # TODO-LOW(andrei): Handle the domain properly.
        est_model_dom = np.linspace(-0.1, 2.0, 1000)
        est_model_vals_ours = model(x, est_model_dom)
        est_model_vals_scipy = model(scipy_opt_result.x, est_model_dom)

        plt.figure()
        if kw['ground_truth'] is not None:
            est_model_vals_gt = model(kw['ground_truth'], est_model_dom)
            plt.plot(est_model_dom, est_model_vals_gt, label="Ground Truth")
        plt.plot(est_model_dom, est_model_vals_ours, label="Our estimate")
        plt.plot(est_model_dom, est_model_vals_scipy, '--', label="SciPy estimate")
        plt.legend()

    plt.show()


def main():
    # A very straightforward function to minimize
    # f, J, name = modified_powell()
    # A much trickier function to minimize
    f, J, name = powell()
    x0 = np.array([3, 1])
    # Alternative values which also work well:
    # convergence_epsilon_g = 1e-8
    # convergence_epsilon_h = 1e-8
    # tau = 1e-3
    x_star, steps, opt_result, trace = levenberg_marquardt(
        f, J, x0,
        convergence_epsilon_g=1e-15, convergence_epsilon_h=1e-15, tau=1.0)
    plot_trace(f, J, x_star, None, opt_result, steps, trace)

    # A four-parameter residual function
    (f_fit, J_fit, name), x_gt = model_fitting_wrapper()
    # The book states that having a good guess of the exponents' signs is
    # important. This claim can be verified by using the following
    # initialization, which will likely converge to a local optimum.
    # x0 = np.random.randn(4)
    # This initialization does not work with our implementation, but does
    # with scipy's.
    # x0 = np.array([-1.0, -1.0, 1.0, -1.0])
    # This initialization is recommended in the book and works well.
    x0 = np.array([-1.0, -2.0, 1.0, -1.0])
    x_star, steps, opt_result, trace = levenberg_marquardt(
        f_fit, J_fit, x0,
        convergence_epsilon_g=1e-8, convergence_epsilon_h=1e-8, tau=1e-3)
    plot_trace(f_fit, J_fit, x_star, model_fitting_M, opt_result, steps, trace,
               ground_truth=x_gt)


if __name__ == '__main__':
    main()
