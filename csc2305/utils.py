import numpy as np
from matplotlib import pyplot as plt

from csc2305.functions import rosenbrock_sep


def is_spd(A):
    """Checks whether the given matrix is symmetric positive-definite."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def l2_norm(x):
    return np.linalg.norm(x, ord=2)


def plot_rosenbrock_contours(contour_count, samples, xlim, ylim):
    x = np.linspace(xlim[0], xlim[1], samples)
    y = np.linspace(ylim[0], ylim[1], samples)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_sep(X, Y)
    x_0_easy = np.array([1.2, 1.2]).reshape(2, 1)
    x_0_hard = np.array([-1.2, 1.0]).reshape(2, 1)
    x_0 = x_0_hard
    # contour_vals_bkg = np.linspace(0.0, np.max(Z) * 0.75, contour_count * 10)
    # cont_background = plt.contour(X, Y, Z, contour_vals_bkg, alpha=0.1)
    contour_vals = np.linspace(0.0, np.max(Z) * 0.25, contour_count)
    cont = plt.contour(X, Y, Z, contour_vals, alpha=0.5)
    plt.colorbar(cont)
    return x_0_easy, x_0_hard


def plot_iterates(results, time_s, label, color, stride=1):
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
    time_ms = int(time_s * 1000)
    label_full = "{} ({} iterations, {}ms))".format(label, len(its_np), time_ms)
    plt.scatter([-1000], [-1000], color=color, label=label_full)


def output_table(results, first=30, last=20):
    def line(fval, i, x, alpha, r, rq):
        print("| {:03d} | {:.4f}, {:.4f} | {:.4f} | {:4f} | {:4f} | {:4f} |".format(i, x[0][0], x[1][0], alpha, fval, r, rq))

    print("| it  | x1, x2         | alpha   | fval     | ratio   | quad_ratio |")
    print("|-----+----------------+---------+----------+---------+------------+")
    sset = results
    for i, (x, fval, alpha, r, r_q) in enumerate(sset):
        if r is not None:
            if len(results) > first + last:
                if i <= first or i >= len(results) - last:
                    line(fval, i, x, alpha, r, r_q)
                elif i == first + 1:
                    print("\n...\n")
            else:
                line(fval, i, x, alpha, r, r_q)
