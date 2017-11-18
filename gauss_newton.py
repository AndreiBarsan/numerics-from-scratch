"""Very simple experimenting with GN and similar optimization algorithms."""

import numpy as np


def f(x, l):
    """The function to optimize.

    Has a global optimum at x = 0 for l < 1.
    """
    return np.array([
        x + 1.0, l * x * x + x - 1.0
    ]).reshape((2, 1))


def F(x, l):
    f_val = f(x, l)
    return 0.5 * np.dot(f_val.T, f_val)


def f_jacobian(x, l):
    return np.array([
        1, 2 * l * x + 1
    ]).reshape((2, 1))


def inv(arr: np.ndarray) -> np.ndarray:
    if len(arr.shape) == 1:
        return 1.0 / arr
    else:
        return np.linalg.inv(arr)


def update_step_old(x, l):
    num = 2.0 * l * l * x * x * x + 3.0 * l * x * x + 2.0 * (1.0 - l) * x
    denom = 4.0 * l * l * x * x + 4.0 * l * x + 2.0
    return -(num / denom)


def solve_normal_equations(x, l):
    # Note: in practice you want to avoid the expensive inversion operation, and
    # replace it with, e.g., a forward substitution approach, since the invertee
    # is positive semidefinite (as long as the Jacobian is full rank).
    num = np.dot(f_jacobian(x, l).T, f(x, l))
    denom = np.dot(f_jacobian(x, l).T, f_jacobian(x, l))
    return -(np.dot(num, inv(denom)))


def solve_normal_equations_generic(x, f, J):
    # Still a naive implementation.
    Jx = J(x)
    fx = f(x)
    num = np.dot(Jx.T, fx)
    denom = np.dot(Jx.T, Jx)

    return -(np.dot(num, inv(denom)))


def gauss_newton():
    MAX_STEPS = 100
    # Just a fixed parameter of the function we are optimizing.
    LAMBDA = 1.5
    step = 0

    # TODO(andrei): What is a smart way to initialize this?
    x = 5.0

    while step < MAX_STEPS:
        step += 1
        print("Step {}: ".format(step), end='')

        # h_opt = solve_normal_equations(x, LAMBDA)
        h_opt = solve_normal_equations_generic(x,
                                               lambda x: f(x, LAMBDA),
                                               lambda x: f_jacobian(x, LAMBDA))
        h_opt_old = update_step_old(x, LAMBDA)

        print("h_opt = {:} \t| ".format(h_opt), end='')
        print("h_opt_old = {:} \t| ".format(h_opt_old), end='')

        alpha = 1.0
        x = (x + alpha * h_opt)[0, 0]

        print("x    = {:} \t| ".format(x), end='')
        print("F(x) = {}".format(F(x, LAMBDA)))

        print()


def main():
    gauss_newton()


if __name__ == "__main__":
    main()
