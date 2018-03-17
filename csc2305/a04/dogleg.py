"""Implementation of the dogleg trust region optimization method.

References
    [NW] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization: Springer
    Series in Operations Research and Financial Engineering. Springer.
"""


def dogleg(func, x_0, alpha_0, f_star_gt):
    """

    Problem statement (4.2 from [NW]):
        Write a program that implements the dogleg method. Choose Bk to be the
        exact Hessian. Apply it to solve Rosenbrock’s function (2.22).

        Experiment with the update rule for the trust region by changing the
        constants in Algorithm 4.1, or by designing your own rules.
    """
    iteration = 0
    x = x_0
    convergence_epsilon = 1e-8

    pass


def main():
    pass


if __name__ == '__main__':
    main()
