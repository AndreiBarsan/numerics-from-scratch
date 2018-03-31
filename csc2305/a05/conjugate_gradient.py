"""Implementation of the linear conjugate gradient method for solving systems of
equations.

References
[NW] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization: Springer
     Series in Operations Research and Financial Engineering. Springer.

Additional info in assignment:
    Also discuss whether your numerical results are consistent with the theory
    for the convergence of the conjugate gradient method developed on pages
    112–118 of your textbook.
"""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as sp_la

from csc2305.utils import l2_norm

CGResult = namedtuple('CGResult', ['iterates', 'final_x', 'final_r'])


def linear_conjugate_gradient(A, b, x_0, **kwargs):
    """Implements a simple linear conjugate gradient solver.

    Based on Algorithm 5.2 from [NW].
    """
    threshold = kwargs.get('threshold', 1e-6)
    max_iter = kwargs.get('max_iter', 500)
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    assert b.shape == (N, 1)
    assert x_0.shape == (N, 1)

    iterates = []
    r = r_0 = np.dot(A, x_0) - b
    p = p_0 = -r_0
    x = x_0

    # TODO recycle the norm square of r.
    # We typically need to perform more than N iterations since
    for i in range(max_iter):
        iterates.append(x)
        old_r_dot = np.dot(r.T, r)
        Apk = np.dot(A, p)
        alpha = old_r_dot / np.dot(p.T, Apk)
        x = x + alpha * p
        r = r + alpha * Apk
        beta = np.dot(r.T, r) / old_r_dot
        p = -r + beta * p

        if l2_norm(r) < threshold:
            break

    if l2_norm(r) > threshold:
        raise ValueError("Programming error: CG did not converge!")

    return CGResult(iterates, x, r)


def main():
    exp_range = [5, 8, 12, 20]
    # exp_range = np.arange(2, 25)
    results = []

    for n in exp_range:
        A = sp_la.hilbert(n)
        cond = np.linalg.cond(A)
        # print("Condition number: {}".format(cond))
        evals, evecs = np.linalg.eig(A)
        # TODO(andreib): Discuss if our results consistent with the theory from
        # pages 112--118.

        b = np.ones((n, 1))
        x_0 = np.zeros_like(b)

        # Ignore warnings about ill-conditioning.
        # warnings.filterwarnings("ignore", r".*scipy.linalg.solve.*")
        # x_builtin = sp_la.solve(A, b)
        # print("Result of built-in solve:", x_builtin)

        result = linear_conjugate_gradient(A, b, x_0)
        results.append((n, len(result.iterates), cond))

    results_df = pd.DataFrame(data=results, columns=['N', 'Iterations', 'cond(A)'])
    print(results_df)
    print(results_df.to_latex(index=False))

    results_df.plot('N', 'Iterations')
    # plt.plot(exp_range, results_df['Iterations'], 'b-*', label="Actual CG Iterations")
    plt.plot(exp_range, exp_range, 'r--', label="Max Theoretical CG Iterations")
    plt.xlabel("Linear system size (n)")
    plt.ylabel("Iterations to convergence")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
