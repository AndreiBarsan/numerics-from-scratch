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
from scipy.sparse import linalg as ss_la

from csc2305.utils import l2_norm

CGResult = namedtuple('CGResult', ['iterates', 'final_x', 'final_r'])


def linear_conjugate_gradient(A, b, x_0, **kwargs):
    """Implements a simple linear conjugate gradient solver.

    Based on Algorithm 5.2 from [NW].
    """
    tolerance = kwargs.get('tolerance', 1e-6)
    max_iter = kwargs.get('max_iter', 5000)
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    assert b.shape == (N, 1)
    assert x_0.shape == (N, 1)

    iterates = []
    r = r_0 = np.dot(A, x_0) - b
    p = p_0 = -r_0
    x = x_0

    cur_r_sq_norm = np.dot(r_0.T, r_0)
    # We typically need to perform more than N iterations since
    for i in range(max_iter):
        iterates.append(x)
        Apk = np.dot(A, p)
        alpha = cur_r_sq_norm / np.dot(p.T, Apk)
        x = x + alpha * p
        r = r + alpha * Apk
        old_r_sq_norm = cur_r_sq_norm
        cur_r_sq_norm = np.dot(r.T, r)
        beta = cur_r_sq_norm / old_r_sq_norm
        p = -r + beta * p

        if l2_norm(r) <= tolerance:
            break

    if l2_norm(r) > tolerance:
        raise ValueError("Likely programming error: CG did not converge after "
                         "the maximum number of iterations, {}!".format(max_iter))

    return CGResult(iterates, x, r)


def main():
    exp_range = [5, 8, 12, 20]
    # Warning: float128 is not supported by most internal subroutines called by
    # numpy. 64-bit floats is the best we can do...

    numeric_type = np.float64
    # exp_range = np.arange(2, 25)
    results = []
    tolerance = 1e-6

    for n in exp_range:
        A = sp_la.hilbert(n).astype(numeric_type)
        cond = np.linalg.cond(A)
        evals, _ = np.linalg.eig(A)
        e_min = float(np.real(evals[-1]))
        e_max = float(np.real(evals[0]))

        b = np.ones((n, 1)).astype(numeric_type)
        x_0 = np.zeros_like(b).astype(numeric_type)

        res, info, it_builtin = ss_la.cg(A, b, tol=tolerance)

        result = linear_conjugate_gradient(A, b, x_0, tolerance=tolerance)
        results.append((n, len(result.iterates), it_builtin, e_min, e_max, cond))

    results_df = pd.DataFrame(data=results,
                              columns=['N', 'Iterations', 'Iterations (SciPy)',
                                       'min eval', 'max eval', 'cond(A)'])
    print(results_df.to_latex(index=False, float_format=lambda f: '{:.4e}'.format(f)))

    results_df.plot('N', 'Iterations', marker='*')
    # plt.plot(exp_range, results_df['Iterations'], 'b-*', label="Actual CG Iterations")
    plt.plot(exp_range, exp_range, 'r--', label="Max Theoretical CG Iterations")
    plt.xlabel("Linear system size (n), tolerance = {}".format(tolerance))
    plt.ylabel("Iterations to convergence")
    # plt.ylim(0, 125)
    plt.legend()
    plt.savefig("cg-fig-{}-{}.eps".format(numeric_type, tolerance))
    plt.show()


if __name__ == '__main__':
    main()
