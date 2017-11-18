"""Playing around with solving triangular systems."""

import unittest

import numpy as np
import scipy.linalg as sp_la


FORWARD_SUBSTITUTION = 1
BACKWARD_SUBSTITUTION = 2
RANDOM_SEED = 1984

np.random.seed(RANDOM_SEED)


def solve_triang_subst(A: np.ndarray, b: np.ndarray,
                       substitution=FORWARD_SUBSTITUTION,
                       refine_result=True, **kw) -> np.ndarray:
    """For educational purposes only. Solves a triangular system via
    forward or backward substitution.

    Note that the precision of this method is not great, and floating point
    errors can easily accumulate, especially the closer A is to being singular.

    See Section 2.5 in Numerical Recipes in C for more information.

    A must be triangular. FORWARD_SUBSTITUTION means A should be
    lower-triangular, BACKWARD_SUBSTITUTION means A should be upper-triangular.

    Question: What does BLAS do?
    Answer: Based on ctrsm.f in BLAS, around line 280, it loops through the rows
            starting from the last one. If the row's b is nonzero, then it sets
            B(row, J) = B(row, J) / A(row, row)
            INTERESTING: It also has an outer loop over columns?
            This seems to account for the case where B is itself a matrix and
            we're essentially solving multiple systems of equations.
    """
    refinement_iterations = kw.get('refinement_iterations', 1)
    rows = len(A)
    x = np.zeros(rows, dtype=A.dtype)
    row_sequence = reversed(range(rows)) if substitution == BACKWARD_SUBSTITUTION else range(rows)
    for row in row_sequence:
        # This would loop over the columns of A.
        delta = b[row] - np.dot(A[row], x)
        cur_x = delta / A[row][row]
        x[row] = cur_x

    if refine_result:
        for _ in range(refinement_iterations):
            print("Performing solution refinement...")
            delta_b = np.dot(A, x) - b
            print("Delta b norm:")
            delta_b_norm = np.linalg.norm(delta_b, ord=2)
            print(delta_b_norm)
            if abs(delta_b_norm) > 1e-16:
                delta_x = solve_triang_subst(A, delta_b, substitution, False)
                x = x - delta_x

    return x


class TestSubstitutions(unittest.TestCase):

    def __init__(self):
        pass


def check(sol: np.ndarray, x_gt: np.ndarray, description: str) -> None:
    print("Solution found with {}:".format(description))
    print(sol)
    if not np.allclose(sol, x_gt):
        print("Ground truth (not achieved...)")
        print(x_gt)
        raise ValueError("{} did not work!".format(description))


def fuzz_test_solving():
    # TODO try proper 100x100, 1000x1000 matrices for testing!
    # TODO make into actual test case
    # TODO plot residual magnitudes with and without refinement

    N_ITERATIONS = 200
    refine_result = True
    for mode in [FORWARD_SUBSTITUTION, BACKWARD_SUBSTITUTION]:
        print("Starting mode {}".format(mode))
        for iteration in range(N_ITERATIONS):
            N = np.random.randint(3, 250)
            A = np.random.uniform(0.0, 1.0, [N, N]).astype(np.float64)

            if mode == BACKWARD_SUBSTITUTION:
                A = np.triu(A)
            elif mode == FORWARD_SUBSTITUTION:
                A = np.tril(A)
            else:
                raise ValueError()

            det = np.linalg.det(A)
            if det < 1e-8:
                # Ensure the coefficient matrix is reasonably conditioned,
                # since otherwise we would get numerical instabilities.
                # Otherwise, due to the very small numbers on the diagonal,
                # the matrix very quickly becomes almost singular.
                A += np.eye(N)

            x_gt = np.random.uniform(0.0, 1.0, N).astype(np.float64)
            b = np.dot(A, x_gt)

            x_est = solve_triang_subst(A, b, substitution=mode,
                                       refine_result=refine_result)
            # TODO report error and count, don't throw!
            # Keep track of error norm!!
            check(x_est, x_gt,
                  "Mode {} custom triang iteration {}".format(mode, iteration))


if __name__ == '__main__':
    # Some random upper-triangular matrix.
    A = np.array([
        [1.0, 1.0, 1.0, 1.5, 2.4],
        [0.0, -1.3, 1.0, 1.2, -5.1],
        [0.0, 0.0, 1.5, 2.7, 5.22],
        [0.0, 0.0, 0.0, 2.0, 3.16],
        [0.0, 0.0, 0.0, 0.0, 0.16]
    ])
    x_gt = np.array([24, 11, 5, 4, -3], dtype=np.float32)
    b = np.dot(A, x_gt)

    print("Coefficient matrix A:")
    print(A)
    print("b:")
    print(b)

    print("Ground truth x:")
    print(x_gt)

    x_inversion = np.dot(np.linalg.inv(A), b)
    check(x_inversion, x_gt, "Solution found with simple inversion:")

    x_solve = sp_la.solve(A, b)
    check(x_solve, x_gt, "Solution found with 'solve'.")

    # This uses the LAPACK routine "trtrs" for solving a system with a
    # triangular coefficient matrix.
    x_solve_triang = sp_la.solve_triangular(A, b)
    check(x_solve_triang, x_gt, "Solution found with 'solve_triangular'.")

    x_solve_triang_mine = solve_triang_subst(A, b, substitution=BACKWARD_SUBSTITUTION)
    check(x_solve_triang_mine, x_gt, "Solution found with custom 'solve_triangular'.")

    print("Starting to fuzz test the custom substitution...")
    fuzz_test_solving()