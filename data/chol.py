# Handcoded Cholesky factorization, for learning purposes

import math

import numpy as np

def pychol_wrong(A):
    """Performs a Cholesky decomposition of the PSD matrix A.

    Based on Appendix A from Nocedal & Wright. Runs in cubic time.
    """
    raise ValueError("Something is incorrect with this implementation.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        L[i, i] = np.sqrt(A[i, i])
        for j in range(i+1, n):
            L[j, i] = A[j, i] / L[i, i]
            for k in range(i+1, j):
                A[j, k] = A[j, k] - L[j, i] * L[k, i]

    return L


def pychol(A):
    L = np.zeros_like(A)
    for i in range(len(A)):
        for j in range(i + 1):
            s = 0
            for k in range(j):
                s += L[i, k] * L[j, k]

            if i == j:
                L[i, j] = math.sqrt(A[i, i] - s)
            else:
                L[i, j] = 1.0 / L[j, j] * (A[i, j] - s)

    return L


def main():
    A = np.array([
        [ 1, 2, 3 ],
        [ 3, 6, 8 ],
        [ 1, 2, -3 ],
    ], dtype=np.float32)

    M = np.dot(A.T, A)
    print(A)
    print(M)
    L_np = np.linalg.cholesky(M)
    print("Numpy lower:")
    print(L_np)
    print(np.dot(L_np, L_np.T))

    L_my = pychol(M)
    print("My lower:")
    print(L_my)
    print(np.dot(L_my, L_my.T))


if __name__ == '__main__':
    main()
