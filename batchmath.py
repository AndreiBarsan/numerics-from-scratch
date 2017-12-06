import numpy as np


def main():
    np.random.seed(1234)
    # Say these are some dummy camera matrices.
    p1 = np.random.randint(0, 25, [3, 4])
    p2 = np.random.randint(0, 25, [3, 4])
    p3 = np.random.randint(0, 25, [3, 4])
    p4 = np.random.randint(0, 25, [3, 4])
    p5 = np.random.randint(0, 25, [3, 4])
    print(p1), print(p2), print(p3)

    ps = np.array([p1, p2, p3, p4, p5])
    print(ps.shape)

    # Let's now say we have some vectors, and we want to batch transform ALL
    # those vectors with their corresponding matrices.
    xs = np.random.randint(-3, 3, [5, 4, 1])
    print(xs)

    # Want: a 5 x 3 (x 1) array where each row k is the result of
    # transforming the kth vector with the kth matrix.
    print("np.dot between {} and {}".format(ps.shape, xs.shape))
    transformed = np.dot(ps, xs)
    # transformed = np.tensordot(ps, xs, axes=[[1, 2], [1]])
    # ps * xs
    print("Result:")
    print(transformed)
    print(transformed.shape)

    transformed = transformed.sum(axis=2).squeeze()
    print("Clean?")
    print(transformed)
    print(transformed.shape)

    print("Manual result")
    res = np.zeros(shape=(ps.shape[0], ps.shape[1]))
    print(res.shape)
    for i in range(ps.shape[0]):
        x = np.dot(ps[i, :, :], xs[i, :, 0])
        res[i, :] = x
    print(res)


if __name__ == '__main__':
    main()
