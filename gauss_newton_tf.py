"""An implementation of the Gauss-Newton algorithm as a TensorFlow graph."""

import tensorflow as tf


def gn_iteration(x, J, f, alpha, i):
    """Refines the estimate x with a GN iteration."""

    with tf.variable_scope("gn-block-{:02d}".format(i)):
        # (J'J) h = J'f | A h = b
        # Where A = J'J and b = J'f

        # TODO use transpose kwarg
        # This is what we want to factorize
        A = tf.matmul(tf.transpose(J), J)
        b = tf.matmul(tf.transpose(J), f)

        chol = tf.cholesky(A, "chol-fact")
        h = tf.cholesky_solve(chol, b, "substitution-for-chol-solve")

        result = x - h * alpha

        return result


def dummy_f(x, lbd):
    return tf.reshape(tf.stack([
        x + 1.0,
        lbd * x * x + x - 1.0
    ]), [2, 1])


def dummy_f_jacobian(x, lbd):
    return tf.reshape(tf.stack([
        tf.constant([[1.0]]),
        2 * lbd * x + 1
    ]), [2, 1])


def gauss_newton(f, J, x0):
    sess = tf.InteractiveSession()
    # Some parameter we don't really care about
    lbd = tf.convert_to_tensor(2.0, tf.float32)
    # cx = tf.convert_to_tensor(x0, tf.float32)
    cx = x0
    N_ITERATIONS = 50

    print(x0)
    print(f(x0, lbd))
    print(J(x0, lbd))

    for i in range(N_ITERATIONS):
        next_x = gn_iteration(cx, J(cx, lbd), f(cx, lbd), 1.0, i)
        cx = next_x

    final_x = cx
    x_val = sess.run(final_x)
    print(x0)
    print("Final solution value:")
    print(x_val)

    print("Final value of function:")
    print(sess.run(dummy_f(final_x, lbd)))

    print("Final (scalar) value of function:")
    print(sess.run(tf.reduce_mean(tf.square(dummy_f(final_x, lbd)))))

    print("Final value of Jacobian:")
    print(sess.run(dummy_f_jacobian(final_x, lbd)))



def main():
    x0 = tf.constant([5], dtype=tf.float32)
    x0 = tf.reshape(x0, [1, 1])
    gauss_newton(dummy_f, dummy_f_jacobian, x0)


if __name__ == '__main__':
    main()
