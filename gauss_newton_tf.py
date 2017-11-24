"""An implementation of the Gauss-Newton algorithm as a TensorFlow graph."""

import os

import numpy as np
import tensorflow as tf

from scipy import linalg as sp_la
from scipy import optimize as sp_optimize

tf.flags.DEFINE_string("label", "", "A custom label for the experiment.")
tf.flags.DEFINE_string("output_dir", "out", "The parent directory in which "
                                            "to place this project's "
                                            "experiment folder.")
tf.flags.DEFINE_string("project_name", "dsfm", "The name of this project.")

FLAGS = tf.flags.FLAGS

LOGDIR_BASE = os.path.join(FLAGS.output_dir, FLAGS.project_name)


def gn_iteration(x, J, f, alpha, block_idx, loss_hist):
    """Refines the estimate x with a GN iteration."""
    N = tf.shape(f)[0]
    print("f's dimension tensor:")
    print(N)

    with tf.variable_scope("gn-block-{:02d}".format(block_idx)):
        # (J'J) h = J'f | A h = b
        # Where A = J'J and b = J'f

        # This is what we want to factorize
        Jt = tf.transpose(J)
        A = tf.matmul(Jt, J)
        b = tf.matmul(Jt, f)

        chol = tf.cholesky(A, "chol-fact")
        h = tf.cholesky_solve(chol, b, "substitution-for-chol-solve")

        # TODO(andrei): Do we need this 'tf.constant'?
        current_loss = (tf.constant(1.0, dtype=tf.float32) / tf.cast(N, tf.float32)) *\
                       tf.matmul(tf.transpose(f), f)

        print("Current loss tensor:")
        print(current_loss)

        # TODO(andrei): Wire each loss from each step as one histogram event,
        # so we should be able to visualize the loss curve as a distribution
        # over time (and verify it keeps having a negative slope).
        tf.summary.scalar("gn-loss-{:02d}".format(block_idx), current_loss[0][0])

        # TODO(andrei): Write this, then dump as histogram for TensorBoard.
        # loss_hist[block_idx] = current_loss

        result = x - h * alpha
        return result


# TODO(andrei): Unify the NP and TF versions.
def dummy_f_np(x, lbd):
    return np.array([
        x[0] + 1.0,
        lbd * x[0] * x[0] + x[0] - 1.0
    ]).T
# TODO do we need the transpose?


def dummy_f_jacobian_np(x, lbd):
    return np.array([
        1.0,
        2 * lbd * x[0] + 1
    ]).reshape([2, 1])


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


class OptimizationResult(object):
    def __init__(self, steps, name, f_value, F_value, x_result, final_jacobian):
        self.name = name
        self.steps = steps
        self.f_value = f_value
        self.F_value = F_value
        self.x_result = x_result
        self.final_jacobian = final_jacobian


def get_logdir(logdir_base):
    if len(FLAGS.label) == 0:
        return logdir_base
    else:
        return logdir_base + "-" + FLAGS.label


def setup_summaries(logdir, sess):
    logdir_train = os.path.join(logdir, 'train')
    tf.gfile.MakeDirs(logdir_train)
    logdir_valid = os.path.join(logdir, 'valid')
    tf.gfile.MakeDirs(logdir_valid)
    train_summary_writer = tf.summary.FileWriter(logdir_train, sess.graph)
    valid_summary_writer = tf.summary.FileWriter(logdir_valid)
    tf.logging.info("Will log training summaries to %s.", logdir_train)
    tf.logging.info("Will log validation summaries to %s.", logdir_valid)

    return train_summary_writer, valid_summary_writer


def gauss_newton(f, J, x0) -> OptimizationResult:
    # TODO(andrei): Modularize better
    # TODO(andrei): How to deal with sparse Jacobians?

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Some parameter we don't really care about
        lbd = tf.convert_to_tensor(2.0, tf.float32)
        cx = x0
        n_steps = 20

        # print(x0)
        # print(f(x0, lbd))
        # print(J(x0, lbd))

        loss_hist = [None] * n_steps
        for i in range(n_steps):
            next_x = gn_iteration(cx, J(cx, lbd), f(cx, lbd), 1.0, i, loss_hist)
            cx = next_x

        print("Loss hist is now:")
        print(loss_hist)

        # This node is the output of the optimization algorithm, which we may
        # want to feed into further computations or a loss function.
        final_x = cx

        summary_op = tf.summary.merge_all()
        logdir = get_logdir(LOGDIR_BASE)
        train_summary_writer, _ = setup_summaries(logdir, sess)

        x_val, summary = sess.run([final_x, summary_op])
        train_summary_writer.add_summary(summary)
        train_summary_writer.flush()

        # print("Final value of function:")
        final_f = sess.run(dummy_f(final_x, lbd))

        # print("Final (scalar) value of function:")
        final_F = sess.run(tf.reduce_mean(tf.square(dummy_f(final_x, lbd))))
        # print(final_value)

        # print("Final value of Jacobian:")
        final_jacobian = sess.run(dummy_f_jacobian(final_x, lbd))
        # print(final_jacobian)


        return OptimizationResult(
            name="TensorFlow GN",
            steps=n_steps,
            f_value=final_f,
            F_value=final_F,
            x_result=x_val,
            final_jacobian=final_jacobian
        )



def main():
    x0_raw = [5]
    x0 = tf.constant(x0_raw, dtype=tf.float32)
    x0 = tf.reshape(x0, [1, 1])
    gn_tf_res = gauss_newton(dummy_f, dummy_f_jacobian, x0)

    # This prolly has to go since it makes things annoyingly complicated.
    LBD = 2.0
    # Note: for sparse constrained problems, SciPy uses the Trust Region
    # Reflective algorithm, which is particularly suitable for large sparse
    # problems with bounds. Generally robust method.
    ff = lambda x: dummy_f_np(x, LBD)
    JJ = lambda x: dummy_f_jacobian_np(x, LBD)
    scipy_res = sp_optimize.least_squares(ff, x0_raw, JJ,
                                          method='lm', verbose=1)

    print("Result obtained by scipy:")
    print(scipy_res)

    print("Result obtained by Ceres:")
    print("TODO")

    print("Results")
    print("-" * 80)
    print("| Method \t | Steps | Cost \t | Final Jacobian (^T) |".expandtabs())
    print("-" * 80)

    print("{} \t | {} \t | {:.6f} \t | {} \t |".format(
        "TensorFlow-GN",
        gn_tf_res.steps,
        gn_tf_res.F_value,
        gn_tf_res.final_jacobian.T
    ).expandtabs())
    print("{} \t | {} \t | {:.6f} \t | {} \t |".format(
        "SciPy-LM",
        scipy_res.nfev,     # Not really the "step" count
        scipy_res.cost,
        scipy_res.fun,
        scipy_res.jac
    ).expandtabs())

    # TODO(andrei): Also ensure to (1) script all of this and (2) write
    # actual tests which check the quality of the solutions automatically.


if __name__ == '__main__':
    main()
