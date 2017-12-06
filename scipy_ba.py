"""Very simple Bundle Adjuster using a scipy solver.

Heavily based on the BA code from the SciPy Cookbook:
https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

# TODO(andrei): Are solvers using the Schur complement available?
# TODO(andrei): Are solvers using sparse methods available?

import random
import time
import os
import pickle

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from mpl_toolkits.mplot3d import Axes3D

from problem import BundleAdjustmentProblem, BALBundleAdjustmentProblem


# TODO(andrei): Refactor classes so that this works with a generic
# 'BundleAdjustmentProblem' instance.
def solve(problem: BALBundleAdjustmentProblem):
    """
    Warning: estimates Jacobians using finite differences!

    The `*_indices` arrays determine which points go with which cameras.
    """
    n_cameras = problem.camera_params.shape[0]
    n_points = problem.points_3d.shape[0]

    plot_results = False

    n = 9 * n_cameras + 3 * n_points
    m = 2 * problem.points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((problem.camera_params.ravel(), problem.points_3d.ravel()))
    x0_copy = np.copy(x0)
    f0 = fun(x0, n_cameras, n_points, problem.camera_indices,
             problem.point_indices, problem.points_2d)

    # plt.ion()

    # TODO(andrei): Flag to select how to compute jacobians.
    # plt.plot(f0)

    A = ba_sparsity(n_cameras, n_points, problem.camera_indices, problem.point_indices)
    print("Jacobian (mask) shape: {}".format(A.shape))

    load = False
    if load and os.path.exists("dump/sfm-dump.npy"):
        with open("dump/sfm-dump.npy", "rb") as f:
            res = pickle.load(f)
    else:
        t0 = time.time()
        res = least_squares(fun,
                            x0,
                            # Setting this but not 'jac' leads to finite
                            # differences being used to approximate the (
                            # sparse) Jacobian at every frame.
                            # jac_sparsity=A,
                            jac=jac,
                            verbose=2,
                            # Scale the variables to equalize their influence on
                            # the cost function. Very important, since the camera
                            # parameters and the 3D points are very different
                            # entities.
                            x_scale='jac',
                            max_nfev=30,        # Strict but quick TODO
                            ftol=1e-4,
                            method='trf',
                            # loss='soft_l1', # seems to work way better than huber/cauchy for BA
                            # Substantially better than linear loss, but MUCH
                            # slower (gets 7300 on the first ladybug dataset,
                            # as opposed to 11300 with the linear loss).
                            args=(n_cameras, n_points, problem.camera_indices,
                                  problem.point_indices, problem.points_2d),
                            )
        t1 = time.time()
        print("Delta time = {:2f}s".format(t1 - t0))
        print("Saving numpy dump.")
        with open("dump/sfm-dump.npy", "wb") as f:
            pickle.dump(res, f)

    # plt.plot(res.fun)
    # plt.pause(1.0)
    # plt.waitforbuttonpress()
    # plt.show()

    if plot_results:
        res_struct = res.x[n_cameras * 9:].reshape(n_points, 3)
        init_struct = x0_copy[n_cameras * 9:].reshape(n_points, 3)
        deltas = np.linalg.norm(res_struct - init_struct, 2, axis=1).reshape(-1, 1)

        render_structure(x0_copy, n_cameras, n_points, "Initial structure")
        render_structure(res.x, n_cameras, n_points, "Refined structure",
                         deltas=deltas)
        plt.show()

    return


def render_structure(x, n_cameras, n_points, title=None, **kw):
    structure = x[n_cameras * 9:].reshape(n_points, 3)
    print(structure.shape)

    sample = np.random.randint(0, structure.shape[0], 2500)
    deltas = kw.get('deltas', None)
    if deltas is not None:
        deltas = deltas[sample]
        d_thresh = np.median(deltas) * 5.0
        deltas[deltas > d_thresh] = d_thresh

        print(deltas.shape)
        print(structure[sample, 0].shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(structure[sample, 0],
               structure[sample, 2],
               structure[sample, 1],
               c=deltas,
               cmap=plt.get_cmap('viridis'),
               s=0.05,
               marker='x')
    if deltas is not None:
        plt.colormaps()

    viz_range = 2.5
    ax.set_xlabel("X")
    ax.set_xlim((-viz_range, viz_range))
    ax.set_ylabel("Y")
    ax.set_ylim((-viz_range * 2, 0.5))
    ax.set_zlabel("Z")
    ax.set_zlim((-0.5, viz_range))
    ax.set_title(title)

    cams = x[:n_cameras * 9].reshape(n_cameras, 9)
    cams_rot = cams[:, 0:3]
    cams_pos = cams[:, 3:6]

    # Very naive rendering of cameras
    # TODO(andrei): Ensure they get rendered right. Looks a bit funny right now.
    # TODO(andrei): Also show camera orientation and quantify the change in
    # their pose undergone after BA.
    ax.scatter(cams_pos[:, 0],
               cams_pos[:, 2],
               cams_pos[:, 1],
               s=5.0,
               c=np.linspace(0.0, 10.0, cams_pos.shape[0]),
               marker='o')


# TODO(andrei): Maybe even animate the plot? Consider downsampling for speed.

# def rotate_mine(points, rot_vecs):
#     """Rotate the points by the given rotation vectors.
#
#     Uses Rogrigues's formula."""
#
#     # Make a column vector with the rotation angles of each rotation vector.
#     theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
#
#     # This is cool! I didn't know about this numpy feature.
#     with np.errstate(invalid='ignore'):
#         # Normalize and de-NaN-ify everything.
#         v = rot_vecs / theta
#         v = np.nan_to_num(v)
#
#     dot = np.sum(points * v, axis = 1)[:, np.newaxis]
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#
#     return cos_theta * points + \
#            sin_theta * np.cross(v, points) + \
#            dot * (1 - cos_theta) * v


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    # Make a column vector with the rotation angles of each rotation vector.
    # axis = 1 => compute the operation for every row, so collapse the column
    #  count.
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def get_P(points, camera_params):
    points_rot = rotate(points, camera_params[:, :3])
    points_trans = points_rot + camera_params[:, 3:6]
    return points_trans


def project(points, camera_params):
    """Project n 3D points to 2D."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, 0:2] / points_proj[:, 2, np.newaxis]

    f = camera_params[:, 6]

    # Apply the radial distortion
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]

    d = np.sum(points_proj ** 2, axis = 1)
    r = 1 + k1 * d + k2 * d**2

    # First LB dataset, no radial: converges more slowly, but does eventually
    # converge to cost ~ 15000, which is decent.
    # r = 1

    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute the residuals for each observation.

    Note that every 2D point produces two residuals.

    Args:
        params: Camera parameters and 3D coordinates, i.e., the stuff we wish to
                optimize the 2D positions of the points are fixed.

    Returns:
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])

    # TODO(andrei): Compute the Jacobian of this function insted of relying
    # on finite differences.
    residuals = (points_proj - points_2d).ravel()

    # TODO(andrei): Treat this like a hook.
    # if random.random() < 0.05:
    #     # Don't plot every single time, since this function actually gets
    #     # called A LOT when estimating the Jacobian using finite differences.
    #     plt.figure()
    #     plt.plot(residuals[::2])
    #     plt.ylim((-6.0, 6.0))
    #     plt.pause(0.5)

    return residuals


def jac_pproj(P):
    """Computes the Jacobian of the BAL perspective projection function.
    The function takes 3D points and outputs 2D points on the (metric) camera
    plane, so the Jacobian is 2x3.

    TODO(andrei): Find clean way of making this able to operate on batches of
    points.

    See Also:
        https://grail.cs.washington.edu/projects/bal/

    Warnings:
        Note that this particular projective model has an additional minus
        sign in the front, so X and Y are flipped after projection.

    Args:
        P: The 3D point in camera coordinates.
    """
    Px = P[0]
    Py = P[1]
    Pz = P[2]
    Pz2 = Pz * Pz
    return np.array([
        [-1.0 / Pz,          0,  Px / Pz2],
        [        0,  -1.0 / Pz,  Py / Pz2]
    ])


def jac(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    # TODO(andrei): Compute the Jacobian, which may be huge.
    # [ grad of out_1 w.r.t. all params ]
    # [ grad of out_2 w.r.t. all params ]
    # [ grad of out_3 w.r.t. all params ]
    # ...
    # params: 3D points and camera params.
    # outputs: every reprojection error (every 3D point in every camera in
    # which it's visible).
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)

    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    f = camera_params[:, 6]
    # TODO(andrei): Use the dude's examples for how to do calculations on
    # entire batches!

    # P = np.array([1, 2, 3]).reshape((3, 1))
    # Ps = np.array([
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3],
    # ]).T.reshape((3, 5))
    # print(P)
    # print(Ps)
    # print(jac_pproj(P).shape)
    # print(jac_pproj(P))
    # print(jac_pproj(Ps).shape)
    # print(jac_pproj(Ps))

    # For every observation, i.e., every (camera, 3D point) pair.
    # print("Estimating jacobian ({}) steps...".format(camera_indices.size))
    # for i in range(camera_indices.size):
    #     if (i+1) % 1000 == 0:
    #         print("Step {}".format(i+1))
    # If this operation is NOT vectorized, it can take about ONE MINUTE to
    # compute the Jacobian for a single 5-frame problem...

    # We need these in pretty much all jacobian computations
    # This is an (n_observations) x 3 array. That is, we compute the point P
    # for every world point X, for every camera in which it is visible.
    P = get_P(points_3d[point_indices], camera_params[camera_indices])
    print("[jac] P.shape = {}".format(P.shape))

    # Here be dragons!
    Px = P[:, 0]
    Px2 = Px * Px
    Py = P[:, 1]
    Py2 = Py * Py
    Pz = P[:, 2]
    Pz2 = Pz * Pz

    d_x_wrt_r1 = - (Px * Py / Pz2)
    d_y_wrt_r1 = - 1 - Py2 / Pz2

    d_x_wrt_r2 = 1 + Px2 / Pz2
    d_y_wrt_r2 = Px * Py / Pz2

    d_x_wrt_r3 = -Py / Pz
    d_y_wrt_r3 = Px / Pz

    d_x_wrt_t1 = -1.0 / Pz
    d_y_wrt_t1 = 0.0
    d_x_wrt_t2 = 0.0
    d_y_wrt_t2 = -1.0 / Pz
    d_x_wrt_t3 = Px / Pz2
    d_y_wrt_t3 = Py / Pz2

    # Deriv of i-th point's x wrt rotation component 1
    A[2 * i, camera_indices * 9 + 0] = f * d_x_wrt_r1
    # Deriv of i-th point's y wrt rotation component 1
    A[2 * i + 1, camera_indices * 9 + 0] = f * d_y_wrt_r1

    # Deriv wrt rotation component 2
    A[2 * i, camera_indices * 9 + 1] = f * d_x_wrt_r2
    A[2 * i + 1, camera_indices * 9 + 1] = f * d_y_wrt_r2

    # Deriv wrt rotation component 3
    A[2 * i, camera_indices * 9 + 2] = f * d_x_wrt_r3
    A[2 * i, camera_indices * 9 + 2] = f * d_y_wrt_r3

    # Deriv wrt translation component 1
    A[2 * i, camera_indices * 9 + 3] = f * d_x_wrt_t1
    A[2 * i, camera_indices * 9 + 3] = f * d_y_wrt_t1

    # Deriv wrt translation component 2
    A[2 * i, camera_indices * 9 + 4] = f * d_x_wrt_t2
    A[2 * i, camera_indices * 9 + 4] = f * d_y_wrt_t2

    # Deriv wrt translation component 3
    A[2 * i, camera_indices * 9 + 5] = f * d_x_wrt_t3
    A[2 * i, camera_indices * 9 + 5] = f * d_y_wrt_t3

    # Deriv wrt focal length (extra thing over the CVPR '14 tutorial)
    A[2 * i, camera_indices * 9 + 6] = 0
    A[2 * i, camera_indices * 9 + 6] = 0

    # Deriv wrt first radial distortion param (ignored, for now)
    A[2 * i, camera_indices * 9 + 7] = 0
    A[2 * i, camera_indices * 9 + 7] = 0

    # Deriv wrt second radial distortion param (ignored, for now)
    A[2 * i, camera_indices * 9 + 8] = 0
    A[2 * i, camera_indices * 9 + 8] = 0

    # TODO(andrei): See if stacking together all projection jacobians as a
    # (n_obs) x 2 x 3 tensor is possible.

    # Construct a rotation matrix for each axis-angle representation
    R = rodrigues(camera_params[:, 0:3])

    # Deriv wrt point's 3D coord y
    A[2 * i, n_cameras * 9 + point_indices * 3 + 0] = 1
    A[2 * i + 1, n_cameras * 9 + point_indices * 3 + 0] = 1

    # Deriv wrt point's 3D coord y
    A[2 * i, n_cameras * 9 + point_indices * 3 + 1] = 1
    A[2 * i + 1, n_cameras * 9 + point_indices * 3 + 1] = 1

    # Deriv wrt point's 3D coord z
    A[2 * i, n_cameras * 9 + point_indices * 3 + 2] = 1
    A[2 * i + 1, n_cameras * 9 + point_indices * 3 + 2] = 1

    return A


def ba_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """Compute a sparsity mask to make the Jacobian estimation tractable."""
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        # For every point, the part. deriv of the i-th observation's x,
        # w.r.t. the 9 camera parameters.
        A[2 * i, camera_indices * 9 + s] = 1
        # same but for the y coord.
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        # For every point, the partial derivative of the i-th observation's
        # x, w.r.t. the 3D point coords.
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

