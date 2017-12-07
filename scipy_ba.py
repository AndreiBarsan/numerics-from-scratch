"""Very simple Bundle Adjuster using a scipy solver.

Based on the BA code from the SciPy Cookbook (apart from the analytic Jacobian
formula):
https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

# TODO(andrei): Are built-in solvers using the Schur complement available?

import random
import time
import os
import pickle

import numdifftools as nd

from algebra import skew
from lie import SO3

# Configure matplotlib before loading the plotting component.
import matplotlib
matplotlib.rc('font', size='8')
# This seems the least slow way of visualizing stuff in 3D. The mayavi library
# may be better, but it requires a local build of VTK with Python 3 support.
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
import scipy.optimize as sopt

from mpl_toolkits.mplot3d import Axes3D

from problem import BundleAdjustmentProblem, BALBundleAdjustmentProblem


# TODO(andrei): Refactor classes so that this works with a generic
# 'BundleAdjustmentProblem' instance.
def solve(problem: BALBundleAdjustmentProblem, **kw):
    n_cameras = problem.camera_params.shape[0]
    n_points = problem.points_3d.shape[0]

    # 5 frames, f but no k, anal: 4.7e+02   (negative skew in deriv)
    # 5 frames, f but no k, anal: 3.3e+04   (positive skew in deriv)
    # 5 frames, f but no k, num:  4.7e+02
    # Conclusion => the sign is very important (No shit!).

    # 10 frames, f but no k, anal: 2.09e+05 (pos skew)
    # 10 frames, f but no k, anal: 3.52e+03 (neg skew)
    # 10 frames, f but no k, num:  1.66e+03
    # Conclusion => there's still bugs, but neg skew seems to be the correct one.

    # 15 frames, f but no k, anal: 1.45e+04 (neg skew)
    # 15 frames, f but no k, anal: 2.08e+05 (pos skew)
    # 15 frames, f but no k, num:  2.73e+03
    # Conclusion => definitely still bugs

    plot_results = kw.get('plot_results', True)
    analytic_jacobian = kw.get('analytic_jacobian', False)

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
    # plt.plot(f0)

    optimization_kwargs = {
        'verbose': 2,
        # Scale the variables to equalize their influence on
        # the cost function. Very important, since the camera
        # parameters and the 3D points are very different
        # entities.
        'x_scale': 'jac',
        # 'max_nfev': 50,        # Strict but quick
        'ftol': 1e-4,
        'method': 'trf',
        # loss='soft_l1', # seems to work way better than huber/cauchy for BA
        # Substantially better than linear loss, but MUCH
        # slower (gets 7300 on the first ladybug dataset,
        # as opposed to 11300 with the linear loss).
        'args': (n_cameras, n_points, problem.camera_indices,
                 problem.point_indices, problem.points_2d),
    }
    if analytic_jacobian:
        optimization_kwargs['jac'] = jac_clean
    else:
        print("Estimating Jacobian using finite differences.")
        A = ba_sparsity(n_cameras, n_points, problem.camera_indices, problem.point_indices)
        print("Jacobian (mask) shape: {}".format(A.shape))

        # Use finite differences to estimate the Jacobian numerically if we're
        # to lazy to code it properly.
        # Setting this but not 'jac' leads to finite
        # differences being used to approximate the (
        # sparse) Jacobian at every frame.
        optimization_kwargs['jac_sparsity'] = A

    # Enable this if you just want to work on the visualization.
    load = False
    if load and os.path.exists("dump/sfm-dump.npy"):
        with open("dump/sfm-dump.npy", "rb") as f:
            res = pickle.load(f)
    else:
        t0 = time.time()
        res = sopt.least_squares(fun, x0, **optimization_kwargs)
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
        render_structure(res.x, n_cameras, n_points, "Refined structure\n(colors reflect the degree to which\na particular point was adjusted)",
                         deltas=deltas)
        plt.show()

    return


def render_structure(x, n_cameras, n_points, title=None, **kw):
    """Renders the 3D scene being reconstructed as a point cloud.

    Args:
        x:          The parameter vector containing 3D points and camera params.
        n_cameras:  The number of cameras in the problem.
        n_points:   The number of 3D points in the problem.
        title:      The title of the plot (optional).
    """
    # TODO(andrei): Maybe even animate the 3D plot? Consider downsampling for speed.
    structure = x[n_cameras * 9:].reshape(n_points, 3)
    print(structure.shape)

    min_3d_points_to_render = 1500
    sample = np.random.choice(
        np.arange(0, structure.shape[0]),
        min(min_3d_points_to_render, structure.shape[0]),
        False)
    deltas = kw.get('deltas', None)
    if deltas is not None:
        deltas = deltas[sample]
        d_thresh = np.median(deltas) * 3.0
        deltas[deltas > d_thresh] = d_thresh

        print(deltas.shape)
        print(structure[sample, 0].shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = ax.scatter(structure[sample, 0],
               structure[sample, 2],
               structure[sample, 1],
               c=deltas,
               cmap=plt.get_cmap('inferno'),
               s=0.075,
               marker='X')
    if deltas is not None:
        fig.colorbar(plot)

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

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    The Rodrigues rotation formula is used.
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

    # TODO(andrei): Test if doing this is the same as computing the rotation
    # matrix and multiplying by it!!!

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
    r = 1

    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, check=None):
    """Compute the residuals for each observation.

    Note that every 2D point produces two residuals.

    Args:
        params: Camera parameters and 3D coordinates, i.e., the stuff we wish to
                optimize the 2D positions of the points are fixed.
        check: Only used by the Jacobian computation.

    Returns:
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])

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


def jac_clean(params, n_cameras, n_points, camera_indices, point_indices, points_2d, check=True):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    # i = np.arange(camera_indices.size)

    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])

    # Even for two frames the jacobian is 3400x4000, so 1.2M parameters...
    # Yes, mostly empty, but still...
    J = lil_matrix((m,n), dtype=float)

    points_proj_count = points_proj.shape[0]

    print("[jac] Starting Jacobian computation.")
    start_t = time.time()
    Ps = get_P(points_3d[point_indices], camera_params[camera_indices])
    print("[jac] Ps.shape = {}".format(Ps.shape))
    Ps_projected = - Ps[:, 0:2] / Ps[:, 2, np.newaxis]

    # NOTE: When using lil_matrices special care has to be taken to not incur
    # huge insertion costs.
    cam_param_count = 9
    for p_idx in range(points_proj_count):
        f = camera_params[camera_indices[p_idx], 6]

        x_idx = p_idx * 2
        cam_idx = camera_indices[p_idx]
        point_3d_idx = point_indices[p_idx]

        # TODO derive Jacobian blocks here as matrices, then assign to the
        # proper slot in the main Jacobian.

        R = SO3.exp(camera_params[camera_indices[p_idx], 0:3])
        P = Ps[p_idx]

        J_proj_wrt_P                = jac_pproj(P)
        J_transform_wrt_twist       = np.hstack((-skew(P), np.eye(3)))
        J_transform_wrt_delta_3d    = R

        # Jacobian wrt the extrinsic camera params
        h1 = f * np.dot(J_proj_wrt_P, J_transform_wrt_twist)
        # Deriv wrt point's 3D coord x
        h2 = f * np.dot(J_proj_wrt_P, J_transform_wrt_delta_3d)
        # Jacobian wrt the intrinsic camera param(s)
        h3 = Ps_projected[p_idx, :, np.newaxis]
        assert h1.shape == (2, 6)
        assert h2.shape == (2, 3)
        assert h3.shape == (2, 1)

        cam_off = cam_idx * cam_param_count
        J[x_idx:x_idx+2, cam_off:cam_off+6] = h1
        J[x_idx:x_idx+2, cam_off+6] = h3

        point_3d_off = n_cameras * cam_param_count + point_3d_idx * 3
        # print(J[x_idx:x_idx+2, off:off+3].shape)
        J[x_idx:x_idx+2, point_3d_off:point_3d_off+3] = h2

    end_t = time.time()
    delta_t_s = end_t - start_t
    print("[jac] Finished computing Jacobian after {} steps, in {:2f}.".format(
        points_proj_count, delta_t_s))

    J_csr = J.tocsr()
    min_val = J_csr.min()
    max_val = J_csr.max()
    val_range = max_val - min_val
    # print("J.min = {:.4f}".format(J.data.min()))
    # print("J.max = {:.4f}".format(J.data.max()))

    # # plt.spy(J)
    # denseJ = J_csr.todense()
    # # denseJ[J_csr.nonzero()] = 50
    # denseJ[denseJ != 0.0] = 100
    # plot = plt.imshow(denseJ, cmap=cm.viridis)
    # plt.colorbar(plot)
    # plt.show()

    if check:
        print("Performing gradient check:")
        # TODO(andrei): If this doesn't work, use numdifftools!
        from finite_differences import numeric_jacobian
        # Note: scipy does not support jacobian checking; we should use the
        # functionality from the pysfm project (finite_differences.py)
        # gradient_err = sopt.check_grad(fun, jac_clean, params,
        #     n_cameras, n_points, camera_indices, point_indices, points_2d, False
        # )
        # print("Estimated gradient error:", gradient_err)

    return J_csr


def jac_slow(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    # i = np.arange(camera_indices.size)

    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])

    # Even for two frames the jacobian is 3400x4000, so 1.2M parameters...
    # Yes, mostly empty, but still...
    J = lil_matrix((m,n), dtype=float)

    points_proj_count = points_proj.shape[0]

    print("[jac] Starting Jacobian computation.")
    start_t = time.time()
    P = get_P(points_3d[point_indices], camera_params[camera_indices])
    print("[jac] P.shape = {}".format(P.shape))

    # Here be dragons!
    Pxs = P[:, 0]
    Pxs2 = Pxs * Pxs
    Pys = P[:, 1]
    Pys2 = Pys * Pys
    Pzs = P[:, 2]
    Pzs2 = Pzs * Pzs

    Ps_projected = - P[:, 0:1] / P[:, 2]

    cam_param_count = 9
    for p_idx in range(points_proj_count):
        f = camera_params[camera_indices[p_idx], 6]

        x_idx = p_idx * 2
        y_idx = p_idx * 2 + 1
        cam_idx = camera_indices[p_idx]
        point_3d_idx = point_indices[p_idx]

        Px = Pxs[p_idx]
        assert Px.shape == (), "Px must be a scalar!"
        Py = Pys[p_idx]
        Pz = Pzs[p_idx]
        Px2 = Pxs2[p_idx]
        Py2 = Pys2[p_idx]
        Pz2 = Pzs2[p_idx]

        # TODO(andrei): Could batch-compute these bad boys as well!
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

        # rot 1
        J[x_idx, cam_idx * cam_param_count + 0] = f * d_x_wrt_r1
        J[y_idx, cam_idx * cam_param_count + 0] = f * d_y_wrt_r1
        # rot 2
        J[x_idx, cam_idx * cam_param_count + 1] = f * d_x_wrt_r2
        J[y_idx, cam_idx * cam_param_count + 1] = f * d_y_wrt_r2
        # rot 3
        J[x_idx, cam_idx * cam_param_count + 2] = f * d_x_wrt_r3
        J[y_idx, cam_idx * cam_param_count + 2] = f * d_y_wrt_r3
        # trans 1
        J[x_idx, cam_idx * cam_param_count + 3] = f * d_x_wrt_t1
        J[y_idx, cam_idx * cam_param_count + 3] = f * d_y_wrt_t1
        # trans 2
        J[x_idx, cam_idx * cam_param_count + 4] = f * d_x_wrt_t2
        J[y_idx, cam_idx * cam_param_count + 4] = f * d_y_wrt_t2
        # trans 3
        J[x_idx, cam_idx * cam_param_count + 5] = f * d_x_wrt_t3
        J[y_idx, cam_idx * cam_param_count + 5] = f * d_y_wrt_t3
        # f
        J[x_idx, cam_idx * cam_param_count + 6] = Ps_projected[p_idx, 0]     # should just be p, the projected point coords (x here)
        J[y_idx, cam_idx * cam_param_count + 6] = Ps_projected[p_idx, 1]     # ...and y here!

        # Skip the two radial distortion parameters for now.
        # TODO(andrei): Support optimizing for the radial distortion, too!

        # TODO(andrei): Could batch-compute this, too!
        R = SO3.exp(camera_params[camera_indices[p_idx], 0:3])

        # Deriv wrt point's 3D coord x
        J[x_idx, n_cameras * cam_param_count + point_3d_idx * 3 + 0] = f * (-R[0, 0] / Pz + R[2, 0] * Px / Pz2)
        J[y_idx, n_cameras * cam_param_count + point_3d_idx * 3 + 0] = f * (-R[1, 0] / Pz + R[2, 2] * Py / Pz2)

        # Deriv wrt point's 3D coord y
        J[x_idx, n_cameras * cam_param_count + point_3d_idx * 3 + 1] = f * (-R[0, 1] / Pz + R[2, 1] * Px / Pz2)
        J[y_idx, n_cameras * cam_param_count + point_3d_idx * 3 + 1] = f * (-R[1, 1] / Pz + R[2, 1] * Py / Pz2)

        # Deriv wrt point's 3D coord z
        J[x_idx, n_cameras * cam_param_count + point_3d_idx * 3 + 2] = f * (-R[0, 2] / Pz + R[2, 2] * Px / Pz2)
        J[y_idx, n_cameras * cam_param_count + point_3d_idx * 3 + 2] = f * (-R[1, 2] / Pz + R[2, 2] * Py / Pz2)

    end_t = time.time()
    delta_t_s = end_t - start_t
    print("[jac] Finished computing Jacobian after {} steps, in {:2f}.".format(
        points_proj_count, delta_t_s))
    return J


def jac_old(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    # [ grad of out_1 w.r.t. all params ]
    # [ grad of out_2 w.r.t. all params ]
    # [ grad of out_3 w.r.t. all params ]
    # ...
    # params: 3D points and camera params.
    # outputs: every reprojection error (every 3D point in every camera in
    # which it's visible).
    raise ValueError("Unsupported method. Please don't use me!")

    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=float)

    i = np.arange(camera_indices.size)

    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    f = camera_params[:, 6]

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

    # TODO(andrei): Ideally, we'd modularize these computations, but it's
    # tricky to decompose them nicely (e.g., to follow the chain rule used to
    # derive them in the first place), since it would involve multiplying
    # together sparse 3D tensors (2D jacobians across the entire dataset),
    # which I'm not sure how to do in numpy.

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

    print(A.shape)
    print(A[i].shape)
    print(A[i, :].shape)
    print(A[i, 3].shape)

    for camera_idx in camera_indices:
        # Deriv of i-th point's x wrt rotation component 1
        A[2 * i, camera_idx * 9 + 0] = f * d_x_wrt_r1
        # Deriv of i-th point's y wrt rotation component 1
        A[2 * i + 1, camera_idx * 9 + 0] = f * d_y_wrt_r1

        # Deriv wrt rotation component 2
        A[2 * i, camera_idx * 9 + 1] = f * d_x_wrt_r2
        A[2 * i + 1, camera_idx * 9 + 1] = f * d_y_wrt_r2

        # Deriv wrt rotation component 3
        A[2 * i, camera_idx * 9 + 2] = f * d_x_wrt_r3
        A[2 * i, camera_idx * 9 + 2] = f * d_y_wrt_r3

        # Deriv wrt translation component 1
        A[2 * i, camera_idx * 9 + 3] = f * d_x_wrt_t1
        A[2 * i, camera_idx * 9 + 3] = f * d_y_wrt_t1

        # Deriv wrt translation component 2
        A[2 * i, camera_idx * 9 + 4] = f * d_x_wrt_t2
        A[2 * i, camera_idx * 9 + 4] = f * d_y_wrt_t2

        # Deriv wrt translation component 3
        A[2 * i, camera_idx * 9 + 5] = f * d_x_wrt_t3
        A[2 * i, camera_idx * 9 + 5] = f * d_y_wrt_t3

        # Deriv wrt focal length (extra thing over the CVPR '14 tutorial)
        # TODO(andrei): Add this later.
        A[2 * i, camera_idx * 9 + 6] = 0
        A[2 * i, camera_idx * 9 + 6] = 0

        # Deriv wrt first radial distortion param (ignored, for now)
        A[2 * i, camera_idx * 9 + 7] = 0
        A[2 * i, camera_idx * 9 + 7] = 0

        # Deriv wrt second radial distortion param (ignored, for now)
        A[2 * i, camera_idx * 9 + 8] = 0
        A[2 * i, camera_idx * 9 + 8] = 0

    # TODO(andrei): See if stacking together all projection jacobians as a
    # (n_obs) x 2 x 3 tensor is possible.

    # Construct a rotation matrix for each axis-angle representation
    R = SO3.exp(camera_params[:, 0:3])

    # Deriv wrt point's 3D coord x
    A[2 * i, n_cameras * 9 + point_indices * 3 + 0] = f * -R(0,0) / Pz + R(2, 0) * Px / Pz2
    A[2 * i + 1, n_cameras * 9 + point_indices * 3 + 0] = f * R(1, 0) / Pz + R(2, 2) * Py / Pz2

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

    # TODO: when testing do tests for extrinsic only, ext+f, and ext+f+k1+k2 !!!

    i = np.arange(camera_indices.size)
    # for s in range(9):
    for s in range(7):      # Use f, but not radial dist. params
    # for s in range(6):  # Ignore f and radial distortion params.
        # For every point, the part. deriv of the i-th observation's x,
        # w.r.t. the 9 camera parameters.
        # Masks every x and every y where the camera is present
        A[2 * i, camera_indices * 9 + s] = 1
        # same but for the y coord.
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        # For every point, the partial derivative of the i-th observation's
        # x, w.r.t. the 3D point coords.
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

