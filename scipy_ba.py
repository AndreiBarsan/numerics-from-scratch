"""Very simple Bundle Adjuster using a scipy solver.

Based on the BA code from the SciPy Cookbook (apart from the analytic Jacobian
formula):
https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
# TODO(andrei): Are built-in solvers using the Schur complement available?

from enum import Enum
import logging
import os
import pickle
import time

# Configure matplotlib before loading the plotting component.
import matplotlib

# Needed even if unused!
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize._numdiff import approx_derivative, check_derivative

matplotlib.rc('font', size='8')
# This seems the least slow way of visualizing stuff in 3D. The mayavi library
# may be better, but it requires a local build of VTK with Python 3 support.
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import scipy.optimize as sopt

from algebra import skew
from bundle_adjustment_result import BundleAdjustmentResult
from finite_differences import *
from lie import SO3, rotate
from problem import BALBundleAdjustmentProblem


logging.basicConfig(level=logging.INFO)

# The most severe bug I had was that I had two functions for transforming a
# point from world to camera coords, but I only updated only one of them to
# account for the reparameterization.

# TODO(andrei): Ensure your rotation method works ok. As a sanity check, instead
# of doing the batch rotations using 'rotate', just compute each camera's rot
# matrix (possibly with some CAREFUL caching) and use that to transform points.

class TransformMode(Enum):
    """Specifies the convention under which a camera pose is expressed."""
    # P = RX + t
    BAL = 1
    # P = R'(X - t)
    CANONICAL = 2


# TODO(andrei): Refactor classes so that this works with a generic
# 'BundleAdjustmentProblem' instance.
def solve(problem: BALBundleAdjustmentProblem, **kw) -> BundleAdjustmentResult:
    n_cameras = problem.camera_params.shape[0]
    n_points = problem.points_3d.shape[0]

    # Results from December 6
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

    # Results from December 7
    # 49 frames, f but no k, num, reparam: 1.505e+04
    # 49 frames, f but no k, num:          1.501e+04
    #
    # 49 frames, ceres, anal:   1.33e+04
    # 49 frames, num:           1.34e+04
    # 49 frames, cookbook, num: 1.34e+04
    # 49 frames, num, reparam:  1.34e+04 => looks like reparameterization is OK

    # Results from December 13
    # 5 frames, num:          4.7065e+03 after 14-20 iterations (max 20 fev)
    # 5 frames, anal:         4.7049e+03 after 15-20 iterations (max 20 fev)
    # 12 frames, num:         1.99989e+03 after 17-20 iterations (max 20 fev)
    # 12 frames, anal:        1.99910e+03 after 17-20 iterations (max 20 fev)

    # 15 frames, num:         2.7316e+03 after 23-27 iterations (max 30 fev)
    # 15 frames, anal:        2.7332e+03 after 21-25 iterations (max 30 fev)
    #
    # !!! First significant difference
    #     (MUCH slower convergence to slightly worse solution)
    # 20 frames, num:           3.9933e+03 after 10-13 iterations (max 30 fev)
    # 20 frames, anal:          4.2684e+03 after 26-30 iterations (max 30 fev)
    # 20 frames, anal, fast:    <TODO> (more vectorized stuff)
    # 20 frames, anal, patched: 3.9933e+03 after 10-13 iterations (max 30 fev)
    #
    # 25 frames, num:         5.11136e+03 after 9-11 iterations (max 30 fev)
    # 25 frames, anal:        6.07250e+03 after 26-30 iteratins (max 30 fev)
    #
    # 35 frames, num:         8.6530e+03 after 13-15 iterations (max 30 fev)
    # 35 frames, anal:        1.0559e+04 after 26-30 iterations (max 30 fev)

    # all frames, num:            1.5051e+04 after 16-19 iterations (max 30 fev)
    # all frames, anal, patched:  1.5043e+04 after 16-18 iterations (max 30 fev)

    # I don't know what could be causing this. I doubt it's numerical stuff,
    # especially since, overall, I'm not getting results as good as the ones
    # using finite differences or ceres. Actually, I should double check with
    # ceres (without radial distortion part).
    # I should do the same thing I do here with Ceres:
    #   - compute analytical jacobian in Ceres
    #     - remove radial distortion first
    #   - compute numerical Jacobian in Ceres (see gradient checker code)
    #   - compare by visualizing; if there's no gap, then there's an issue with
    #   how I'm computing the bad part of the jacobian. Otherwise, the issue is
    #   someplace else in my implementation.

    plot_results        = kw.get('plot_results', False)
    analytic_jacobian   = kw.get('analytic_jacobian', False)
    transform_mode      = kw.get('transform_mode', TransformMode.CANONICAL)
    check_jacobians     = kw.get('check_jacobians', True)
    # TODO(andrei): Error when leftover kwargs.

    if transform_mode != TransformMode.CANONICAL and analytic_jacobian:
        raise ValueError("The analytical jacobian computation is only "
                         "supported when the camera extrinsics are expressed "
                         "using the canonical parametrization.")

    n = 9 * n_cameras + 3 * n_points
    m = 2 * problem.points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((problem.camera_params.flatten(), problem.points_3d.flatten()))
    x0_copy = np.copy(x0)
    f0 = fun(x0, n_cameras, n_points, problem.camera_indices,
             problem.point_indices, problem.points_2d, transform_mode, False)

    # plt.ion()
    # plt.plot(f0)

    optimization_kwargs = {
        'verbose': 2,
        # Scale the variables to equalize their influence on
        # the cost function. Very important, since the camera
        # parameters and the 3D points are very different
        # entities.
        'x_scale': 'jac',
        'ftol': 1e-4,
        'method': 'trf',
        # loss='soft_l1', # seems to work way better than huber/cauchy for BA
        # Substantially better than linear loss, but MUCH
        # slower (gets 7300 on the first ladybug dataset,
        # as opposed to 11300 with the linear loss).
        'args': (n_cameras, n_points, problem.camera_indices,
                 problem.point_indices, problem.points_2d, transform_mode,
                 check_jacobians),
        'max_nfev': kw.get('max_nfev', None),
    }
    if analytic_jacobian:
        print("Computing Jacobian analytically")
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

    ba_result = BundleAdjustmentResult(res.cost)
    return ba_result


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

    max_3d_points_to_render = 1500
    sample = np.random.choice(
        np.arange(0, structure.shape[0]),
        min(max_3d_points_to_render, structure.shape[0]),
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

    # From the docs:
    #   - R' * [0 0 -1]' = a camera's viewing dir
    #   - -R' * t        = a camera's position

    neg_zs = np.zeros_like(cams_pos)
    neg_zs[:, 2] = -1

    cams_3d_pos = rotate(cams_pos, cams_rot)
    cams_3d_dir = rotate(neg_zs, cams_rot)

    # Very naive rendering of cameras
    # TODO(andrei): Ensure they get rendered right. Looks a bit funny right now.
    # TODO(andrei): Also show camera orientation and quantify the change in
    # their pose undergone after BA. (plot the camera axes, making the forward,
    # i.e., -Z in the BAL convention, an arrow).
    ax.scatter(cams_3d_pos[:, 0],
               cams_3d_pos[:, 2],
               cams_3d_pos[:, 1],
               s=0.5,
               c=np.linspace(0.0, 10.0, cams_pos.shape[0]),
               marker='o')
    ax.quiver(cams_3d_pos[:, 0],
              cams_3d_pos[:, 2],
              cams_3d_pos[:, 1],
              cams_3d_dir[:, 0],
              cams_3d_dir[:, 2],
              cams_3d_dir[:, 1],
              length=0.5)


# TODO(andrei): There is code duplication between this and 'project'. Get rid of it.
def get_P(points, camera_params, transform_mode):
    if transform_mode == TransformMode.BAL:
        points_rot = rotate(points, camera_params[:, :3])
        points_transformed = points_rot + camera_params[:, 3:6]
        return points_transformed
    elif transform_mode == TransformMode.CANONICAL:
        points_off = points - camera_params[:, 3:6]
        points_transformed = rotate(points_off, -camera_params[:, :3])
        return points_transformed


def project(points, camera_params, transform_mode):
    """Project n 3D points to 2D."""
    if transform_mode == TransformMode.BAL:
        points_proj = rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
    else:
        # This is the formulation presented in, e.g., the CVPR '14 Tutorial by
        # Prof. Frank Dellaert.
        # P = R' * (X - t)
        points_off = points - camera_params[:, 3:6]
        # Note: negating an axis-angle representation is the same as transposing
        # (i.e., inverting the otherwise orthogonal) corresponding rotation
        # matrix.
        points_proj = rotate(points_off, -camera_params[:, :3])

    points_proj = -points_proj[:, 0:2] / points_proj[:, 2, np.newaxis]

    f = camera_params[:, 6]

    # Apply the radial distortion
    # k1 = camera_params[:, 7]
    # k2 = camera_params[:, 8]

    # d = np.sum(points_proj ** 2, axis = 1)
    # r = 1 + k1 * d + k2 * d**2

    # First LB dataset, no radial: converges more slowly, but does eventually
    # converge to cost ~ 15000, which is decent.
    # TODO(andrei): Proper flag to toggle this!
    r = 1

    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, transform_mode, check=None):
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
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], transform_mode)

    residuals = (points_proj - points_2d).flatten()

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
    assert Px.shape == Py.shape
    assert Py.shape == Pz.shape
    assert Pz.shape == Pz2.shape
    return np.array([
        [-1.0 / Pz,          0,  Px / Pz2],
        [        0,  -1.0 / Pz,  Py / Pz2]
    ])


def gvnn_deriv(omg, R):
    # XXX document this or remove
    for i in range(3):
        pass


def jac_clean(params, n_cameras, n_points, camera_indices, point_indices, points_2d, transform_mode, check_jacobians):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3

    # i = np.arange(camera_indices.size)

    # TODO(andrei): This looks VERY redundant.
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], transform_mode)

    # Even for two frames the jacobian is 3400x4000, so 1.2M parameters...
    # Yes, mostly empty, but still...
    J = lil_matrix((m,n), dtype=float)

    points_proj_count = points_proj.shape[0]

    print("[jac] Starting Jacobian computation.")
    start_t = time.time()
    Ps = get_P(points_3d[point_indices], camera_params[camera_indices], transform_mode)
    print("[jac] Ps.shape = {}".format(Ps.shape))
    Ps_projected = - Ps[:, 0:2] / Ps[:, 2, np.newaxis]
    print("[jac] Ps_projected.shape = {}".format(Ps_projected.shape))

    # NOTE: When using lil_matrices special care has to be taken to not incur
    # huge insertion costs.
    cam_param_count = 9
    for p_idx in range(points_proj_count):
        f = camera_params[camera_indices[p_idx], 6]
        P_world = points_3d[point_indices[p_idx]]

        x_idx = p_idx * 2
        cam_idx = camera_indices[p_idx]
        point_3d_idx = point_indices[p_idx]

        # TODO(andrei): Look at what derivatives are used in the gvnn paper and
        # in the Torch (Lua) source code.

        # The axis-angle representation of the cam. rotation
        omg = camera_params[camera_indices[p_idx], 0:3]
        # Wait, is this sensible? We should prolly use the full ugly formula...
        t_cam = camera_params[camera_indices[p_idx], 3:6]

        R = SO3.exp(omg)
        P = Ps[p_idx]

        J_proj_wrt_P                = jac_pproj(P)
        # The approach from the 2014 CVPR tutorial. Does not seem to work.
        # J_transform_wrt_omega = skew(P)
        #
        # NOPE PILE
        # J_transform_wrt_omega = eye(3)
        # J_transform_wrt_omega = skew(P_world)
        # J_transform_wrt_omega = np.dot(R.transpose(), skew(P_world))
        # J_transform_wrt_omega = np.dot(-R.transpose(), skew(P_world))
        # J_transform_wrt_omega = np.dot(R, skew(P_world))
        # J_transform_wrt_omega = skew(np.dot(R, P_world))
        # J_transform_wrt_omega = skew(-np.dot(R, P_world))
        # J_transform_wrt_omega = -skew(np.dot(R, P_world))
        # J_transform_wrt_omega = skew(np.dot(R.transpose(), P_world))
        # J_transform_wrt_omega = skew(np.dot(R, P)) Almost OK
        J_transform_wrt_omega = skew(np.dot(R.transpose(), P))   # Almost OK
        # J_transform_wrt_omega = np.dot(R, skew(P))
        # J_transform_wrt_omega = np.dot(R.transpose(), skew(P))
        # J_transform_wrt_omega = skew(np.dot(-R.transpose(), P))
        # J_transform_wrt_omega = skew(-np.dot(R.transpose(), P))
        # END NOPE PILE

        # ensure you use the right translation
        # J_transform_wrt_omega = skew(np.dot(R.transpose(), P_world - t_cam))

        # J_transform_wrt_twist       = np.dot(R.transpose(), np.hstack((J_transform_wrt_omega, -np.eye(3))))
        # December 14: I briefly skimmed doc/math.pdf from the GTSAM docs. Found
        # some relevant stuff but I need to read more carefully. -R.transpose()
        # is the correct derivative of the second half of the matrix!! It leads
        # to basically zero error.
        J_transform_wrt_twist       = np.hstack((J_transform_wrt_omega, -R.transpose()))

        # J_transform_wrt_twist       = np.hstack((skew(P), -np.eye(3)))

        J_transform_wrt_delta_3d    = R.transpose()
        assert J_proj_wrt_P.shape == (2, 3)
        assert J_transform_wrt_twist.shape == (3, 6)
        assert J_transform_wrt_delta_3d.shape == (3, 3)

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

    # plt.figure()
    # plt.spy(J)
    # plt.title("spy(J)")
    #
    # plt.figure()
    # plt.spy(J, precision=0.001)
    # plt.title("spy(J) with prec")
    # plt.show()

    if check_jacobians:
        def fun_proxy(xxx):
            return fun(xxx, n_cameras, n_points, camera_indices,
                       point_indices, points_2d, transform_mode, check=False)
        def jac_proxy(jxx):
            return jac_clean(jxx, n_cameras, n_points, camera_indices,
                             point_indices, points_2d, transform_mode, check_jacobians=False)

        print("[jac][check] Estimating Jacobian numerically...")
        # Estimate the Jacobian numerically using a function built into SciPy
        # (but which is tricky to find since it's not part of the public API).
        A = ba_sparsity(n_cameras, n_points, camera_indices, point_indices)
        num_jac = approx_derivative(fun_proxy, params, sparsity=A)
        show_delta = True

        # As of dec 13, I'm getting like 5-6 on this, so my Jacobian is
        # definitely NOT correct.
        acc = check_derivative(fun_proxy, jac_proxy, params)
        print("[jac][check] Checked jac acc: {:.8f}".format(acc))
        print("[jac][check] Lower than say 1e-6 means jac impl is likely OK.")

        if show_delta:
            print("[jac][check] Preparing plots...")
            # Work on a subset for simplicity's sake, since if there's a bug, it
            # will show up everywhere anyway.
            jac_plotlim = 150
            denseJ = J_csr[:jac_plotlim, :jac_plotlim].todense()
            num_jac = num_jac[:jac_plotlim, :jac_plotlim]
            err = denseJ - num_jac
            max_err = 100
            err[err > max_err] = max_err
            err[err < -max_err] = -max_err

            plt.figure()
            plot = plt.imshow(err)
            plt.title("Delta")
            plt.colorbar(plot)

            plt.figure()
            plot = plt.imshow(denseJ)
            plt.title("Analytic version")
            plt.colorbar(plot)

            plt.figure()
            plot = plt.imshow(num_jac.todense())
            plt.title("Numerical Jacobian")
            plt.colorbar(plot)

            plt.show()

        # plt.figure()
        # denseJ[J_csr.nonzero()] = 50
        # denseJ[denseJ != 0.0] = 100
        # plot = plt.imshow(denseJ) #, cmap=cm.viridis)

        # print("Patching analytic jacobian...")
        # Yes, this seems to solve our problems! (checked on entire 49-frame sequence)
        # J_csr[:, 0:n_cameras*cam_param_count] = num_jac[:, 0:n_cameras*cam_param_count]

        # print("Will show plots.")
        # plt.show()

        # FFS WTF MAN. So. SciPy can perform gradient checks, but not jacobian
        # checks, even though it CLEARLY estimates jacobians when doing
        # optimization. OH, perhaps that estimation part is deep inside LAPACK
        # and scipy hasnt wrapped it properly?
        # TODO(andrei): See how scipy does the numerical estimation of the
        # gradient.
        # print("Performing gradient check:")
        # TODO(andrei): If this doesn't work, use numdifftools!
        # Note: scipy does not support jacobian checking; we should use the
        # functionality from the pysfm project (finite_differences.py)
        # gradient_err = sopt.check_grad(fun, jac_clean, params,
        #     n_cameras, n_points, camera_indices, point_indices, points_2d, False
        # )
        # print("Estimated gradient error:", gradient_err)

        # raise ValueError("Try \"patching\" the jacobian, in that you keep the "
        #                  "analytic one, but replace the camera bits with the "
        #                  "numerical estimation, and see if you converge to the "
        #                  "same cost as the purely numerical method.")

    return J_csr


# Do NOT use these jacobian functions.
# TODO(andrei): Remove them once 'jac_clean' is confirmed to be correct!

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

