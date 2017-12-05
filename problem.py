"""Problem definitions to be used when running BA benchmarks."""

import bz2

import numpy as np


class Problem(object):
    # Should specify some problem like a toy function, a toy-but-ugly-function
    # like the Rosenbrock banana function, or simple nonlinear lsq problem, or a
    # bundle adjustment problem.
    #
    # Should be quite generic, since depending on the problem you may want to
    # call a different ceres program.

    def __init__(self, name):
        self.name = name

    def solve(self):
        # TODO(andrei): Use ABCs properly.
        raise ValueError("Cannot solve generic problem.")


class ToyProblem(Problem):
    pass


class SimpleLeastSquaresProblem(Problem):
    pass


class BundleAdjustmentProblem(Problem):
    def __init__(self, name):
        Problem.__init__(self, name)


class BALBundleAdjustmentProblem(BundleAdjustmentProblem):
    def __init__(self, name, data_fpath):
        BundleAdjustmentProblem.__init__(self, name)
        self.data_fpath = data_fpath
        self._load(self.data_fpath)

    def _load_vanilla(self, data_fpath):
        with open(data_fpath, "rt") as file:
            n_cameras, n_points, n_observations = map(
                int, file.readline().split())

            camera_indices = np.empty(n_observations, dtype=int)
            point_indices = np.empty(n_observations, dtype=int)
            points_2d = np.empty((n_observations, 2))

            for i in range(n_observations):
                camera_index, point_index, x, y = file.readline().split()
                camera_indices[i] = int(camera_index)
                point_indices[i] = int(point_index)
                points_2d[i] = [float(x), float(y)]

            camera_params = np.empty(n_cameras * 9)
            for i in range(n_cameras * 9):
                camera_params[i] = float(file.readline())
            camera_params = camera_params.reshape((n_cameras, -1))

            points_3d = np.empty(n_points * 3)
            for i in range(n_points * 3):
                points_3d[i] = float(file.readline())
            points_3d = points_3d.reshape((n_points, -1))

        # return camera_params, points_3d, camera_indices, point_indices, points_2d
        self.camera_params = camera_params
        self.points_3d = points_3d
        self.camera_indices = camera_indices
        self.point_indices = point_indices
        self.points_2d = points_2d

    def _load(self, data_fpath, max_frames=5):
        # TODO(andrei): Once you complete this, document it so you can share
        # it to other people who wish to test on subsets of BA!
        # Based on the code from: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
        if data_fpath.endswith("bz2"):
            opener = bz2.open
        else:
            opener = open

        with opener(data_fpath, "rt") as file:
            n_cameras, n_points, n_observations = map(
                int, file.readline().split())

            camera_indices = np.empty(n_observations, dtype=int)
            point_indices = np.empty(n_observations, dtype=int)
            points_2d = np.empty((n_observations, 2))

            i = 0
            seen_point_indexes = set()
            for line_idx in range(n_observations):
                camera_index, point_index, x, y = file.readline().split()
                if max_frames == -1 or int(camera_index) < max_frames:
                    camera_indices[i] = int(camera_index)
                    point_indices[i] = int(point_index)
                    points_2d[i] = [float(x), float(y)]
                    i += 1
                    seen_point_indexes.add(int(point_index))

            if max_frames != -1:
                camera_indices = camera_indices[:i]
                point_indices = point_indices[:i]
                points_2d = points_2d[:i]

            camera_params = np.empty(n_cameras * 9)
            for i in range(n_cameras * 9):
                camera_params[i] = float(file.readline())
            camera_params = camera_params.reshape((n_cameras, -1))

            n_real_points = len(seen_point_indexes)
            points_3d = np.empty(n_real_points * 3)
            points_3d_buf = np.empty(n_points * 3)

            print("Seen point count:  {}".format(n_real_points))
            print("Given point count: {}".format(n_points))

            point_3d_idx_map = {}

            # Only read in points which have been seen in at least one
            # observation. However, since the observations rely on 3D point
            # indices to represent matches, we need to remember the new
            # indices of the points we do keep, so we can properly update the
            # observations.
            for point_coord_idx in range(n_points * 3):
                # WHY is this part of the file suddenly one-coord-per-line???
                point = float(file.readline())
                points_3d_buf[point_coord_idx] = point

            points_3d_buf = points_3d_buf.reshape((n_points, -1))
            points_3d = points_3d.reshape((n_real_points, -1))

            idx = 0
            for point_idx in range(n_points):
                point = points_3d_buf[point_idx, :]
                if point_idx in seen_point_indexes:
                    point_3d_idx_map[point_idx] = idx
                    points_3d[idx, :] = point
                    idx += 1
                elif max_frames == -1:
                    # If we're not trimming frames, we expect to see every 3D
                    # point at least once!
                    print("Point idx {} NOT seen? WTF!!".format(point_idx))

            print(len(point_3d_idx_map))
            for i in range(len(point_indices)):
                if point_indices[i] in point_3d_idx_map:
                    print("Adjust 3D point index in observation {}: {} -> {}".format(
                        i, point_indices[i], point_3d_idx_map[point_indices[i]]
                    ))
                    point_indices[i] = point_3d_idx_map[point_indices[i]]


            print(camera_params.shape)
            print(max_frames)

            if max_frames != -1:
                camera_params = camera_params[:max_frames,:]

            print(camera_params.shape)

        self.camera_params = camera_params
        self.points_3d = points_3d
        self.camera_indices = camera_indices
        self.point_indices = point_indices
        self.points_2d = points_2d


class NISTBundleAdjustmentProblem(BundleAdjustmentProblem):
    def __init__(self):
        raise NotImplementedError()
