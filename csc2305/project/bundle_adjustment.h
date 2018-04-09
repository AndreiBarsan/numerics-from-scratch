#ifndef BUNDLEADJUSTMENT_H
#define BUNDLEADJUSTMENT_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

// TODO(andreib): Wrap this in a simple namespace, like ba.

using ceres::AngleAxisRotatePoint;
using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


// TODO(andreib): Basic header with info about WTF is going on.

template<typename T>
void transform_point(const T *camera, const T *point_world, bool reparametrized, T *result) {
  if (reparametrized) {
    // p = R'(p - t)
    // Translate first
    result[0] = point_world[0] - camera[3];
    result[1] = point_world[1] - camera[4];
    result[2] = point_world[2] - camera[5];

    // Then rotate (using the transpose of the rotation matrix)
    T neg_axis_angle[3];
    neg_axis_angle[0] = -camera[0];
    neg_axis_angle[1] = -camera[1];
    neg_axis_angle[2] = -camera[2];
    T p_cpy[3];
    p_cpy[0] = result[0];
    p_cpy[1] = result[1];
    p_cpy[2] = result[2];
    AngleAxisRotatePoint(neg_axis_angle, p_cpy, result);
  } else {
    // p = Rp + t
    // Rotate the point
    AngleAxisRotatePoint(camera, point_world, result);

    // Apply the translation
    result[0] += camera[3];
    result[1] += camera[4];
    result[2] += camera[5];
  }
}

template<typename T>
bool compute_reprojection_residual(const double observed_x,
                                   const double observed_y,
                                   const T *camera,
                                   const T *point_world,
                                   T *residuals,
                                   bool enable_radial,
                                   bool reparametrized) {
  // camera[0, 1, 2] represents the rotation (angle-axis);
  // Pc = R ( Pw + c )
  T p[3];
  transform_point(camera, point_world, reparametrized, p);

  // Apply the projection
  //   The sign change comes from the camera model assumed by the
  //   bundler tool for which this dataset was originally designed,
  //   which had a negative z-axis, that is, -z was in front of the
  //   camera.
  T x_center = -p[0] / p[2];
  T y_center = -p[1] / p[2];

  T distortion = T(1.0);
  if (enable_radial) {
    // Apply the radial distortion
    //  (the second and fourth order radial distortion)
    const T &l1 = camera[7];
    const T &l2 = camera[8];
    T r2 = x_center * x_center + y_center * y_center;
    T r4 = r2 * r2;
    distortion += r2 * l1 + r4 * l2;
  }

  // Compute the final projected point position
  // It seems that the system just assumes the principal point is zero.
  // This is just the convention used by the BAL dataset.
  const T &focal = camera[6];
  T predicted_x_2d = x_center * focal * distortion;
  T predicted_y_2d = y_center * focal * distortion;

  // TODO(andrei): What happens if we group these together into a distance, and then robustify?
  // Ceres seems to not be hindered by the fact that x and y are separate when adding
  // a robust estimator. Is the SciPy solver then weird, or does Ceres do anything special?
  residuals[0] = predicted_x_2d - T(observed_x);
  residuals[1] = predicted_y_2d - T(observed_y);

  return true;
}


struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y, bool enable_radial, bool reparametrized)
      : observed_x(observed_x),
        observed_y(observed_y),
        enable_radial(enable_radial),
        reparametrized(reparametrized) {}

  // Note that *this whole method is auto-diffable*, thanks to the 'AngleAxisRotatePoint' helper!
  template<typename T>
  bool operator()(const T *const camera, const T *const point, T *residuals) const {
    return compute_reprojection_residual(observed_x, observed_y, camera, point, residuals, enable_radial,
                                         reparametrized);
  }

  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y,
                                     bool enable_radial,
                                     bool reparametrize
  ) {
    // Two residuals and two param blocks, with 9 (camera) and 3 (3D point) params, respectively,
    // hence the 2, 9, 3 template args.
    return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y, enable_radial, reparametrize));
  }

  double observed_x;
  double observed_y;
  bool enable_radial;
  bool reparametrized;
};

#endif // BUNDLEADJUSTMENT_H
