#include "BALProblem.h"

#include <glog/logging.h>
#include <ceres/rotation.h>

#include <omp.h>

void BALProblem::reparametrize_cameras() {
  // TODO(andreib): Embarrassingly parallel. Do it with multiple threads; should be a nice easy tutorial...
  LOG(INFO) << "Performing camera reparametrization..." << std::endl;

#pragma omp parallel for default(shared)
  for (int i = 0; i < num_cameras_; ++i) {
    int cam_idx = i * kNumParamsPerCam;

    double t_raw[3];
    t_raw[0] = parameters_[cam_idx + 3];
    t_raw[1] = parameters_[cam_idx + 4];
    t_raw[2] = parameters_[cam_idx + 5];
    double aa_rot_new[3];
    aa_rot_new[0] = -parameters_[cam_idx + 0];
    aa_rot_new[1] = -parameters_[cam_idx + 1];
    aa_rot_new[2] = -parameters_[cam_idx + 2];

//            printf("Camera %d translation: %.4lf %.4lf %.4lf.\n", i, t_raw[0], t_raw[1], t_raw[2]);
//            printf("Camera %d rotation: %.4lf %.4lf %.4lf.\n", i,
//                   -aa_rot_new[0], -aa_rot_new[1], -aa_rot_new[2]);

    double t_rotated[3];
    ceres::AngleAxisRotatePoint(aa_rot_new, t_raw, t_rotated);

    // Write back the new rotation (R')
    parameters_[cam_idx + 0 + 0] = aa_rot_new[0];
    parameters_[cam_idx + 0 + 1] = aa_rot_new[1];
    parameters_[cam_idx + 0 + 2] = aa_rot_new[2];

    // Write back the new translation -(R' t)
    parameters_[cam_idx + 3 + 0] = -t_rotated[0];
    parameters_[cam_idx + 3 + 1] = -t_rotated[1];
    parameters_[cam_idx + 3 + 2] = -t_rotated[2];

    // The remaining parameters (f, k1, k2) remain unchanged.
  }
}